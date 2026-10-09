import argparse
import json
import math
import os
import tempfile
from pathlib import Path

import torch

from src.data.basics.pdb_writer import write_complex_to_pdbs
from src.model.latent.checkpoint_utils import (
    build_file_identity,
)
from src.model.latent.latent_flow import LatentFlowModule, load_state_dict_with_policy
from src.utils.config_schema import (
    apply_paper_inference_protocol,
    resolve_flow_artifact_paths,
    validate_flow_config,
)
from src.utils.sampling_provenance import (
    resolve_sampling_device as _resolve_sampling_device,
)
from src.utils.sampling_provenance import (
    sampling_runtime_provenance as _shared_sampling_runtime_provenance,
)
from src.utils.simple_config import load_yaml_config
from src.utils.torch_runtime import configure_torch_runtime, seed_everything


SAMPLE_ARTIFACT_SCHEMA = "la_ribo.single_sample.v2"


def _canonical_sample_paths(output_path: str | Path) -> tuple[Path, Path, Path]:
    pdb_path = Path(output_path)
    if pdb_path.suffix != ".pdb":
        pdb_path = pdb_path.with_suffix(".pdb")
    summary_path = pdb_path.with_suffix(".summary.json")
    lock_path = pdb_path.with_name(f".{pdb_path.name}.sampling.lock")
    return pdb_path, summary_path, lock_path


def _repository_identity() -> dict[str, object]:
    repo_root = Path(__file__).resolve().parent
    return {"path": str(repo_root)}


def _resolved_sampling_protocol(
    model: LatentFlowModule,
    *,
    num_res: int,
) -> dict[str, object]:
    schedules: dict[str, dict[str, object]] = {}
    for modality in model.flow_matcher.modalities:
        spec = getattr(model.sampling_schedule_conf, modality, None)
        kind = str(getattr(spec, "kind", "linear")).lower()
        if kind not in {"linear", "uniform", "power"}:
            raise ValueError(
                f"Unsupported sampling schedule kind for {modality}: {kind}"
            )
        schedule: dict[str, object] = {"kind": kind}
        if kind == "power":
            power = float(getattr(spec, "power", 1.0))
            if not math.isfinite(power) or power <= 0.0:
                raise ValueError(
                    f"Sampling schedule power for {modality} must be finite and positive"
                )
            schedule["power"] = power
        schedules[modality] = schedule
    return {
        "num_res": num_res,
        "num_timesteps": model.flow_matcher.num_timesteps,
        "schedules": schedules,
    }


def _acquire_sample_output_lock(
    output_path: Path,
    summary_path: Path,
    lock_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    collisions = [
        path for path in (output_path, summary_path, lock_path) if path.exists()
    ]
    if collisions:
        raise FileExistsError(f"Refusing to overwrite sampling artifacts: {collisions}")
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError as exc:
        raise FileExistsError(
            f"Sampling output is already reserved by another process: {lock_path}"
        ) from exc
    with os.fdopen(lock_fd, "w") as lock_handle:
        lock_handle.write(
            f"pid={os.getpid()}\noutput={output_path.resolve()}\n"
            f"summary={summary_path.resolve()}\n"
        )


def _temporary_path(parent: Path, *, prefix: str, suffix: str = "") -> Path:
    with tempfile.NamedTemporaryFile(
        dir=parent,
        prefix=prefix,
        suffix=suffix,
        delete=False,
    ) as handle:
        temporary_path = Path(handle.name)
    temporary_path.chmod(0o644)
    return temporary_path


def _publish_no_overwrite(temporary_path: Path, final_path: Path) -> None:
    try:
        os.link(temporary_path, final_path)
    except FileExistsError as exc:
        raise FileExistsError(
            f"Refusing to overwrite sampling artifact: {final_path}"
        ) from exc


def main(
    config_path: str,
    checkpoint_path: str,
    num_res: int,
    output_path: str,
    autoencoder_checkpoint: str | None = None,
    latent_stats: str | None = None,
    coarse_stats: str | None = None,
    seed: int | None = None,
    allow_device_fallback: bool = False,
    device_override: str | None = None,
) -> None:
    if seed is None or not 0 <= seed <= 2**32 - 1:
        raise ValueError("Sampling seed must be set explicitly in [0, 2**32 - 1]")
    if num_res <= 0:
        raise ValueError("num_res must be positive")
    output_path, summary_path, lock_path = _canonical_sample_paths(output_path)
    collisions = [
        path for path in (output_path, summary_path, lock_path) if path.exists()
    ]
    if collisions:
        raise FileExistsError(f"Refusing to overwrite sampling artifacts: {collisions}")
    seed_everything(seed)
    config_identity = build_file_identity(config_path)
    checkpoint_identity = build_file_identity(checkpoint_path)
    cfg = load_yaml_config(config_path)
    apply_paper_inference_protocol(cfg)
    if device_override is not None:
        cfg.runtime_conf.device = device_override
    validate_flow_config(cfg)
    resolve_flow_artifact_paths(
        cfg,
        autoencoder_checkpoint=autoencoder_checkpoint,
        latent_stats=latent_stats,
        coarse_stats=coarse_stats,
    )
    requested_device = str(getattr(cfg.runtime_conf, "device", "cpu"))
    device, device_fallback_used = _resolve_sampling_device(
        requested_device,
        allow_device_fallback=allow_device_fallback,
    )
    configure_torch_runtime(device)

    source_artifacts = {
        "autoencoder_checkpoint": build_file_identity(
            cfg.flow_module_conf.autoencoder_ckpt_path
        ),
        "latent_stats": build_file_identity(
            cfg.flow_module_conf.latent_preprocess.stats_path
        ),
        "coarse_stats": build_file_identity(
            cfg.flow_module_conf.coarse_preprocess.stats_path
        ),
    }
    model = LatentFlowModule(cfg.flow_module_conf).to(device)
    current_source_artifacts = {
        "autoencoder_checkpoint": build_file_identity(
            cfg.flow_module_conf.autoencoder_ckpt_path
        ),
        "latent_stats": build_file_identity(
            cfg.flow_module_conf.latent_preprocess.stats_path
        ),
        "coarse_stats": build_file_identity(
            cfg.flow_module_conf.coarse_preprocess.stats_path
        ),
    }
    if current_source_artifacts != source_artifacts:
        raise RuntimeError(
            "Flow source artifacts changed while constructing LatentFlowModule"
        )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"latent-flow checkpoint {checkpoint_path} is not a mapping")
    weight_provenance = {"kind": "ema", "source": "checkpoint_state_dict"}
    model.validate_checkpoint(
        checkpoint,
        context=f"latent-flow checkpoint {checkpoint_path}",
    )
    state_dict = checkpoint.get("model", checkpoint)
    state_dict = {
        key: value
        for key, value in state_dict.items()
        if not key.startswith("autoencoder.encoder.")
    }
    load_state_dict_with_policy(
        model,
        state_dict,
        context=f"latent-flow checkpoint {checkpoint_path}",
    )
    model.eval()
    sampling_protocol = _resolved_sampling_protocol(
        model,
        num_res=num_res,
    )
    sampling_runtime = _sampling_runtime_provenance(
        requested_device=requested_device,
        resolved_device=device,
        device_fallback_used=device_fallback_used,
    )
    _acquire_sample_output_lock(output_path, summary_path, lock_path)
    seed_everything(seed)

    with torch.no_grad():
        sample = model.sample(
            num_batch=1,
            num_res=num_res,
            num_timesteps=int(sampling_protocol["num_timesteps"]),
            device=device,
        )

    atom37 = sample["pred_atom37"][0].detach().cpu().numpy()
    atom37_mask = sample["atom_mask"][0].detach().cpu().numpy()
    restype = sample["pred_aatypes"][0].detach().cpu().numpy()
    summary = {"artifact_schema": SAMPLE_ARTIFACT_SCHEMA}
    summary["sampling_seed"] = seed
    summary["sampling"] = {**sampling_protocol, "seed": seed, "selection": "direct"}
    final_config_identity = build_file_identity(config_path)
    final_checkpoint_identity = build_file_identity(checkpoint_path)
    final_source_artifacts = {
        "autoencoder_checkpoint": build_file_identity(
            cfg.flow_module_conf.autoencoder_ckpt_path
        ),
        "latent_stats": build_file_identity(
            cfg.flow_module_conf.latent_preprocess.stats_path
        ),
        "coarse_stats": build_file_identity(
            cfg.flow_module_conf.coarse_preprocess.stats_path
        ),
    }
    if final_config_identity != config_identity:
        raise RuntimeError("Config changed while sampling was running")
    if final_checkpoint_identity != checkpoint_identity:
        raise RuntimeError("Checkpoint changed while sampling was running")
    if final_source_artifacts != source_artifacts:
        raise RuntimeError("Flow source artifacts changed while sampling was running")

    temporary_pdb = _temporary_path(
        output_path.parent,
        prefix=f".{output_path.name}.tmp.",
        suffix=".pdb",
    )
    temporary_summary = _temporary_path(
        summary_path.parent,
        prefix=f".{summary_path.name}.tmp.",
    )
    pdb_published = False
    summary_published = False
    try:
        write_complex_to_pdbs(
            atom37,
            temporary_pdb,
            restype=restype,
            atom37_mask=atom37_mask,
            is_na_residue_mask=torch.ones(num_res).numpy(),
        )
        generated_sample = build_file_identity(temporary_pdb)
        generated_sample["path"] = str(output_path.resolve())
        summary["provenance"] = {
            "config": final_config_identity,
            "checkpoint": final_checkpoint_identity,
            **final_source_artifacts,
            "weights": weight_provenance["kind"],
            "weight_provenance": weight_provenance,
            "generated_sample": generated_sample,
        }

        if build_file_identity(config_path) != config_identity:
            raise RuntimeError("Config changed before publishing the sample")
        if build_file_identity(checkpoint_path) != checkpoint_identity:
            raise RuntimeError("Checkpoint changed before publishing the sample")
        if {
            "autoencoder_checkpoint": build_file_identity(
                cfg.flow_module_conf.autoencoder_ckpt_path
            ),
            "latent_stats": build_file_identity(
                cfg.flow_module_conf.latent_preprocess.stats_path
            ),
            "coarse_stats": build_file_identity(
                cfg.flow_module_conf.coarse_preprocess.stats_path
            ),
        } != source_artifacts:
            raise RuntimeError(
                "Flow source artifacts changed before publishing the sample"
            )
        final_sampling_runtime = _sampling_runtime_provenance(
            requested_device=requested_device,
            resolved_device=device,
            device_fallback_used=device_fallback_used,
        )
        if final_sampling_runtime != sampling_runtime:
            raise RuntimeError(
                "Sampling runtime or code changed before publishing the sample"
            )
        summary["provenance"].update(
            {
                "allow_device_fallback": allow_device_fallback,
                "sampling_runtime": final_sampling_runtime,
            }
        )

        temporary_summary.write_text(
            json.dumps(summary, allow_nan=False, indent=2, sort_keys=True)
        )
        _publish_no_overwrite(temporary_pdb, output_path)
        pdb_published = True
        _publish_no_overwrite(temporary_summary, summary_path)
        summary_published = True
    finally:
        temporary_pdb.unlink(missing_ok=True)
        temporary_summary.unlink(missing_ok=True)
        if not pdb_published or summary_published:
            lock_path.unlink(missing_ok=True)
    print(f"Saved sample to {output_path}")
    print(f"Saved sampling provenance to {summary_path}")


def _sampling_runtime_provenance(
    *,
    requested_device: str,
    resolved_device: torch.device,
    device_fallback_used: bool,
) -> dict[str, object]:
    return _shared_sampling_runtime_provenance(
        entrypoint="sample_latent_flow.py",
        repository_identity=_repository_identity(),
        requested_device=requested_device,
        resolved_device=resolved_device,
        device_fallback_used=device_fallback_used,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/la-ribo-tri.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num_res", type=int, required=True)
    parser.add_argument("--output", default="samples/latent_rna_sample.pdb")
    parser.add_argument("--autoencoder_checkpoint", default=None)
    parser.add_argument("--latent_stats", default=None)
    parser.add_argument("--coarse_stats", default=None)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default=None)
    parser.add_argument(
        "--allow_device_fallback",
        action="store_true",
        help=(
            "Allow a requested CUDA device to fall back to CPU. "
            "The resulting summary is not publication-eligible."
        ),
    )
    args = parser.parse_args()
    main(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        num_res=args.num_res,
        output_path=args.output,
        autoencoder_checkpoint=args.autoencoder_checkpoint,
        latent_stats=args.latent_stats,
        coarse_stats=args.coarse_stats,
        seed=args.seed,
        allow_device_fallback=args.allow_device_fallback,
        device_override=args.device,
    )
