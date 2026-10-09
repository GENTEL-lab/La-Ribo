# La-Ribo: RNA Co-Design via Geometry-Latent Flow Matching

[![arXiv](https://img.shields.io/badge/arXiv-2610.12236-b31b1b.svg)](https://arxiv.org/abs/2610.12236)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-GENTEL--Lab%2FLa--Ribo-yellow)](https://huggingface.co/GENTEL-Lab/La-Ribo)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Official inference code for **La-Ribo** ([paper](https://arxiv.org/pdf/2610.12236)).

La-Ribo generates RNA sequences and heavy-atom structures jointly. A flow-matching model samples a geometry–latent representation: a sparse backbone of phosphate, sugar and base positions, plus per-nucleotide latent vectors that encode identity and local shape. A scaffold-conditioned autoencoder then decodes this representation into a sequence and a full-atom structure.

## Model variants

| Variant | Config | Difference |
|---|---|---|
| La-Ribo-Base | `config/la-ribo-base.yaml` | No pair updates in the flow model |
| La-Ribo-Tri | `config/la-ribo-tri.yaml` | Triangle pair updates every 2 layers (cuEquivariance kernels) |

Both variants use the same latent dimension (16) and a 12-layer flow model.

## Installation

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then clone the inference repository and install the locked environment:

```bash
git clone https://github.com/GENTEL-lab/La-Ribo.git
cd La-Ribo
uv python install 3.14
uv sync --locked
```

Run the remaining commands from this repository directory. Python 3.14+ is required. Supported inference devices are CPU (including macOS) and NVIDIA CUDA on Linux. CUDA inference requires a driver compatible with CUDA 13. The default device selects CUDA when available and CPU otherwise.

## Download pretrained models

The pretrained model bundle is available in the [Hugging Face repository](https://huggingface.co/GENTEL-Lab/La-Ribo/tree/main/weights). Download the five files below from `weights/`.

| File | Purpose |
|---|---|
| `weights/la_ribo_base.pt` | Base flow model, exported EMA weights |
| `weights/la_ribo_tri.pt` | Tri flow model, exported EMA weights |
| `weights/la_ribo_vae.pt` | Shared autoencoder weights |
| `weights/latent_stats.pt` | Latent whitening statistics |
| `weights/coarse_stats.pt` | Scaffold coordinate scaling statistics |

The three model files are pure tensor state dictionaries. Base and Tri already contain EMA weights, so no `--use_ema` flag is needed. Both variants share the same autoencoder and statistics. Download the statistics with the weights; they are required to reconstruct coordinates at the correct scale.

```bash
uvx --from huggingface_hub hf download GENTEL-Lab/La-Ribo \
  weights/la_ribo_base.pt \
  weights/la_ribo_tri.pt \
  weights/la_ribo_vae.pt \
  weights/latent_stats.pt \
  weights/coarse_stats.pt \
  --local-dir .
```

This creates the following local layout:

```text
La-Ribo/
├── config/
├── sample_latent_flow.py
└── weights/
    ├── la_ribo_base.pt
    ├── la_ribo_tri.pt
    ├── la_ribo_vae.pt
    ├── latent_stats.pt
    └── coarse_stats.pt
```

For a public release, downloading does not require login. To download a private or gated release, first run `uvx --from huggingface_hub hf auth login` with an account that has access. See the [Hugging Face CLI documentation](https://huggingface.co/docs/huggingface_hub/guides/cli) for additional download options.

## Generate an RNA structure

The default protocol uses 200 Euler steps with scaffold power 1.25 and latent power 2.0 for both variants. Each command below generates one 80-nucleotide RNA sequence and heavy-atom structure.

### La-Ribo-Base

```bash
uv run --no-sync python sample_latent_flow.py \
  --config config/la-ribo-base.yaml \
  --checkpoint weights/la_ribo_base.pt \
  --autoencoder_checkpoint weights/la_ribo_vae.pt \
  --latent_stats weights/latent_stats.pt \
  --coarse_stats weights/coarse_stats.pt \
  --num_res 80 \
  --seed 2027 \
  --device auto \
  --output samples/base_80_seed2027.pdb
```

### La-Ribo-Tri

```bash
uv run --no-sync python sample_latent_flow.py \
  --config config/la-ribo-tri.yaml \
  --checkpoint weights/la_ribo_tri.pt \
  --autoencoder_checkpoint weights/la_ribo_vae.pt \
  --latent_stats weights/latent_stats.pt \
  --coarse_stats weights/coarse_stats.pt \
  --num_res 80 \
  --seed 2027 \
  --device auto \
  --output samples/tri_80_seed2027.pdb
```

Use the matching config and flow weights for each variant. Set `--device cpu` to select CPU explicitly, or `--device cuda` to require a CUDA GPU. To select a particular NVIDIA GPU, prefix the command with `CUDA_VISIBLE_DEVICES=0`, replacing `0` with its index.

Change `--num_res` for the RNA length and `--seed` for another sample. The paper's unconditional-generation benchmark covers 40–150 nucleotides in increments of ten. A matching seed alone does not guarantee identical CPU/CUDA coordinates because the backends use different random-number implementations.

### Outputs

For the Base example, the sampler creates:

- `samples/base_80_seed2027.pdb`: generated nucleotide identities and heavy-atom coordinates.
- `samples/base_80_seed2027.summary.json`: sampling parameters, device, input-file identities, and output provenance.

The output directory is created automatically. Existing output files are not overwritten: use a new output filename when sampling again. The summary contains local paths and runtime details; review it before sharing it publicly.

| Argument | Description |
|---|---|
| `--config` | Variant config; default: `config/la-ribo-tri.yaml` |
| `--checkpoint` | Matching Base or Tri flow weights |
| `--autoencoder_checkpoint` | Shared `la_ribo_vae.pt` |
| `--latent_stats` | Matching latent whitening statistics |
| `--coarse_stats` | Matching scaffold scaling statistics |
| `--num_res` | Number of nucleotides; required |
| `--seed` | Sampling seed; required |
| `--device` | `auto`, `cuda`, or `cpu` |
| `--output` | Output PDB path; default: `samples/latent_rna_sample.pdb` |
| `--allow_device_fallback` | Explicitly allow an unavailable requested CUDA device to fall back to CPU |

These commands perform sampling only. They do not run inverse folding, refolding, or the paper's designability evaluation.

## Optional dataset download

The training dataset is not required for inference. The [Hugging Face repository](https://huggingface.co/GENTEL-Lab/La-Ribo) also hosts the data released with the paper:

- `data/metadata.csv`: one row per RNA (`name`, `source_db`, `length`, `mean_plddt`, `sequence`, ...).
- `data/trran_data_publish.tar.gz`: trRosettaRNA2-predicted structures. Each entry holds `model_1_relaxed200.pdb` and a per-residue `plddt.csv`.

```bash
uvx --from huggingface_hub hf download GENTEL-Lab/La-Ribo \
  --include "data/*" --local-dir .
tar -xzf data/trran_data_publish.tar.gz -C data
```

## Citation

```bibtex
@article{ma2026laribo,
  title         = {La-Ribo: RNA Co-Design via Geometry-Latent Flow Matching},
  author        = {Ma, Runze and Hua, Will and Zheng, Shuangjia},
  year          = {2026},
  eprint        = {2610.12236},
  archivePrefix = {arXiv},
  primaryClass  = {q-bio.BM},
  url           = {https://arxiv.org/abs/2610.12236}
}
```

## License

Released under the [MIT License](LICENSE).
