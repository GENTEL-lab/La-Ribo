


from typing import TYPE_CHECKING


__all__ = ["LatentFlowModel", "LatentFlowModule", "RNAPartialAutoencoder"]

_LAZY_EXPORTS = {
    "RNAPartialAutoencoder": "src.model.latent.rna_autoencoder",
    "LatentFlowModel": "src.model.latent.latent_flow",
    "LatentFlowModule": "src.model.latent.latent_flow",
}

if TYPE_CHECKING:
    from src.model.latent.latent_flow import LatentFlowModel, LatentFlowModule
    from src.model.latent.rna_autoencoder import RNAPartialAutoencoder


def __getattr__(name: str) -> object:
    module_path = _LAZY_EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(module_path), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY_EXPORTS})
