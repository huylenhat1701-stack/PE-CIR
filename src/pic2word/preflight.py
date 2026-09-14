"""Check training inputs before allocating GPU memory or downloading CLIP."""
from typing import Any

from pic2word.data.cc3m import CC3MImageDataset


def validate_training_inputs(config: dict[str, Any]) -> dict[str, int]:
    train, data = config["training"], config["data"]
    batch_size = int(train["batch_size_per_device"])
    if batch_size < 2:
        raise ValueError("Contrastive training requires batch_size_per_device >= 2")
    for key in ("epochs", "save_every_epochs"):
        if int(train[key]) <= 0:
            raise ValueError(f"{key} must be positive")
    if train.get("max_steps") is not None and int(train["max_steps"]) <= 0:
        raise ValueError("max_steps must be positive")
    if int(train.get("save_every_steps", 0)) < 0:
        raise ValueError("save_every_steps cannot be negative")
    if int(train["num_workers"]) < 0:
        raise ValueError("num_workers cannot be negative")
    if config["model"]["prompt"].count("*") != 1:
        raise ValueError("Training prompt must contain exactly one *")
    dataset = CC3MImageDataset(
        data["train_manifest"], data["image_root"], None,
        image_column=data.get("image_column"), max_samples=data.get("max_samples"),
    )
    missing = [str(path) for path in dataset.image_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} training images missing; first: {missing[0]}")
    if len(set(dataset.image_paths)) != len(dataset):
        raise ValueError("Duplicate image paths in training manifest create false negatives")
    if len(dataset) < batch_size:
        raise ValueError("Dataset is smaller than batch size; drop_last would produce zero steps")
    return {"samples": len(dataset), "batches_per_epoch": len(dataset) // batch_size}
