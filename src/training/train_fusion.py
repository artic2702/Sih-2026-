"""
Training entry point for the Optical-SAR Fusion model (Person 4).

Usage:
    python -m src.training.train_fusion --config configs/config.yaml
    python -m src.training.train_fusion --config configs/config.yaml --epochs 5

FIXES vs. the original template (see AUDIT_STATUS.md "Not yet done"):
  1. get_dataloader() now called WITH dataset="bigearthnet_mm" (the template
     omitted `dataset=`, which raises ValueError immediately against the
     current dataset_loader.py).
  2. model.load() is now actually called before training, so the real
     nn.Module encoders + fusion head exist (the template built a bare
     FusionModel() and tried to train nothing).
  3. Real validation loop each epoch: computes multilabel_f1 via
     src/evaluation/metrics.py, drives EarlyStopper, saves a "best"
     checkpoint on improvement.
  4. CHECKPOINT SHAPE FIX (important, found while wiring this up): the
     shared trainer_utils.save_checkpoint()/load_checkpoint() route through
     BaseRSModel.torch_module(), which for FusionModel returns an
     nn.ModuleDict bundling optical_encoder + sar_encoder + fusion_head
     together. But FusionModel.load() only ever restores
     `self.fusion_head.load_state_dict(...)` — a state-dict shaped for the
     ModuleDict (keys like "fusion_head.net.0.weight") does NOT match what
     fusion_head.load_state_dict() expects (keys like "net.0.weight") and
     will raise a key-mismatch error. Since encoders are frozen by default
     this sprint (only the fusion head trains), we deliberately save/load
     ONLY the fusion_head's own state dict here — matching FusionModel.load()
     exactly — rather than editing the shared torch_module()/load()
     contract (which other people's models also depend on). If
     freeze_encoders is ever flipped to fine-tune the encoders too, this
     needs revisiting together as a team (see docs/DECISIONS.md).
"""

import argparse
import datetime
import json
import os

import numpy as np
import torch
import yaml

from src.models.fusion_model import FusionModel, train_step
from src.preprocessing.dataset_loader import get_dataloader
from src.training.trainer_utils import (
    set_seed, get_device, build_optimizer, get_git_commit,
    SimpleLogger, EarlyStopper,
)
from src.evaluation.metrics import multilabel_f1

DATASET_NAME = "bigearthnet_mm"


def _labels_to_multihot(label_lists, class_names):
    """batch["label"] is List[List[str]] (multi-label class names per
    sample) — convert to an (N, K) binary array against the model's fixed
    label order. Mirrors the equivalent inline logic in
    fusion_model.py::train_step, duplicated here (not imported) since it's
    a 4-line helper and importing it would mean reaching into another
    function's internals rather than the public train_step contract.
    """
    out = np.zeros((len(label_lists), len(class_names)), dtype=np.float32)
    for i, labels in enumerate(label_lists):
        for lbl in labels:
            if lbl in class_names:
                out[i, class_names.index(lbl)] = 1.0
    return out


def save_fusion_checkpoint(model: FusionModel, path: str, epoch: int,
                            config: dict, is_best: bool = False) -> dict:
    """Save ONLY the fusion_head's state dict (see module docstring for
    why). Mirrors trainer_utils.save_checkpoint's metadata sidecar
    convention so tooling/humans reading .meta.json get the same shape
    regardless of which train_*.py wrote it.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    state = {"model_state_dict": model.fusion_head.state_dict(), "epoch": epoch}
    torch.save(state, path)

    metadata = {
        "model": model.name,
        "backbone": model.backbone,
        "task": model.task,
        "dataset": DATASET_NAME,
        "dataset_split": "train",
        "epoch": epoch,
        "encoder_weights": model.encoder_weights_status,
        "freeze_encoders": config["models"]["fusion"].get("freeze_encoders", True),
        "checkpoint_scope": "fusion_head_only",  # see module docstring
        "git_commit": get_git_commit(),
        "date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    with open(path + ".meta.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    if is_best:
        torch.save(state, path + ".best")
        with open(path + ".best.meta.json", "w") as f:
            json.dump(metadata, f, indent=2, default=str)

    return metadata


@torch.no_grad()
def evaluate_epoch(model: FusionModel, val_loader) -> dict:
    """Full-fusion (optical+SAR) validation metric, used for early stopping
    / best-checkpoint selection during training. The mandatory 3-way
    ablation (optical-only / SAR-only / fusion) is a separate, deliberately
    more thorough pass — see scripts/run_fusion_ablation.py — run once
    after training finishes, not every epoch (too slow with 3x forward
    passes per sample for a training-loop metric).
    """
    model.fusion_head.eval()
    all_probs, all_gold = [], []
    for batch in val_loader:
        optical = batch["image_optical"].to(model.device)
        sar = batch["image_sar"].to(model.device)
        optical_feat = model.optical_encoder(optical)
        sar_feat = model.sar_encoder(sar)
        logits = model.fusion_head(optical_feat, sar_feat)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_gold.append(_labels_to_multihot(batch["label"], model.labels))
    pred_probs = np.concatenate(all_probs, axis=0)
    gold_labels = np.concatenate(all_gold, axis=0)
    return multilabel_f1(pred_probs, gold_labels)


def main(config_path: str, epochs_override: int = None):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    set_seed(config["training"]["seed"])
    device = get_device(config["training"]["device"])
    epochs = epochs_override or config["training"]["epochs"]
    batch_size = config["training"]["batch_size"]

    print(f"[train_fusion] device={device} epochs={epochs} batch_size={batch_size}")

    train_loader = get_dataloader(task="fusion", split="train", dataset=DATASET_NAME,
                                   config=config, batch_size=batch_size)
    val_loader = get_dataloader(task="fusion", split="val", dataset=DATASET_NAME,
                                 config=config, batch_size=batch_size)
    print(f"[train_fusion] train samples={len(train_loader.dataset)}  "
          f"val samples={len(val_loader.dataset)}")

    model = FusionModel(config=config)
    checkpoint_path = config["models"]["fusion"]["checkpoint"]
    # load() builds the encoders (frozen by default) + a freshly-initialized
    # fusion head. A "no checkpoint found" print here is expected on a
    # first-ever run — that's exactly the state we're about to train from.
    model.load(checkpoint_path, device=device)
    model.device = device

    optimizer = build_optimizer(model, config)
    logger = SimpleLogger(log_file="models/checkpoints/fusion_train.log")
    stopper = EarlyStopper(patience=4, mode="max")  # tracks macro_f1

    step = 0
    for epoch in range(epochs):
        for batch in train_loader:
            loss = train_step(model, batch, optimizer, config)
            if step % config["training"]["log_every"] == 0:
                logger.log(step, {"loss": loss})
            step += 1

        val_metrics = evaluate_epoch(model, val_loader)
        macro_f1 = val_metrics["macro_f1"]
        print(f"[epoch {epoch}] val_macro_f1={macro_f1:.4f} val_micro_f1={val_metrics['micro_f1']:.4f}")

        is_best = stopper.step(macro_f1)
        save_fusion_checkpoint(model, checkpoint_path, epoch, config, is_best=is_best)
        # Keep the primary model.pt as the BEST checkpoint, because the
        # Streamlit/React integration should never accidentally load a
        # worse final epoch just because it ran after the best epoch.
        if is_best:
            best_path = checkpoint_path + ".best"
            if os.path.exists(best_path):
                import shutil
                shutil.copy2(best_path, checkpoint_path)
        # Since checkpoint_loaded gates status="success" vs "low_confidence"
        # in predict() (see fusion_model.py::_result_from_probs), flip it
        # true the moment we've saved a real trained checkpoint at least
        # once, so predict() calls after training report status correctly.
        model._checkpoint_loaded = True

        if stopper.should_stop:
            print(f"[train_fusion] early stopping at epoch {epoch} "
                  f"(no improvement for {stopper.patience} epochs, best={stopper.best:.4f})")
            break

    print(f"Training complete. Best val_macro_f1={stopper.best:.4f}. "
          f"Checkpoint: {checkpoint_path} (+ .best variant).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--epochs", type=int, default=None,
                         help="override configs/config.yaml training.epochs "
                              "(e.g. --epochs 2 for a quick smoke run)")
    args = parser.parse_args()
    main(args.config, epochs_override=args.epochs)
