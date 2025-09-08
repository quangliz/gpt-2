import os
import glob
import torch


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_checkpoint(model, optimizer, scheduler, epoch, loss, path=None):
    # create a default path if no path is provided
    if path is None:
        ensure_dir("checkpoints")
        path = f"checkpoints/checkpoint_{epoch}.pth"
    else:
        ensure_dir(os.path.dirname(path) or ".")
    
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "loss": loss
    }, path)


def load_checkpoint(model, optimizer, scheduler, path):
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    return checkpoint["epoch"], checkpoint["loss"]


def load_latest_checkpoint(model, optimizer, scheduler, directory="checkpoints"):
    ensure_dir(directory)
    pattern = os.path.join(directory, "checkpoint_*.pth")
    files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    if not files:
        return None, None
    # Try loading from newest to oldest; skip corrupted/incomplete checkpoints
    for path in files:
        try:
            return load_checkpoint(model, optimizer, scheduler, path)
        except Exception:
            continue
    return None, None