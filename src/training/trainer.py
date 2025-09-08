import torch
import torch.nn as nn
import tqdm.auto as tqdm
from src.utils.checkpoint import save_checkpoint

def compute_cross_entropy_loss(logits, targets):
    vocab_size = logits.size(-1)
    return nn.functional.cross_entropy(
        logits.view(-1, vocab_size),
        targets.view(-1)
    )


def train(model, optimizer, scheduler, train_loader, val_loader, device, logger, epochs, checkpoint_dir="checkpoints", starting_epoch=0):
    best_val_loss = float("inf")
    global_step = 0
    for epoch_idx in range(epochs):
        epoch = starting_epoch + epoch_idx
        model.train()
        running_loss = 0.0
        pbar = tqdm.tqdm(train_loader, desc=f"Train {epoch}", leave=False)
        for batch_idx, (input_ids, target_ids) in enumerate(pbar):
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            optimizer.zero_grad()
            logits = model(input_ids)
            loss = compute_cross_entropy_loss(logits, target_ids)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            running_loss += loss.item()
            global_step += 1
            lr = scheduler.get_last_lr()[0] if scheduler is not None else None
            avg_loss = running_loss / (batch_idx + 1)
            pbar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{avg_loss:.4f}", lr=(f"{lr:.2e}" if lr is not None else "n/a"))
        train_avg = running_loss / max(1, len(train_loader))

        # Validation phase
        val_loss = evaluate(model, val_loader, device, logger, desc=f"Val  {epoch}")

        # Log epoch summary
        logger.info(f"Epoch {epoch} - train_loss: {train_avg:.4f} - val_loss: {val_loss:.4f}")

        # Save per-epoch checkpoint and best checkpoint
        save_checkpoint(model, optimizer, scheduler, epoch, val_loss, path=f"{checkpoint_dir}/checkpoint_{epoch}.pth")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, scheduler, epoch, val_loss, path=f"{checkpoint_dir}/checkpoint_best.pth")
    return model


def evaluate(model, dataloader, device, logger=None, desc="Eval"):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    with torch.no_grad():
        pbar = tqdm.tqdm(dataloader, desc=desc, leave=False)
        for input_ids, target_ids in pbar:
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            logits = model(input_ids)
            loss = compute_cross_entropy_loss(logits, target_ids)
            total_loss += loss.item()
            num_batches += 1
            avg = total_loss / num_batches
            pbar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{avg:.4f}")
    if num_batches == 0:
        if logger is not None:
            logger.warning(f"[{desc}] No batches in dataloader; returning NaN loss")
        return float("nan")
    avg_loss = total_loss / num_batches
    if logger is not None:
        logger.info(f"[{desc}] avg_loss: {avg_loss:.4f}")
    return avg_loss


if __name__ == "__main__":
    pass