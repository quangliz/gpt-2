from configs.model_configs import MODEL_CONFIGS
from configs.training_configs import TRAINING_CONFIGS
from src.utils.logging import setup_logging, log_training_config, log_model_config
from src.utils.checkpoint import save_checkpoint, load_checkpoint, load_latest_checkpoint
from data.dataset import create_data_loaders
from src.training.trainer import train, evaluate
from src.model.gpt import GPTModel
import torch
import torch.optim as optim



def main():

    # customize your data
    with open("data/the-verdict.txt", "r", encoding="utf-8") as f:
        dataset_text = f.read()
    f.close()

    
    print("Hello from gpt-2!")
    logger = setup_logging(TRAINING_CONFIGS["log_dir"])
    log_training_config(logger, TRAINING_CONFIGS)
    log_model_config(logger, MODEL_CONFIGS)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = GPTModel(MODEL_CONFIGS).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=TRAINING_CONFIGS["lr"], weight_decay=TRAINING_CONFIGS["weight_decay"])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=TRAINING_CONFIGS["step_size"], gamma=TRAINING_CONFIGS["gamma"])
    train_loader, val_loader, test_loader = create_data_loaders(
        dataset_text,
        batch_size=TRAINING_CONFIGS["batch_size"],
        max_length=TRAINING_CONFIGS["max_length"],
        stride=TRAINING_CONFIGS["stride"],
        shuffle_train=TRAINING_CONFIGS["shuffle"],
        drop_last=TRAINING_CONFIGS["drop_last"],
        num_workers=TRAINING_CONFIGS["num_workers"],
    )

    # Optionally resume from latest checkpoint
    start_epoch = 0
    latest = load_latest_checkpoint(model, optimizer, scheduler)
    if latest != (None, None):
        last_epoch, last_loss = latest
        logger.info(f"Resumed from epoch {last_epoch} with val_loss {last_loss:.4f}")
        start_epoch = last_epoch + 1

    train(
        model,
        optimizer,
        scheduler,
        train_loader,
        val_loader,
        device,
        logger,
        TRAINING_CONFIGS["epochs"],
        checkpoint_dir="checkpoints",
        starting_epoch=start_epoch,
    )
    # Final evaluation on test set
    evaluate(model, test_loader, device, logger, desc="Test")


if __name__ == "__main__":
    main()
