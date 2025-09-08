TRAINING_CONFIGS = {
    "epochs": 10,
    "lr": 3e-4,
    "step_size": 1000,
    "gamma": 0.95,
    "batch_size": 4,
    "max_length": 256,
    "stride": 128,
    "shuffle": True,
    "drop_last": True,
    "num_workers": 0,
    # "dataset_text": ("The quick brown fox jumps over the lazy dog. " * 4000),
    "log_dir": "logs",
    "weight_decay": 0.01
}