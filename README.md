## GPT-2 Implementation

A GPT-2 you can train on raw text, evaluate with train/val/test splits, save/resume checkpoints, and run inference from the CLI.

### Quick start

1) Install Python 3.12+ and dependencies

```bash
# install uv if havent yet
pip install uv

# sync
uv sync
```

2) Put your training text into `data/the-verdict.txt` (or replace with your own file and update `main.py` to read it).

3) Train (auto-resume if checkpoints exist)

```bash
python main.py
```

4) Inference (after/without training, load a checkpoint and generate text)

```bash
python -m src.inference.generate \
  --prompt "Once upon a time" \
  --ckpt checkpoints/checkpoint_best.pth \
  --max_new_tokens 80 \
  --temperature 0.9 \
  --top_k 50
```

### What’s inside

- `src/model/` — GPT-2 components (`layer.py`, `transformer.py`, `gpt.py`). `GPTModel.generate(...)` for autoregressive sampling.
- `data/dataset.py` — Tokenizes with `tiktoken`, builds fixed-length sliding-window samples, and exposes:
  - `create_data_loaders(...)` → returns train/val/test `DataLoader`s.
- `src/training/trainer.py` — Training loop with tqdm progress bars, per-epoch validation, and average loss reporting. Saves `checkpoints/checkpoint_{epoch}.pth` and `checkpoints/checkpoint_best.pth`.
- `src/utils/checkpoint.py` — Robust resume: skips corrupted/incomplete checkpoints and loads the newest valid one.
- `src/inference/generate.py` — CLI for loading a checkpoint and generating text.
- Logging to `logs/log.txt` via `src/utils/logging.py`.

### Configuration

- Edit `configs/training_configs.py`:
  - `epochs`, `lr`, `step_size`, `gamma`, `batch_size`, `max_length`, `stride`, `shuffle`, `drop_last`, `num_workers`, `log_dir`, `weight_decay`.
  - Optimizer is AdamW in `main.py`, using `weight_decay` from this file.
- Edit `configs/model_configs.py`:
  - `MODEL_CONFIGS` points to a default (currently `TEST_CONFIGS`). Options include GPT-2 sized presets with `vocab_size`, `context_length`, `emb_dim`, `n_heads`, `n_layers`, `drop_rate`, `qkv_bias`.
  - Larger configs require more memory and time.

### Data and splits

- Samples are created with a sliding window over token IDs: `max_length` with stride `stride`.
- `create_data_loaders(...)` splits into train/val/test (defaults: 0.8/0.1/0.1). Val/test loaders do not drop the last batch.

### Checkpoints and resume

- Per-epoch and best checkpoints are saved under `checkpoints/`.
- On startup, `main.py` attempts to resume from the latest valid checkpoint. Corrupted/incomplete files are skipped automatically.

### Logging

- Training, config, and evaluation logs are written to `logs/log.txt`.
- Progress bars are shown in the terminal via `tqdm` for train/val/test.

### Tips

- Optimizer: AdamW improves generalization on Transformers compared to Adam. If desired, you can exclude biases and LayerNorm parameters from weight decay via parameter groups.
- If you change `context_length`, ensure your `max_length` in the dataset does not exceed it; generation crops the context to the model’s `context_length`.
- If validation loss looks suspiciously small, ensure the validation split has at least one batch for your text length, `max_length`, and `stride`.

### License

For educational use.