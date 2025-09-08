import torch
import tiktoken

from src.model.gpt import GPTModel


def load_model(model_configs, checkpoint_path=None, device=None):
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    model = GPTModel(model_configs).to(device)
    if checkpoint_path is not None:
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state["model"])
    model.eval()
    return model, device


@torch.no_grad()
def generate_text(model, prompt, max_new_tokens=50, temperature=1.0, top_k=50):
    tokenizer = tiktoken.get_encoding("gpt2")
    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=next(model.parameters()).device)
    out_ids = model.generate(input_ids, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
    text = tokenizer.decode(out_ids[0].tolist())
    return text


if __name__ == "__main__":
    from configs.model_configs import MODEL_CONFIGS
    import argparse

    parser = argparse.ArgumentParser(description="GPT-2 like text generation")
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt text")
    parser.add_argument("--ckpt", type=str, default=None, help="Path to checkpoint .pth file")
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=50)
    args = parser.parse_args()

    model, device = load_model(MODEL_CONFIGS, args.ckpt)
    text = generate_text(model, args.prompt, max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_k=args.top_k)
    print(text)


