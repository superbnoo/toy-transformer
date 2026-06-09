import argparse
from pathlib import Path

import torch

from model import GPT2, GPTConfig


def load_model(base_dir: Path, device: str):
    ckpt_path = base_dir / "data" / "gpt2_minimal.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    meta = ckpt["meta"]
    config = GPTConfig(
        vocab_size=meta["vocab_size"],
        block_size=meta["block_size"],
        n_layer=meta.get("config", {}).get("n_layer", 4) if isinstance(meta, dict) else 4,
        n_head=meta.get("config", {}).get("n_head", 3) if isinstance(meta, dict) else 3,
        n_embd=meta.get("config", {}).get("n_embd", 6) if isinstance(meta, dict) else 6,
        dropout=0.0,
    )
    model = GPT2(config).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-new", type=int, default=5, help="max new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="sampling temperature")
    parser.add_argument("--top-k", type=int, default=5, help="top-k filtering")
    parser.add_argument("--num-samples", type=int, default=1, help="number of independent samples to generate")
    args = parser.parse_args()

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    base = Path(__file__).parent
    model, meta = load_model(base, device)
    stoi, itos = meta["stoi"], meta["itos"]

    print("Model loaded.")
    print("Enter a prompt (Ctrl+C to exit)\n")

    try:
        while True:
            prompt = input("> ").strip()

            if not prompt:
                continue

            words = prompt.split()

            missing = [w for w in words if w not in stoi]
            if missing:
                print(f"Tokens not in vocab: {missing}")
                continue

            idx_prompt = torch.tensor(
                [[stoi[w] for w in words]],
                dtype=torch.long,
                device=device,
            )

            print("Prompt tokens:", words)
            print("Prompt ids:", idx_prompt.cpu())

            for sample_idx in range(args.num_samples):
                with torch.no_grad():
                    out = model.generate(
                        idx_prompt,
                        max_new_tokens=args.max_new,
                        temperature=args.temperature,
                        top_k=args.top_k,
                    )

                raw_decoded = [itos[i] for i in out[0].tolist()]
                trunc_decoded = raw_decoded

                if meta["eos_token"] in trunc_decoded:
                    trunc_decoded = trunc_decoded[:trunc_decoded.index(meta["eos_token"])]

                print(f"\nSample {sample_idx + 1}/{args.num_samples}")
                print("Generated token ids:", out.cpu())
                print("Raw decoded tokens:", raw_decoded)
                print("Raw decoded text:", " ".join(raw_decoded))
                print("Truncated at <eos> tokens:", trunc_decoded)
                print("Truncated text:", " ".join(trunc_decoded))

            print()

    except KeyboardInterrupt:
        print("\nExiting...")


if __name__ == "__main__":
    main()
