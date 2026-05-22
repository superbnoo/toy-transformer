import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from model import GPT2, GPTConfig

DATA_DIR = Path(__file__).parent / "data"


class SentenceDataset(Dataset):
    def __init__(self, split: str):
        payload = torch.load(DATA_DIR / f"{split}.pt")
        self.x = payload["input_ids"]
        self.y = payload["target_ids"]

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def main():
    meta = json.loads((DATA_DIR / "meta.json").read_text())
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    config = GPTConfig(
        vocab_size=meta["vocab_size"],
        block_size=meta["block_size"],
        n_layer=4,
        n_head=3,
        n_embd=6,
        dropout=0.0,
    )
    print("device is:", device)
    model = GPT2(config).to(device)

    train_ds = SentenceDataset("train")
    val_ds = SentenceDataset("val")
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=len(val_ds), shuffle=False) if len(val_ds) > 0 else None

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    epochs = 50

    def evaluate():
        model.eval()
        if val_loader is None:
            return None
        with torch.no_grad():
            losses = []
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                _, loss = model(xb, yb)
                losses.append(loss.item())
            return sum(losses) / len(losses)

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            _, loss = model(xb, yb)
            loss.backward()
            optimizer.step()
        if epoch % 5 == 0 or epoch == epochs:
            val_loss = evaluate()
            if val_loss is None:
                print(f"epoch {epoch:03d} | train_loss {loss.item():.4f}")
            else:
                print(f"epoch {epoch:03d} | train_loss {loss.item():.4f} | val_loss {val_loss:.4f}")

    ckpt_path = DATA_DIR / "gpt2_minimal.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": config.__dict__,
            "meta": meta,
        },
        ckpt_path,
    )
    print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
