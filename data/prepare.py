import argparse
import json
import random
from pathlib import Path

import torch

DATA_DIR = Path(__file__).parent
RAW_SENTENCES = DATA_DIR / "sentences.txt"
EOS_TOKEN = "<eos>"
BLOCK_SIZE = 8  # max sequence length


def add_eos(sent: str) -> str:
    parts = sent.strip().split()
    if not parts:
        return ""
    return " ".join(parts + [EOS_TOKEN]) if parts[-1] != EOS_TOKEN else " ".join(parts)


def parse_args():
    p = argparse.ArgumentParser(description="Prepare toy GPT-2 dataset")
    p.add_argument(
        "--train-frac",
        type=float,
        default=0.8,
        help="Fraction of samples for training (0.0-1.0). Use 1.0 for no validation split.",
    )
    p.add_argument(
        "--generate",
        type=int,
        default=0,
        help="If >0, auto-generate this many common-sense sentences; also overwrites sentences.txt",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed for shuffling/generation")
    return p.parse_args()


def main():
    args = parse_args()
    train_frac = min(1.0, max(0.0, args.train_frac))
    random.seed(args.seed)

    land_animals = ["cat", "dog", "rabbit", "horse", "cow", "sheep", "goat", "pig", "bear", "deer"]
    air_animals = ["bird", "duck", "owl"]
    water_animals = ["fish", "seal", "turtle", "frog"]
    people = ["child", "kid", "girl", "boy", "mom", "dad", "teacher", "friend"]
    foods = ["apple", "bread", "carrot", "grass", "hay", "seed", "fish", "corn", "berry", "milk", "water", "soup", "rice"]
    toys = ["ball", "kite", "car", "doll", "blocks", "puzzle", "boat"]
    nature = ["tree", "rock", "flower", "cloud", "sun", "moon", "star", "river", "lake", "pond", "hill", "beach", "forest", "garden", "field", "yard", "farm", "barn"]
    cozy = ["bed", "hay", "grass", "mat", "blanket"]
    friends = ["friend", "mom", "dad", "teacher", "kid"]

    def s(subj):
        # Return a plausible sentence for the subject
        if subj in land_animals:
            choices = [
                f"{subj} eats {random.choice(foods)}",
                f"{subj} runs to the {random.choice(['park','yard','field','garden'])}",
                f"{subj} plays with a {random.choice(toys)}",
                f"{subj} sleeps on {random.choice(cozy)}",
                f"{subj} drinks cool water",
                f"{subj} chases a {random.choice(['ball','butterfly'])}",
            ]
        elif subj in air_animals:
            choices = [
                f"{subj} flies over the {random.choice(nature)}",
                f"{subj} sings in the {random.choice(['tree','park'])}",
                f"{subj} lands on a {random.choice(['branch','rock'])}",
                f"{subj} eats {random.choice(['seed','berry'])}",
            ]
        elif subj in water_animals:
            choices = [
                f"{subj} swims in the {random.choice(['lake','pond','river'])}",
                f"{subj} dives under the water",
                f"{subj} eats {random.choice(['fish','worm','seed'])}",
                f"{subj} rests on a rock",
                f"{subj} jumps over a log",
            ]
        else:  # people
            choices = [
                f"{subj} reads a book",
                f"{subj} plays with a {random.choice(toys)}",
                f"{subj} walks to the {random.choice(['park','school','home'])}",
                f"{subj} helps a {random.choice(friends)}",
                f"{subj} draws a {random.choice(['cat','dog','house','tree'])}",
                f"{subj} carries a bag",
            ]
        return random.choice(choices)

    if args.generate > 0:
        pool = land_animals + air_animals + water_animals + people
        sentences = [add_eos(s(random.choice(pool))) for _ in range(args.generate)]
        RAW_SENTENCES.write_text("\n".join(sentence.replace(f" {EOS_TOKEN}", "") for sentence in sentences) + "\n")
    else:
        if not RAW_SENTENCES.exists():
            raise FileNotFoundError(f"Missing {RAW_SENTENCES}")
        lines = [line.strip() for line in RAW_SENTENCES.read_text().splitlines() if line.strip()]
        if not lines:
            raise ValueError("No sentences found in sentences.txt")
        sentences = [add_eos(line) for line in lines]
    tokens_per_sentence = [s.split() for s in sentences]

    vocab = sorted({tok for sent in tokens_per_sentence for tok in sent})
    stoi = {tok: idx for idx, tok in enumerate(vocab)}
    vocab_size = len(vocab)

    lengths = [len(s) for s in tokens_per_sentence]
    max_len = max(lengths)
    if max_len > BLOCK_SIZE:
        raise ValueError(f"Max sentence length {max_len} exceeds block size {BLOCK_SIZE}")

    inputs, targets = [], []
    pad_id = stoi[EOS_TOKEN]
    for toks in tokens_per_sentence:
        ids = torch.tensor([stoi[t] for t in toks], dtype=torch.long)
        padded = torch.full((max_len,), pad_id, dtype=torch.long)
        padded[: len(ids)] = ids

        tgt = torch.full((max_len,), pad_id, dtype=torch.long)
        if len(ids) > 1:
            tgt[: len(ids) - 1] = ids[1:]
        tgt[len(ids) - 1] = pad_id  # predict eos after the last token in the sentence

        inputs.append(padded)
        targets.append(tgt)

    seq_len = max_len
    input_tensor = torch.stack(inputs)
    target_tensor = torch.stack(targets)

    indices = list(range(input_tensor.size(0)))
    random.shuffle(indices)
    train_count = int(train_frac * len(indices))
    train_count = min(len(indices), max(0, train_count))
    train_idx, val_idx = indices[:train_count], indices[train_count:]

    torch.save(
        {"input_ids": input_tensor[train_idx], "target_ids": target_tensor[train_idx]},
        DATA_DIR / "train.pt",
    )
    torch.save(
        {"input_ids": input_tensor[val_idx], "target_ids": target_tensor[val_idx]},
        DATA_DIR / "val.pt",
    )

    meta = {
        "vocab_size": vocab_size,
        "vocab": vocab,
        "stoi": stoi,
        "itos": vocab,  # list aligned with indices
        "block_size": BLOCK_SIZE,
        "eos_token": EOS_TOKEN,
        "sentence_length": seq_len,
        "num_train": len(train_idx),
        "num_val": len(val_idx),
        "train_frac": train_frac,
        "generated": args.generate,
        "seed": args.seed,
    }
    (DATA_DIR / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"Prepared dataset")
    print(f"Vocab size: {vocab_size}")
    print(f"Train samples: {len(train_idx)}, Val samples: {len(val_idx)}")
    print(f"Sequence length: {seq_len}, Block size: {BLOCK_SIZE}")


if __name__ == "__main__":
    main()
