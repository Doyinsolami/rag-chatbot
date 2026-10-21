import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

DATA_DIR = Path(__file__).resolve().parent / "data"
GEIA_DIR = DATA_DIR / "geia"
MODEL_OUT_DIR = Path(__file__).resolve().parent / "models" / "attacker"

VICTIM_MODEL = "all-MiniLM-L6-v2"
ATTACKER_MODEL = "gpt2"
MINILM_DIM = 384
GPT2_DIM = 768
PREFIX_LENGTH = 10
MAX_LENGTH = 256
SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class NoteDataset(Dataset):
    def __init__(self, jsonl_path, limit=None):
        self.texts = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                self.texts.append(record["text"])
        if limit is not None:
            self.texts = self.texts[:limit]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        return self.texts[index]

    def collate(self, batch):
        return batch


class Projection(nn.Module):
    def __init__(self, in_dim=MINILM_DIM, out_dim=GPT2_DIM, prefix_length=PREFIX_LENGTH):
        super().__init__()
        self.prefix_length = prefix_length
        self.out_dim = out_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim * prefix_length),
        )

    def forward(self, x):
        return self.net(x).view(x.size(0), self.prefix_length, self.out_dim)


def build_prefix(embeddings, projection, attacker, tokenizer, device):
    prefix = projection(embeddings)
    bos_ids = torch.full(
        (embeddings.size(0), 1),
        tokenizer.bos_token_id,
        dtype=torch.long,
        device=device,
    )
    bos_embeds = attacker.transformer.wte(bos_ids)
    return torch.cat((bos_embeds, prefix), dim=1)


def train_on_batch(batch_texts, victim, projection, attacker, tokenizer, device,
                   shuffle_embeddings=False):
    with torch.no_grad():
        embeddings = victim.encode(batch_texts, convert_to_tensor=True).to(device)
    embeddings = embeddings.clone().detach()
    if shuffle_embeddings:
        embeddings = torch.roll(embeddings, shifts=1, dims=0)

    prefix_embeds = build_prefix(embeddings, projection, attacker, tokenizer, device)
    prefix_size = prefix_embeds.size(1)

    encoded = tokenizer(
        batch_texts,
        add_special_tokens=False,
        padding=False,
        truncation=True,
        max_length=MAX_LENGTH - 1,
    )
    input_ids = torch.full(
        (len(batch_texts), MAX_LENGTH),
        tokenizer.pad_token_id,
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros(
        (len(batch_texts), MAX_LENGTH), dtype=torch.long, device=device
    )
    for row, token_ids in enumerate(encoded["input_ids"]):
        sequence = token_ids + [tokenizer.eos_token_id]
        sequence_length = len(sequence)
        input_ids[row, :sequence_length] = torch.tensor(sequence, device=device)
        attention_mask[row, :sequence_length] = 1

    token_embeds = attacker.transformer.wte(input_ids)

    inputs_embeds = torch.cat((prefix_embeds, token_embeds), dim=1)
    model_attention_mask = torch.cat(
        (
            torch.ones((len(batch_texts), prefix_size), dtype=torch.long, device=device),
            attention_mask,
        ),
        dim=1,
    )

    outputs = attacker(
        inputs_embeds=inputs_embeds,
        attention_mask=model_attention_mask,
    )
    logits = outputs.logits

    shift_logits = logits[:, prefix_size - 1:-1, :].contiguous()
    shift_labels = input_ids.masked_fill(attention_mask == 0, -100).contiguous()

    loss = nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )
    return loss


def evaluate(dataloader, victim, projection, attacker, tokenizer, device):
    attacker.eval()
    projection.eval()
    total_loss = 0.0
    total_shuffled_loss = 0.0
    with torch.no_grad():
        for batch_texts in dataloader:
            total_loss += train_on_batch(
                batch_texts, victim, projection, attacker, tokenizer, device
            ).item()
            total_shuffled_loss += train_on_batch(
                batch_texts, victim, projection, attacker, tokenizer, device,
                shuffle_embeddings=True,
            ).item()
    attacker.train()
    projection.train()
    return total_loss / len(dataloader), total_shuffled_loss / len(dataloader)


def save_checkpoint(attacker, tokenizer, projection):
    MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)
    gpt2_dir = MODEL_OUT_DIR / "gpt2"
    attacker.save_pretrained(gpt2_dir)
    tokenizer.save_pretrained(gpt2_dir)
    torch.save(projection.state_dict(), MODEL_OUT_DIR / "projection.pt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--projection_lr", type=float, default=5e-4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--patience", type=int, default=3)
    args = parser.parse_args()

    set_seed(args.seed)
    print(f"Seed: {args.seed}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    victim = SentenceTransformer(VICTIM_MODEL, device=str(device))
    attacker = AutoModelForCausalLM.from_pretrained(ATTACKER_MODEL).to(device)
    tokenizer = AutoTokenizer.from_pretrained(ATTACKER_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    projection = Projection().to(device)

    train_dataset = NoteDataset(GEIA_DIR / "train.jsonl", limit=args.limit)
    shuffle_generator = torch.Generator()
    shuffle_generator.manual_seed(args.seed)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate,
        generator=shuffle_generator,
    )
    print(f"Training on {len(train_dataset)} notes")

    validation_dataset = NoteDataset(GEIA_DIR / "validation.jsonl")
    validation_generator = torch.Generator()
    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=validation_dataset.collate,
        generator=validation_generator,
    )
    print(f"Validating on {len(validation_dataset)} notes")

    optimizer = torch.optim.AdamW(
        [
            {"params": projection.parameters(), "lr": args.projection_lr},
            {"params": attacker.parameters(), "lr": args.lr},
        ]
    )

    best_val_loss = float("inf")
    best_shuffled_val_loss = None
    best_epoch = None
    epochs_without_improvement = 0
    epochs_completed = 0
    history = []

    attacker.train()
    projection.train()
    for epoch in range(args.epochs):
        running_loss = 0.0
        for step, batch_texts in enumerate(train_dataloader):
            optimizer.zero_grad()
            loss = train_on_batch(
                batch_texts, victim, projection, attacker, tokenizer, device
            )
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if step % 20 == 0:
                print(f"epoch {epoch} step {step} loss {loss.item():.4f}")
        avg_train_loss = running_loss / len(train_dataloader)
        epochs_completed = epoch + 1

        validation_generator.manual_seed(args.seed)
        val_loss, shuffled_val_loss = evaluate(
            validation_dataloader, victim, projection, attacker, tokenizer, device
        )
        gap = shuffled_val_loss - val_loss
        history.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "validation_loss": val_loss,
            "shuffled_validation_loss": shuffled_val_loss,
            "gap": gap,
        })
        print(f"== epoch {epoch} done, average train loss {avg_train_loss:.4f}, "
              f"validation loss (real embeddings) {val_loss:.4f}, "
              f"validation loss (shuffled embeddings) {shuffled_val_loss:.4f}, "
              f"gap {gap:.4f} ==")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_shuffled_val_loss = shuffled_val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(attacker, tokenizer, projection)
            print(f"Validation loss improved, saved checkpoint from epoch {epoch}")
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epoch(s)")
            if epochs_without_improvement >= args.patience:
                print(f"Stopping early after epoch {epoch}: "
                      f"no validation improvement in {args.patience} epochs")
                break

    run_config = {
        "seed": args.seed,
        "epochs_requested": args.epochs,
        "epochs_completed": epochs_completed,
        "early_stopped": epochs_completed < args.epochs,
        "patience": args.patience,
        "best_epoch": best_epoch,
        "best_validation_loss": best_val_loss,
        "best_shuffled_validation_loss": best_shuffled_val_loss,
        "best_gap": best_shuffled_val_loss - best_val_loss,
        "history": history,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "projection_lr": args.projection_lr,
        "prefix_length": PREFIX_LENGTH,
        "limit": args.limit,
        "victim_model": VICTIM_MODEL,
        "attacker_model": ATTACKER_MODEL,
        "max_length": MAX_LENGTH,
        "train_notes": len(train_dataset),
        "validation_notes": len(validation_dataset),
        "device": str(device),
    }
    MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)
    (MODEL_OUT_DIR / "training_run_config.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )
    print(f"Best model (epoch {best_epoch}, validation loss {best_val_loss:.4f}) "
          f"saved to {MODEL_OUT_DIR}")


if __name__ == "__main__":
    main()