import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from data import ChestXrayDataset, prepare_splits
from model import ConvVAE, choose_device, vae_loss


def run_epoch(model, loader, device, optimizer=None):
    model.train(optimizer is not None)
    total_loss = 0
    reconstruction_loss = 0
    kl_loss = 0
    count = 0

    with torch.set_grad_enabled(optimizer is not None):
        for images in loader:
            images = images.to(device)
            logits, mu, logvar = model(images)
            loss, reconstruction, kl = vae_loss(logits, images, mu, logvar)

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            batch_size = len(images)
            total_loss += loss.item() * batch_size
            reconstruction_loss += reconstruction.item() * batch_size
            kl_loss += kl.item() * batch_size
            count += batch_size

    return total_loss / count, reconstruction_loss / count, kl_loss / count


def plot_losses(history, output):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    epochs = [row["epoch"] for row in history]
    for axis, name in zip(axes, ["loss", "reconstruction", "kl"]):
        axis.plot(epochs, [row[f"train_{name}"] for row in history], label="Train")
        axis.plot(epochs, [row[f"val_{name}"] for row in history], label="Validation")
        axis.set_title("KL" if name == "kl" else name.capitalize())
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Loss per image")
        axis.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch-size must be positive")

    seed = 42
    latent_dim = 128
    learning_rate = 0.001
    torch.manual_seed(seed)
    device = choose_device(args.device)
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "model.pt").exists():
        raise FileExistsError("Choose a new --out-dir for a new training run.")

    splits = prepare_splits(args.data_dir, seed)
    train_loader = DataLoader(ChestXrayDataset(splits["train"]), args.batch_size, shuffle=True)
    val_loader = DataLoader(ChestXrayDataset(splits["val"]), args.batch_size)
    config = {
        **vars(args), "seed": seed, "latent_dim": latent_dim, "lr": learning_rate,
        "image_size": 64, "beta": 1, "train_images": len(splits["train"]),
        "val_images": len(splits["val"]), "test_images": len(splits["test"]),
    }
    (output / "config.json").write_text(json.dumps(config, indent=2))
    (output / "splits.json").write_text(json.dumps(splits, indent=2))

    model = ConvVAE(latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    fixed_noise = torch.randn(16, latent_dim).to(device)
    history = []
    best_loss = float("inf")
    print(f"Device: {device}, train: {len(splits['train'])}, validation: {len(splits['val'])}")

    for epoch in range(1, args.epochs + 1):
        train_values = run_epoch(model, train_loader, device, optimizer)
        val_values = run_epoch(model, val_loader, device)
        row = {"epoch": epoch}
        for index, name in enumerate(["loss", "reconstruction", "kl"]):
            row[f"train_{name}"] = train_values[index]
            row[f"val_{name}"] = val_values[index]
        history.append(row)

        if val_values[0] < best_loss:
            best_loss = val_values[0]
            torch.save({
                "model_state": model.state_dict(), "latent_dim": latent_dim, "epoch": epoch,
            }, output / "model.pt")

        with (output / "loss.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=row.keys())
            writer.writeheader()
            writer.writerows(history)
        print(f"Epoch {epoch}/{args.epochs}: train={train_values[0]:.2f}, val={val_values[0]:.2f}", flush=True)

    checkpoint = torch.load(output / "model.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    with torch.no_grad():
        generated = model.decode(fixed_noise).sigmoid()
        save_image(generated.cpu(), output / "generated.png", nrow=4)
        originals = next(iter(val_loader))[:8].to(device)
        mu, _ = model.encode(originals)
        reconstructed = model.decode(mu).sigmoid()
        save_image(torch.cat([originals, reconstructed]).cpu(), output / "reconstruction.png", nrow=len(originals))

    plot_losses(history, output / "loss.png")
    print(f"Saved epoch {checkpoint['epoch']} and images to {output}")


if __name__ == "__main__":
    main()
