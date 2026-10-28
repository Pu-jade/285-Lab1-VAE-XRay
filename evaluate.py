import argparse
import json
from pathlib import Path

import ignite
import torch
from ignite.metrics import FID, InceptionScore
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF

from data import ChestXrayDataset, discover_images
from model import ConvVAE


def prepare_images(images):
    resized = []
    for image in images.detach().cpu():
        image = TF.to_pil_image(image.clamp(0, 1)).convert("RGB")
        image = image.resize((299, 299), Image.Resampling.BILINEAR)
        resized.append(TF.to_tensor(image))
    return torch.stack(resized)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", default="results/model.pt")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive")

    torch.manual_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = ConvVAE(checkpoint["latent_dim"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    paths = discover_images(Path(args.data_dir) / "test")
    if len(paths) < 2:
        raise ValueError("The test folder needs at least two images.")
    loader = DataLoader(ChestXrayDataset(paths), batch_size=args.batch_size)

    # Metrics follow the evaluation tutorial linked in the assignment.
    fid = FID(device=device)
    inception_score = InceptionScore(device=device)
    with torch.no_grad():
        for real in loader:
            noise = torch.randn(len(real), model.latent_dim).to(device)
            fake = model.decode(noise).sigmoid()
            real = prepare_images(real).to(device)
            fake = prepare_images(fake).to(device)
            fid.update((fake, real))
            inception_score.update(fake)

    results = {
        "FID": float(fid.compute()),
        "IS": float(inception_score.compute()),
        "samples": len(paths),
        "implementation": "Ignite default 1000-output Inception; tutorial preprocessing",
        "ignite_version": ignite.__version__,
    }
    output = Path(args.checkpoint).parent / "metrics.json"
    output.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
