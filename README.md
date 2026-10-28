# Chest X-ray VAE

A small convolutional VAE for Lab 1. It generates 64 x 64 grayscale chest X-rays using PyTorch.

## Setup

```bash
python -m pip install -r requirements.txt
```

Download the [Chest X-Ray Images (Pneumonia) dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) and place its `train/` and `test/` folders under `data/chest_xray/`.

## Run

```bash
python train.py --data-dir data/chest_xray --epochs 50 --out-dir results_new
python evaluate.py --data-dir data/chest_xray --checkpoint results_new/model.pt
```

Training automatically uses CUDA, Apple MPS, or CPU. Use a new output directory for each run.

The output includes model weights, loss curves, generated samples, reconstructions, and IS/FID scores. Evaluation follows the [PyTorch-Ignite tutorial](https://pytorch-ignite.ai/blog/gan-evaluation-with-fid-and-is/) linked in the assignment.

The included `results/` folder is from the earlier local run. The final report uses the DSMLP run.
