# DoubleDeep Training Workflow

GitHub Pages cannot train a model. Use this folder on a computer with Python and, preferably, an NVIDIA GPU. After training, export a new ONNX model and copy it into the site root.

## Dataset layout

Create a curated dataset with this structure:

```text
training/data/
  real/
    camera_000001.jpg
    camera_000002.jpg
  ai/
    generated_000001.jpg
    generated_000002.jpg
```

Recommended real-image sources:

- COCO, Open Images, FFHQ, CelebA-HQ and your own camera images.
- Use original camera images when possible, not screenshots of camera images.

Recommended AI-image sources:

- GenImage, Synthbuster, CIFAKE, DiffusionDB-derived images and images generated from current tools such as Midjourney, DALL-E, Stable Diffusion, Flux and Firefly.
- Keep generator names in filenames or sidecar metadata so you can audit weak spots later.

Good training data matters more than raw volume. Balance real and AI images, remove duplicates, include compressed/social-media variants, and keep a separate validation set from sources not seen during training.

## Install

```bash
cd /path/to/doubledeep-ai
python3 -m venv .venv
. .venv/bin/activate
pip install -r training/requirements.txt
```

## Train

```bash
python training/fine_tune_siglip.py \
  --data-dir training/data \
  --epochs 3 \
  --batch-size 16 \
  --output-dir training/runs/siglip-finetuned
```

## Export into the website

```bash
python training/fine_tune_siglip.py \
  --data-dir training/data \
  --epochs 3 \
  --batch-size 16 \
  --output-dir training/runs/siglip-finetuned \
  --site-export-dir .
```

Then test locally:

```bash
python3 -m http.server 8080
```

The script exports `model.onnx` and `model-manifest.json`. If the exported model becomes too large for GitHub Pages, split it with ONNX external data before committing.
