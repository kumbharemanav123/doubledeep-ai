# DoubleDeep AI

A static, browser-native AI image and video-origin detector. The public site uses exported correction memory, ONNX Runtime Web and the Apache-2.0 SigLIP classifier `Ateeqq/ai-vs-human-image-detector`. Images and videos are processed locally in the browser.

## Features

- Image analysis for JPEG, PNG and WEBP files.
- Deepfake video analyst for MP4, MOV and WEBM videos up to 5 minutes.
- Direct video URL analysis when the host allows browser cross-origin frame reads.
- YouTube/Shorts and social-video analysis through the optional local backend.
- Frame sampling, multi-frame model consensus, motion instability and freeze-loop checks.
- Web trace kit with reverse-search and source-search links.
- User feedback prompt after each answer.
- Local correction memory stored in `localStorage` and exportable as reviewed training data.
- Downloadable JSON reports for audits and retraining pipelines.

## Run locally

For browser-only image, uploaded-video and direct MP4/WebM analysis, serve this directory with any static HTTP server:

```bash
python3 -m http.server 8080
```

Then visit:

```text
http://localhost:8080/
```

Opening `index.html` directly from disk is not recommended because browsers may block ONNX model files and video frame reads from `file://` URLs.

## Run with YouTube link support

YouTube, Shorts, TikTok, Instagram and similar sites do not allow a static GitHub Pages page to read raw video frames. Run the local backend when you want to analyze those links:

```bash
cd /path/to/doubledeep-ai
python3 -m venv .venv
. .venv/bin/activate
pip install -r server/requirements.txt
python3 server/app.py
```

Then open:

```text
http://127.0.0.1:8787/
```

Paste the YouTube Shorts URL in the Video link tab. The backend resolves the page with `yt-dlp`, samples frames with OpenCV, runs the bundled ONNX model with ONNX Runtime, and returns the report to the website.

## Deploy to GitHub Pages

Commit and push the static files to the repository branch used by GitHub Pages:

```bash
git add index.html app.js styles.css README.md
git commit -m "Add video deepfake analyst"
git push origin main
```

If GitHub Pages is configured for the `main` branch root, the site will update automatically.

## Model and data

- Source model: `Ateeqq/ai-vs-human-image-detector`
- Input: whole-image, regional crops and sampled video frames resized to 224 x 224 RGB
- Runtime: ONNX Runtime Web / WebAssembly
- Fast path: exact SHA-256 and perceptual correction memory
- Feedback: saved locally in the browser and exportable for offline training

Static GitHub Pages cannot continuously train a large global ONNX model from user uploads. To improve the public model, export reviewed feedback, curate labels, retrain or fine-tune offline, export a new ONNX model, then replace `model.onnx`, `model-data-*.bin` and `model-manifest.json`.

The training scaffold is in `training/`. It expects curated folders named `training/data/real` and `training/data/ai`, then fine-tunes the SigLIP classifier and exports ONNX files for the site.

## Limitations

Detector results are probabilistic. Accuracy can shift with new generators, editing, screenshots, compression, camera recapture and platform re-encoding. The browser demo does not inspect audio, device sensor noise, server-side metadata or private social-platform data, and it cannot crawl the web automatically.
