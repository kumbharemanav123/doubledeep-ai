# Run DoubleDeep AI On Any Computer

DoubleDeep has two modes:

- GitHub Pages mode: image uploads, uploaded video files and some direct MP4/WebM links.
- Full local mode: everything above plus YouTube Shorts/social video links through the local backend.

## Windows

1. Install Python 3.10 or newer from `https://www.python.org/downloads/`.
2. Download or clone this repository.
3. Double-click `run_windows.bat`.
4. Open `http://127.0.0.1:8787/` if the browser does not open automatically.

## macOS or Linux

```bash
cd doubledeep-ai
chmod +x run.sh
./run.sh
```

Then open:

```text
http://127.0.0.1:8787/
```

## Requirements

- Python 3.10 or newer.
- Internet access on first run, because the script installs Python packages.
- Around 1 GB of free disk space for Python packages and model files.

YouTube analysis depends on `yt-dlp`. If YouTube changes its delivery system, rerun the script later so dependencies can be updated.
