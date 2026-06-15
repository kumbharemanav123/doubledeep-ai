# DoubleDeep AI

A static, browser-native AI image detector. The public site uses ONNX Runtime Web and an MIT-licensed Community Forensics model. Images are processed locally and are not uploaded.

## Run locally

Serve this directory with any static HTTP server, then open `index.html`.

## Model

- Source: `OwensLab/commfor-model-384`
- Input: center-cropped 384 x 384 RGB image
- Runtime: ONNX Runtime Web / WebAssembly
- Use: research and assistive evaluation only
