# DoubleDeep AI

A static, browser-native AI image detector. The public site uses exported local correction memory plus ONNX Runtime Web and the same Apache-2.0 SigLIP primary classifier as the local DoubleDeep server. Images are processed locally and are not uploaded.

## Run locally

Serve this directory with any static HTTP server, then open `index.html`.

## Model

- Source: `Ateeqq/ai-vs-human-image-detector`
- Input: whole-image and regional 224 x 224 RGB views
- Aggregation: the same conservative multi-view probability and uncertainty logic used by the local server
- Fast path: reviewed exact and perceptual correction memory from local feedback
- Runtime: ONNX Runtime Web / WebAssembly
- Use: research and assistive evaluation only
