# GAN Visualizer — macOS (CoreML)

A native macOS port of the browser visualizer. It discovers the generator
models on the Hugging Face Hub, downloads the ready-made **CoreML packages**
from the repository's `coreml/` folder, compiles and caches them, and drives
them from your microphone in real time. Rendering is Metal; the NCHW model
output is turned into an RGBA texture (with the hue channel-roll) by a compute
shader, and the audio→latent DSP runs in `AVAudioEngine` + vDSP.

## Requirements

- macOS 14 or later
- Xcode 16+ (tested with Xcode 27). The **Metal Toolchain** is a separate Xcode
  component in recent versions:
  ```bash
  xcodebuild -downloadComponent MetalToolchain
  ```

No Python, ONNX Runtime or other third-party dependencies are needed at build
or run time — the models are already converted to CoreML and hosted on the Hub.

## Build & run

```bash
cd macos
xcodegen generate          # regenerates GANVisualizer.xcodeproj (optional)
open GANVisualizer.xcodeproj
```

Run the **GANVisualizer** scheme, or build from the CLI:

```bash
xcodebuild -project GANVisualizer.xcodeproj -scheme GANVisualizer \
           -configuration Release -destination 'platform=macOS' build
```

On first launch the app fetches the model list, downloads the first model's
CoreML package (~6–145 MB) and compiles it. Later launches reuse the cache in
`~/Library/Application Support/GANVisualizer/Models/`.

## How model loading works

All models live in the `coreml/` folder of the Hugging Face repository
`lorenzsichert/gan-visualizer`. Each model has its own directory:

```
coreml/<model>/meta.json                                   # shape / precision info
coreml/<model>/model.mlpackage/Manifest.json
coreml/<model>/model.mlpackage/Data/com.apple.CoreML/model.mlmodel
coreml/<model>/model.mlpackage/Data/com.apple.CoreML/weights/weight.bin
```

`ModelManager` lists that folder through the Hub API — following the API's
pagination cursor so every model is included, with no hardcoded model list —
then for the selected model:

1. Downloads the package files with `URLSession` (with progress), skipping files
   already present in the cache with the expected size.
2. Compiles `model.mlpackage` to `model.mlmodelc` with
   `MLModel.compileModel(at:)`.
3. Loads the compiled model into `InferenceEngine`, which discovers the input /
   output names and shapes from `MLModel.modelDescription`.

No conversion happens on the device; the hosted packages already contain the
right precision (float16 or float32) and are used as-is.

### Performance on an Apple-silicon Mac

| Model | Output | Precision | Inference |
|---|---|---|---|
| `abstract_photo_medium` | 512×512 | float16 | ~3.8 ms |
| `abstract_photo_big` | 512×512 | float16 | ~24 ms |
| `flowers_big` (StyleGAN2) | 256×256 | float32 | ~29 ms |

Figures are from the `meta.json` shipped with each hosted model; actual speed
depends on the Mac and compute unit.

## Audio input

The control panel's **Input** selector offers:

- **Mic** — `AVAudioEngine`, using an `AVAudioSinkNode` rather than an input
  tap. A tap batches delivery into large buffers on macOS (and ignores the
  requested buffer size), which adds ~100 ms before the visualizer hears
  anything; the sink node receives the raw hardware IO buffer (a few ms)
  directly on the CoreAudio real-time thread. The shared 2048-point sliding
  window still means the spectrum itself reflects the last ~43 ms of audio at
  48 kHz.
- **System** — system output, preferring a **Core Audio process tap**
  (`AudioHardwareCreateProcessTap`, macOS 14.2+), the lowest-latency public
  route: a private, tap-only aggregate device whose IO block runs directly on
  the Core Audio real-time thread, with none of ScreenCaptureKit's
  sample-handler buffering. It carries its own **System Audio Recording**
  permission (System Settings › Privacy & Security); grant it and relaunch. If
  the tap is unavailable (macOS 14.0/14.1) or cannot be created — e.g. the
  permission is still pending — the app falls back to **ScreenCaptureKit**,
  which needs **Screen Recording** permission instead. The fallback stream uses
  the minimum `queueDepth`, drains its 2×2 / 1 fps video output on a separate
  low-priority queue (so video can never back-pressure audio), and delivers
  audio on a `.userInteractive` queue.

  Dev builds are ad-hoc signed, so macOS ties each grant to that build's
  signature. After a rebuild the binary is a different "app" to TCC, so you may
  have to reset and re-grant even though the toggle still looks enabled:
  ```bash
  tccutil reset ScreenCapture com.jakobsichert.GANVisualizer
  ```
  Signing with a stable identity (an Apple Development certificate, or a
  self-signed cert trusted for code signing) makes the grant persist across
  rebuilds.
- **Demo** — a synthesized signal, for testing without any capture.
- **Off**.

Every source feeds the same `SpectrumAnalyzer` — one vDSP 2048-point FFT per
capture callback, run on the capture thread itself (the CoreAudio real-time
thread for the mic and for the process tap) so no extra thread hop is added.
Microphone and system audio therefore have identical DSP cost.

## Architecture

| Piece | File |
|---|---|
| App + window, menus | `Sources/App/` |
| Settings (port of `settings.js`/`midi.py`) | `Sources/Model/Settings.swift` |
| LSD latent modulation (port of `lsd.js`) | `Sources/Model/LSDLatent.swift` |
| Mic capture + vDSP FFT | `Sources/Model/AudioPipeline.swift` |
| Audio → latent DSP (port of `computeLatent`) | `Sources/Model/LatentEngine.swift` |
| CoreML session, latest-wins pipeline | `Sources/Model/InferenceEngine.swift` |
| Metal renderer (mirror tiling) | `Sources/Model/Renderer.swift` |
| Shaders | `Resources/Shaders.metal` |
| HF discovery, download of hosted CoreML packages, compile + cache | `Sources/Model/ModelManager.swift` |
| Control panel | `Sources/Views/` |

## Current scope

Implemented: hosted CoreML model discovery + download + compile + cache, CoreML inference,
Metal rendering with the three mirror modes, microphone capture with the full
A-weighted/smoothed band-filter pipeline, demo signal, LSD modulation, a
web-style control sidebar (telemetry, preamp and hue pills, live spectrum with
draggable band handles and long-press width editing, General / Motion / Pulse /
Brightness tabs with explanations, generator gallery with Hub author credits),
brightness-direction discovery with a per-model cache, FPS/inference HUD, pause,
fullscreen.

Planned (the "extras"): model cover thumbnails in the gallery, the compute-unit
benchmark/override dropdown, and pointer-idle panel hiding.
