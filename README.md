# GAN Visualizer — macOS (CoreML)

A native macOS port of the browser visualizer. It discovers the generator
models on the Hugging Face Hub, downloads the ONNX files, **converts them to
CoreML** (`.mlpackage`) with the bundled Python converter, caches the compiled
model, and drives it from your microphone in real time. Rendering is Metal; the
NCHW model output is turned into an RGBA texture (with the hue channel-roll) by
a compute shader, and the audio→latent DSP runs in `AVAudioEngine` + vDSP.

## Requirements

- macOS 14 or later
- Xcode 16+ (tested with Xcode 27). The **Metal Toolchain** is a separate Xcode
  component in recent versions:
  ```bash
  xcodebuild -downloadComponent MetalToolchain
  ```
- Python 3 with the converter dependencies (`Converter/requirements.txt`):
  `coremltools`, `onnx`, `onnx2torch`, `onnxruntime`, `torch`, `numpy`.
  If they aren't importable by `python3`, create a venv:
  ```bash
  ./GANVisualizer/Converter/setup_converter.sh
  export GAN_CONVERTER_PYTHON="$PWD/GANVisualizer/Converter/.venv-converter/bin/python3"
  ```

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

On first launch the app fetches the model list, downloads the first model
(~14–105 MB) and converts it to CoreML (a few seconds). Later launches reuse the
cache in `~/Library/Application Support/GANVisualizer/Models/`.

## How the ONNX → CoreML conversion works

`Converter/convert_model.py` runs the whole pipeline and streams JSON progress:

1. **Download** the `.onnx` (the app does this with URLSession and progress).
2. Optional ONNX Runtime **constant-fold** (`ORT_ENABLE_BASIC`) so weights that
   are produced by `Transpose` nodes become plain initializers.
3. **Normalize** the graph: relabel opset 18 → 13 (the ops already use opset-13
   semantics), drop `Reshape` `allowzero`, and rewrite the StyleGAN2
   rank-6 `Reshape → Pad → Reshape` zero-stuff upsample into rank-≤5 steps
   (CoreML caps tensors at rank 5).
4. `onnx2torch` → `torch.jit.trace`.
5. `coremltools.convert` to an ML Program.
6. **Validate** the CoreML output against the PyTorch reference and, if
   float16 is too inaccurate (some StyleGAN2 demodulation graphs), re-convert in
   float32 automatically. The chosen precision is recorded in `meta.json`.

The `direct` strategy is tried first; if the graph needs constant folding the
`fold` strategy is used automatically.

### Performance on an Apple-silicon Mac

| Model | Output | Precision | Inference |
|---|---|---|---|
| `abstract_photo` | 512×512 | float16 | ~1.3 ms (ANE) |
| `flowers_big` (StyleGAN2) | 256×256 | float32 | ~28 ms |

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
| HF discovery, download, cache | `Sources/Model/ModelManager.swift` |
| Python converter runner | `Sources/Model/Converter.swift` |
| Control panel / sliders | `Sources/Views/` |
| ONNX → CoreML converter | `Converter/convert_model.py` |

## Current scope

Implemented: model discovery + download + conversion + cache, CoreML inference,
Metal rendering with the three mirror modes, microphone capture with the full
A-weighted/smoothed band-filter pipeline, demo signal, LSD modulation, all
parameter sliders (General / Motion / Pulse / Brightness tabs), FPS/inference
HUD, fullscreen.

Planned (the "extras"): the draggable spectrum editor, brightness-direction
discovery, model cover thumbnails in the gallery, the compute-unit
benchmark/override dropdown, and pointer-idle panel hiding.
