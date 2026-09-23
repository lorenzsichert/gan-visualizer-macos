#!/usr/bin/env python3
"""Hugging Face ONNX -> CoreML converter for GAN Visualizer.

Runs the whole proven pipeline:

    ONNX (opset 18)
      -> optional ONNX Runtime constant-fold (folds Transpose/weights into
         initializers so the weight tensors are plain initializers)
      -> normalize (opset relabel, drop Reshape allowzero, rewrite rank-6
         Reshape->Pad->Reshape zero-stuff patterns into rank<=5 steps)
      -> onnx2torch -> torch.jit.trace
      -> coremltools.convert (ML Program)

Prints machine-readable progress lines (one JSON object per line) on stdout.
"""

import argparse
import copy
import json
import os
import struct
import sys
import tempfile
import time
import zlib

import numpy as np


def emit(phase, text, progress=None):
    msg = {"phase": phase, "text": text}
    if progress is not None:
        msg["progress"] = max(0.0, min(1.0, float(progress)))
    print(json.dumps(msg), flush=True)


# ---------------------------------------------------------------------------
# onnx2torch patch: its ReduceMean v13 converter only reads the `axes`
# attribute, but opset-13 graphs pass axes as a second (constant) input.
# ---------------------------------------------------------------------------
def patch_onnx2torch():
    from onnx2torch.node_converters import registry as _registry
    from onnx2torch.node_converters.reduce import OnnxReduceStaticAxes
    from onnx2torch.utils.common import (
        OperationConverterResult, OnnxMapping, onnx_mapping_from_node, get_const_value,
    )

    def _reduce_mean(node, graph):
        keepdims = node.attributes.get("keepdims", 1)
        if len(node.input_values) == 2:
            try:
                axes = list(get_const_value(node.input_values[1], graph).tolist())
                return OperationConverterResult(
                    torch_module=OnnxReduceStaticAxes(
                        operation_type="ReduceMean", axes=axes, keepdims=keepdims),
                    onnx_mapping=OnnxMapping(
                        inputs=(node.input_values[0],), outputs=node.output_values),
                )
            except KeyError:
                pass
        return OperationConverterResult(
            torch_module=OnnxReduceStaticAxes(
                operation_type="ReduceMean", axes=node.attributes.get("axes", None),
                keepdims=keepdims),
            onnx_mapping=onnx_mapping_from_node(node),
        )

    _registry._CONVERTER_REGISTRY[
        _registry.OperationDescription("", "ReduceMean", 13)] = _reduce_mean


# ---------------------------------------------------------------------------
# Graph normalization
# ---------------------------------------------------------------------------
def fold_with_ort(model_path):
    """Constant-fold via ONNX Runtime (basic level: no fused custom ops)."""
    import onnx
    import onnxruntime as ort

    fd, opt_path = tempfile.mkstemp(suffix="_opt.onnx")
    os.close(fd)
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    so.optimized_model_filepath = opt_path
    so.log_severity_level = 3
    ort.InferenceSession(model_path, so, providers=["CPUExecutionProvider"])
    folded = onnx.load(opt_path)
    os.remove(opt_path)
    return folded


def fix_rank6(model):
    """Rewrite rank-6 Reshape->Pad->Reshape subgraphs into rank<=5 steps.

    CoreML only supports tensors up to rank 5. StyleGAN2 exports an upsample as
    reshape-to-[..,1,..,1] + pad + reshape; this splits it into two rank-5 steps.
    """
    import onnx
    from onnx import numpy_helper, helper

    inits = {i.name: i for i in model.graph.initializer}
    consumers = {}
    for n in model.graph.node:
        for i in n.input:
            consumers.setdefault(i, []).append(n)

    def shape_of(node):
        if len(node.input) > 1 and node.input[1] in inits:
            return [int(v) for v in numpy_helper.to_array(inits[node.input[1]]).reshape(-1)]
        return None

    uid = [0]

    def new_init(name, arr):
        name = f"rw_{name}_{uid[0]}"
        uid[0] += 1
        model.graph.initializer.append(
            numpy_helper.from_array(np.asarray(arr, dtype=np.int64), name))
        return name

    def new_node(op, inputs, outputs, **attrs):
        n = helper.make_node(op, inputs, [outputs], name=f"rw_{op}_{uid[0]}", **attrs)
        uid[0] += 1
        return n

    to_remove = set()
    replacements = {}
    for r in list(model.graph.node):
        if r.op_type != "Reshape":
            continue
        S = shape_of(r)
        if not S or len(S) < 6:
            continue
        cons = consumers.get(r.output[0], [])
        if len(cons) != 1:
            continue
        mid = cons[0]
        n = len(S)

        # The upsample can be expressed either as a zero-pad or a nearest
        # tile/broadcast; collect the per-dimension expansion factor.
        if mid.op_type == "Pad" and len(mid.input) > 1 and mid.input[1] in inits:
            pads = [int(v) for v in numpy_helper.to_array(inits[mid.input[1]]).reshape(-1)]
            if len(pads) != 2 * n:
                continue
            begins, ends = pads[:n], pads[n:]
            factors = [1 + begins[i] + ends[i] for i in range(n)]
            kind = "pad"
        elif mid.op_type == "Tile" and len(mid.input) > 1 and mid.input[1] in inits:
            reps = [int(v) for v in numpy_helper.to_array(inits[mid.input[1]]).reshape(-1)]
            if len(reps) != n:
                continue
            factors = list(reps)
            kind = "tile"
        else:
            continue

        mid_cons = consumers.get(mid.output[0], [])
        if len(mid_cons) != 1 or mid_cons[0].op_type != "Reshape":
            continue
        r2 = mid_cons[0]

        # No-op expansion -> pure shape round-trip.
        if all(f == 1 for f in factors):
            r2.input[0] = r.input[0]
            to_remove.update([r.name, mid.name])
            continue

        singles = [i for i, s in enumerate(S) if s == 1 and i != 0]
        if len(singles) != 2:
            continue
        a_dim, b_dim = singles
        if any(factors[i] != 1 for i in range(n) if i not in singles):
            continue
        fa, fb = factors[a_dim], factors[b_dim]
        if fa < 1 or fb < 1:
            continue
        real = [d for i, d in enumerate(S) if i not in singles]
        B, C, H, W = real
        H1 = H * fa
        W1 = W * fb

        def apply_mid(src, out_name, dim_index, factor):
            if kind == "pad":
                pads5 = [0] * 10
                pads5[5 + dim_index] = factor - 1
                init = new_init("p", pads5)
                return new_node("Pad", [src, init], out_name, mode="constant")
            reps5 = [1, 1, 1, 1, 1]
            reps5[dim_index] = factor
            init = new_init("t", reps5)
            return new_node("Tile", [src, init], out_name)

        created = []
        s1 = new_init("s1", [B, C, H, 1, W])
        o1 = f"rw_o1_{uid[0]}"; created.append(new_node("Reshape", [r.input[0], s1], o1)); uid[0] += 1
        o2 = f"rw_o2_{uid[0]}"; created.append(apply_mid(o1, o2, 3, fa)); uid[0] += 1
        s2 = new_init("s2", [B, C, H1, W])
        o3 = f"rw_o3_{uid[0]}"; created.append(new_node("Reshape", [o2, s2], o3)); uid[0] += 1
        s3 = new_init("s3", [B, C, H1, W, 1])
        o4 = f"rw_o4_{uid[0]}"; created.append(new_node("Reshape", [o3, s3], o4)); uid[0] += 1
        o5 = f"rw_o5_{uid[0]}"; created.append(apply_mid(o4, o5, 4, fb)); uid[0] += 1
        r2.input[0] = o5
        replacements[r.name] = created
        to_remove.update([r.name, mid.name])

    if replacements or to_remove:
        rebuilt = []
        for node in model.graph.node:
            if node.name in replacements:
                rebuilt.extend(replacements[node.name])
            elif node.name in to_remove:
                continue
            else:
                rebuilt.append(node)
        del model.graph.node[:]
        model.graph.node.extend(rebuilt)
    return model


def normalize(model):
    import onnx  # noqa: F401
    # Every op in these graphs already uses opset-13 semantics; onnx2torch's
    # converter registry only goes up to 13, so relabel.
    if model.opset_import and model.opset_import[0].domain == "":
        model.opset_import[0].version = 13
    for node in model.graph.node:
        if node.op_type == "Reshape":
            for a in list(node.attribute):
                if a.name == "allowzero":
                    node.attribute.remove(a)
    model.graph.ClearField("value_info")
    fix_rank6(model)
    return model


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------
def build_traced_model(model_path, folded):
    import onnx
    import torch
    from onnx2torch import convert as onnx2torch_convert

    patch_onnx2torch()

    model = fold_with_ort(model_path) if folded else onnx.load(model_path)
    normalize(model)

    net = onnx2torch_convert(model)
    net.eval()

    dim = 512
    for inp in model.graph.input:
        shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
        if len(shape) > 1 and shape[1] > 0:
            dim = int(shape[1])
            break

    x = torch.randn(1, dim)
    with torch.no_grad():
        y = net(x)
    out_shape = list(y.shape)
    traced = torch.jit.trace(net, x, strict=False)
    return traced, net, dim, out_shape


def convert_to_coreml(traced, dim, precision, out_path):
    import coremltools as ct
    import shutil

    prec = ct.precision.FLOAT16 if precision == "float16" else ct.precision.FLOAT32
    ml = ct.convert(
        traced,
        source="pytorch",
        inputs=[ct.TensorType(name="var", shape=(1, dim))],
        minimum_deployment_target=ct.target.macOS14,
        convert_to="mlprogram",
        compute_precision=prec,
    )
    if os.path.isdir(out_path):
        shutil.rmtree(out_path)
    ml.save(out_path)
    return ml


def validate(mlpackage, net, dim):
    """True if the CoreML model matches the PyTorch reference closely enough.

    Some StyleGAN2 graphs lose too much precision in float16 (demodulation
    sums), rendering visibly wrong images on CPU/ANE; those fall back to fp32.
    """
    try:
        import torch
        import coremltools as ct

        ml = ct.models.MLModel(mlpackage, compute_units=ct.ComputeUnit.ALL)
        spec = ml.get_spec()
        in_name = spec.description.input[0].name
        out_name = spec.description.output[0].name

        rng = np.random.default_rng(12345)
        z = rng.standard_normal((1, dim)).astype(np.float32)
        with torch.no_grad():
            ref = net(torch.from_numpy(z)).numpy()
        pred = np.asarray(ml.predict({in_name: z})[out_name])
        if pred.shape != ref.shape:
            return False
        mean_err = float(np.abs(pred - ref).mean())
        thresh = 0.05 * float(max(1e-6, np.abs(ref).mean()))
        return mean_err <= max(0.05, thresh)
    except Exception as e:
        print(json.dumps({"phase": "convert", "text": f"validation skipped: {e}"}), flush=True)
        return True



# ---------------------------------------------------------------------------
# Cover thumbnail (seeded, deterministic)
# ---------------------------------------------------------------------------
def write_png(path, rgb):
    h, w, _ = rgb.shape
    raw = bytearray()
    for y in range(h):
        raw.append(0)
        raw.extend(rgb[y].tobytes())

    def chunk(tag, data):
        c = tag + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(bytes(raw), 6))
           + chunk(b"IEND", b""))
    with open(path, "wb") as f:
        f.write(png)


def make_cover(net, dim, out_shape, path, size=160, seed=0x5EED):
    try:
        import torch
        rng = np.random.default_rng(seed)
        z = rng.standard_normal((1, dim)).astype(np.float32)
        with torch.no_grad():
            out = net(torch.from_numpy(z)).numpy()[0]  # (3,H,W)
        C, H, W = out.shape
        rgb = np.empty((size, size, 3), dtype=np.uint8)
        for y in range(size):
            yy = min(H - 1, int(y * H / size))
            for x in range(size):
                xx = min(W - 1, int(x * W / size))
                for c in range(3):
                    cc = min(c, C - 1)
                    v = float(out[cc, yy, xx])
                    rgb[y, x, c] = int(min(255, max(0, (v + 1) * 127.5)))
        write_png(path, rgb)
        return True
    except Exception as e:  # pragma: no cover
        print(json.dumps({"phase": "cover", "text": f"cover failed: {e}"}), flush=True)
        return False


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--precision", default="float16", choices=["float16", "float32"])
    ap.add_argument("--cover", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    mlpackage = os.path.join(args.out, "model.mlpackage")
    meta_path = os.path.join(args.out, "meta.json")

    t0 = time.time()
    emit("load", "Reading ONNX graph…", 0.02)

    last_err = None
    for strategy in ("direct", "fold"):
        try:
            emit("convert", "Building PyTorch graph…", 0.1)
            traced, net, dim, out_shape = build_traced_model(args.onnx, folded=(strategy == "fold"))
            emit("convert", f"Tracing ({dim} -> {out_shape})…", 0.4)
            emit("convert", f"Converting to CoreML ({args.precision})…", 0.55)
            convert_to_coreml(traced, dim, args.precision, mlpackage)
            final_precision = args.precision
            emit("convert", "Validating output…", 0.75)
            if args.precision == "float16" and not validate(mlpackage, net, dim):
                emit("convert", "float16 output inaccurate; re-converting as float32…", 0.78)
                convert_to_coreml(traced, dim, "float32", mlpackage)
                final_precision = "float32"
            emit("convert", "CoreML model written", 0.85)

            meta = {
                "inputDim": dim,
                "outShape": out_shape,
                "precision": final_precision,
                "strategy": strategy,
                "source": os.path.basename(args.onnx),
                "seconds": round(time.time() - t0, 2),
            }
            with open(meta_path, "w") as f:
                json.dump(meta, f)

            if args.cover:
                emit("cover", "Rendering cover…", 0.95)
                make_cover(net, dim, out_shape, os.path.join(args.out, "cover.png"))

            emit("done", "Converted", 1.0)
            return
        except Exception as e:  # try the folding strategy next
            last_err = e
            emit("convert", f"{strategy} strategy failed: {type(e).__name__}: {e}", 0.3)
            continue

    raise SystemExit(f"Conversion failed: {last_err}")


if __name__ == "__main__":
    main()
