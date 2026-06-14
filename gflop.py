"""Quick StarDist param count and FLOPs estimate."""
import numpy as np
import tensorflow as tf
from predict import load_model

m = load_model()
km = m.keras_model

n_params = sum(int(np.prod(w.shape)) for w in km.weights)
print(f"Parameters: {n_params:,} ({n_params / 1e6:.2f}M)")

# Run a forward pass to get actual output shapes per layer
inp = np.zeros((1, 256, 256, 3), dtype=np.float32)

# Extract intermediate output shapes by building a feature model
layer_outputs = {}
for layer in km.layers:
    if type(layer).__name__ == "InputLayer":
        continue
    try:
        layer_outputs[layer.name] = layer.output_shape
    except (AttributeError, ValueError):
        pass

total_macs = 0
for layer in km.layers:
    cls = type(layer).__name__
    out_shape = layer_outputs.get(layer.name)
    if out_shape is None:
        continue

    if not isinstance(out_shape, (tuple, list)) or len(out_shape) < 3:
        continue

    if cls == "Conv2D":
        kh, kw = layer.kernel.shape[:2]
        in_ch = layer.kernel.shape[2] if len(layer.kernel.shape) == 4 else 1
        out_ch = layer.kernel.shape[3] if len(layer.kernel.shape) == 4 else layer.kernel.shape[-1]
        oh = out_shape[1] if out_shape[1] else 256
        ow = out_shape[2] if out_shape[2] else 256
        macs = int(kh) * int(kw) * int(in_ch) * int(out_ch) * oh * ow
        total_macs += macs
        if macs > 1e6:
            print(f"  {layer.name:40s} {cls:20s} {oh}x{ow}x{in_ch}->{out_ch}  {macs/1e6:.1f}MMACs")
    elif cls == "DepthwiseConv2D":
        kh, kw = layer.kernel.shape[:2]
        channels = layer.kernel.shape[2] if len(layer.kernel.shape) == 3 else 1
        oh = out_shape[1] if out_shape[1] else 256
        ow = out_shape[2] if out_shape[2] else 256
        macs = int(kh) * int(kw) * int(channels) * oh * ow
        total_macs += macs
        if macs > 1e6:
            print(f"  {layer.name:40s} {cls:20s} {oh}x{ow}x{channels}      {macs/1e6:.1f}MMACs")
    elif cls in ("Dense",):
        units = layer.units
        in_dim = int(layer.input.shape[-1]) if layer.input.shape[-1] else 0
        macs = in_dim * units
        total_macs += macs
        if macs > 1e6:
            print(f"  {layer.name:40s} {cls:20s} {in_dim}->{units}     {macs/1e6:.1f}MMACs")

flops = total_macs * 2
print(f"\nConv2D MACs: {total_macs / 1e9:.2f} GMACs")
print(f"Estimated FLOPs: {flops / 1e9:.2f} GFLOPs (at 256x256)")
print(f"Note: Excludes custom/non-standard layers (StarDist polygon head, NMS).")
