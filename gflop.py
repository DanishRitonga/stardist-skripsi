"""Quick StarDist param count and FLOPs estimate."""
import numpy as np
from predict import load_model

m = load_model()
km = m.keras_model

n_params = sum(int(np.prod(w.shape)) for w in km.weights)
print(f"Parameters: {n_params:,} ({n_params / 1e6:.2f}M)")

# Print all layers with shapes to estimate FLOPs
total_macs = 0
for layer in km.layers:
    cls = type(layer).__name__
    try:
        out_shape = layer.output.shape
        in_shape = layer.input.shape
    except (AttributeError, ValueError):
        continue

    if cls == "Conv2D":
        kh, kw = layer.kernel.shape[:2]
        in_ch = layer.kernel.shape[2] if len(layer.kernel.shape) == 4 else 1
        out_ch = layer.kernel.shape[3] if len(layer.kernel.shape) == 4 else layer.kernel.shape[-1]
        oh = int(out_shape[1]) if out_shape[1] else 256
        ow = int(out_shape[2]) if out_shape[2] else 256
        macs = int(kh) * int(kw) * int(in_ch) * int(out_ch) * oh * ow
        total_macs += macs
        if macs > 1e6:
            print(f"  {layer.name:40s} {cls:20s} {oh}x{ow}x{in_ch}->{out_ch}  {macs/1e6:.1f}MMACs")
    elif cls == "DepthwiseConv2D":
        kh, kw = layer.kernel.shape[:2]
        channels = layer.kernel.shape[2] if len(layer.kernel.shape) == 3 else 1
        oh = int(out_shape[1]) if out_shape[1] else 256
        ow = int(out_shape[2]) if out_shape[2] else 256
        macs = int(kh) * int(kw) * int(channels) * oh * ow
        total_macs += macs
        if macs > 1e6:
            print(f"  {layer.name:40s} {cls:20s} {oh}x{ow}x{channels}      {macs/1e6:.1f}MMACs"
                  )
    elif cls in ("Dense",):
        units = layer.units
        in_dim = int(in_shape[-1]) if in_shape[-1] else 0
        macs = in_dim * units
        total_macs += macs
        if macs > 1e6:
            print(f"  {layer.name:40s} {cls:20s} {in_dim}->{units}     {macs/1e6:.1f}MMACs")

flops = total_macs * 2  # MACs → FLOPs (multiply + add)
print(f"\nConv2D MACs: {total_macs / 1e9:.2f} GMACs")
print(f"Estimated FLOPs: {flops / 1e9:.2f} GFLOPs (at 256x256)")
print(f"Note: Excludes custom/non-standard layers (StarDist polygon head, NMS).")
