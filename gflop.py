"""Measure StarDist GFLOPs via TF graph profiling."""
import numpy as np
import tensorflow as tf
from predict import load_model

m = load_model()
km = m.keras_model

n_params = sum(int(np.prod(w.shape)) for w in km.weights)
print(f"Parameters: {n_params:,} ({n_params / 1e6:.2f}M)")

# Build a concrete function for profiling
inp_spec = tf.TensorSpec((1, 256, 256, 3), tf.float32)

@tf.function(input_signature=[inp_spec])
def fwd(x):
    return km(x)

cf = fwd.get_concrete_function()

# Profile FLOPs
try:
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
    frozen = convert_variables_to_constants_v2(cf)
    graph_def = frozen.graph

    run_meta = tf.compat.v1.RunMetadata()
    opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()

    _ = tf.compat.v1.profiler.profile(
        graph_def,
        run_meta=run_meta,
        cmd="scope",
        options=opts,
    )
except Exception:
    pass

# Direct FLOP count from the graph
total_flops = 0
for op in cf.graph.get_operations():
    if op.type in ("MatMul", "Conv2D", "Conv2DBackpropInput"):
        # These ops have _FLOPs attribute or we estimate from shapes
        flops_attr = None
        for attr_name in ["_FLOPs", "flops"]:
            try:
                flops_attr = op.get_attr(attr_name)
            except (ValueError, KeyError):
                pass
        if flops_attr is not None:
            total_flops += int(flops_attr)

if total_flops > 0:
    print(f"\nTotal FLOPs (graph): {total_flops / 1e9:.2f} GFLOPs (at 256x256)")
else:
    # Fallback: estimate from layer shapes
    print("\nGraph FLOP attrs not found, estimating from layers...")
    total_macs = 0
    for layer in km.layers:
        cls = type(layer).__name__
        if cls == "InputLayer":
            continue
        try:
            out_shape = layer.output_shape
        except (AttributeError, ValueError, TypeError):
            continue
        if not isinstance(out_shape, (tuple, list)) or len(out_shape) != 4:
            continue

        if cls == "Conv2D" and hasattr(layer, "kernel"):
            k = layer.kernel.shape
            oh, ow = out_shape[1], out_shape[2]
            if oh is None or ow is None:
                continue
            macs = int(k[0]) * int(k[1]) * int(k[2]) * int(k[3]) * int(oh) * int(ow)
            total_macs += macs
            if macs > 1e6:
                print(f"  {layer.name:40s} {oh}x{ow}x{k[2]}->{k[3]}  {macs/1e6:.1f}MMACs")

    flops = total_macs * 2
    print(f"\nEstimated FLOPs: {flops / 1e9:.2f} GFLOPs (at 256x256)")
