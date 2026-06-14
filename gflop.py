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

# Profile FLOPs via TF v1 profiler
run_meta = tf.compat.v1.RunMetadata()
opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
result = tf.compat.v1.profiler.profile(
    cf.graph,
    run_meta=run_meta,
    cmd="scope",
    options=opts,
)

total_flops = result.total_float_ops
print(f"\nTotal FLOPs: {total_flops / 1e9:.2f} GFLOPs (at 256x256)")
print(f"Parameters:  {n_params / 1e6:.2f}M")
