import torch
from statistics import median


# ============================================================================
# Part 1: Implement PyTorch Functions
# ============================================================================
#
# TASK 1a: Implement an operation with the lowest arithmetic intensity.
# Use an op that performs essentially memory traffic with ~0 useful FLOPs
# per element.


def lowest_ai_fn(x: torch.Tensor) -> torch.Tensor:
    """Lowest arithmetic intensity baseline (0 FLOP/Byte)."""
    # clone() copies the tensor to a new output tensor.
    # That means the GPU mostly reads x and writes the result; it does not do
    # meaningful floating-point arithmetic. This is a good "memory traffic only"
    # baseline for the left side of the roofline plot.
    return x.clone()


# TASK 1b: Implement a function with configurable arithmetic intensity.
# Build an element-wise compute operation where work increases with `num_ops`.
# Design it so fused arithmetic intensity grows roughly linearly with `num_ops`,
# while each element is still read/written once at the kernel boundary.
# Return either the eager function or a compiled version depending on the
# `compiled` flag so we can compare both on the roofline plot.
#
# Use an accumulator variable and implement fused multiply-add (FMA) style work
# explicitly, e.g. `acc = acc * x + x`, so each loop iteration contributes
# about 2 FLOPs per element in a realistic GPU-friendly pattern. We prefer this
# pattern here mainly because it gives clean FLOP accounting and resembles the
# kind of floating-point work GPUs are designed to do; Avoid patterns like repeated
# doubling (`x = x + x`), since long self-dependent pointwise chains can trigger
# very poor Inductor compile-time behavior and are also less useful for this
# roofline exercise.


def make_compute_fn(num_ops: int, compiled: bool = True):
    """Return an eager or compiled function whose work scales with num_ops."""

    def fn(x: torch.Tensor) -> torch.Tensor:
        # acc is the running per-element value. Each loop iteration performs:
        #   1 multiply: acc * x
        #   1 add:      (...) + x
        # So each iteration contributes 2 FLOPs per tensor element.
        acc = x
        for _ in range(num_ops):
            acc = acc * x + x
        return acc

    # In eager mode, PyTorch usually launches separate kernels for the multiply
    # and add operations. In compiled mode, torch.compile can fuse the whole
    # pointwise chain into a much smaller number of kernels, often one kernel.
    # That fusion is the reason compiled arithmetic intensity grows with num_ops.
    return torch.compile(fn) if compiled else fn


# ============================================================================
# Part 2: Benchmarking
# ============================================================================
#
# TASK 2: Complete the benchmark function using CUDA events.
# CUDA events measure GPU time precisely (not CPU wall time), which avoids
# including kernel launch overhead or CPU-GPU synchronization delays.


def benchmark_fn(fn, *args, warmup=25, rep=100) -> float:
    """Benchmark a GPU function using CUDA events.

    Returns median execution time in milliseconds.
    """
    # Warmup (triggers torch.compile on first call, then warms caches)
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    # CUDA kernels launch asynchronously: the CPU can continue before the GPU
    # has finished the work. CUDA events are recorded on the GPU timeline, so
    # elapsed_time() measures GPU execution time instead of Python wall-clock
    # overhead.
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]

    for i in range(rep):
        # Record one start/end event pair around each invocation. The actual
        # timing values are only valid after the final synchronize below.
        start_events[i].record()
        fn(*args)
        end_events[i].record()

    torch.cuda.synchronize()
    times_ms = [start.elapsed_time(end) for start, end in zip(start_events, end_events)]

    # The median is more stable than the mean when one run is unusually slow
    # because of a transient system effect.
    return median(times_ms)


# TASK 3: Compute element-wise operation metrics from measured runtime.
# Count every arithmetic operation performed inside the loop (careful: each
# `acc = acc * x + x` iteration does more than one FLOP per element).
#
# Use different byte-traffic models for the two variants:
#   - compiled: assume the operation is fused, so each element is read once and
#     written once at the kernel boundary
#   - eager: estimate the traffic from the separate multiply and add operations
#     launched by PyTorch in each loop iteration, including intermediate tensors
#
# Return a tuple with:
#   - total_flops
#   - arithmetic_intensity  (FLOP / Byte)
#   - achieved_flops        (FLOP / s)


def compute_elementwise_metrics(num_elements, num_ops, bytes_per_element, ms, variant):
    # The loop body is `acc = acc * x + x`.
    # Per element, per loop iteration:
    #   multiply = 1 FLOP
    #   add      = 1 FLOP
    # Therefore total FLOPs = 2 * number of elements * number of iterations.
    total_flops = 2 * num_elements * num_ops

    if variant == "compiled":
        # Fused compiled model:
        # torch.compile can keep intermediate values inside registers. At the
        # kernel boundary, the idealized traffic is just one read of x and one
        # write of the final output.
        total_bytes = 2 * num_elements * bytes_per_element
    elif variant == "eager":
        # Eager model:
        # Each loop iteration is approximated as two separate pointwise kernels:
        #   multiply: read acc, read x, write intermediate = 3 element transfers
        #   add:      read intermediate, read x, write acc = 3 element transfers
        # That gives 6 element transfers per iteration.
        total_bytes = 6 * num_elements * bytes_per_element * num_ops
    else:
        raise ValueError(f"Unknown element-wise variant: {variant}")

    # Arithmetic intensity says how much computation we get per byte moved.
    # Achieved FLOP/s says how much arithmetic the measured runtime delivered.
    ai = total_flops / total_bytes
    achieved_flops = total_flops / (ms * 1e-3)
    return total_flops, ai, achieved_flops


# ============================================================================
# Part 3: Short Writeup
# ============================================================================
# Answer these after you generate `results/roofline.png` and inspect the points.
#
# Q1. Look at the compiled element-wise operations from `1 ops` through `64 ops`.
# Why does performance rise as arithmetic intensity increases even though the
# measured runtime changes only a little?
#
# Q2. In one sample run, `matmul 1024x1024` achieved lower FLOP/s than the
# `128 ops` compiled element-wise operation. Give one or two reasons why that can
# happen on a large GPU like an H100.
#
# Q3. Between `64 ops` and `128 ops`, runtime increases more noticeably than it
# did for smaller operations. What does that suggest about what resource is
# becoming the bottleneck?
#
# Q4. Why do the eager `ops-K` points look so different from the compiled ones?
