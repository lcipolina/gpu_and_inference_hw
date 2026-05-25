"""Runtime utilities for HW1's GPU roofline experiment.

This file is the "driver" support code for the homework. It does not contain
the functions you are graded on; those live in `hw1_task_impl.py`. Instead, this
module:

1. Detects the GPU and loads its theoretical peak FP32 compute and bandwidth.
2. Allocates CUDA tensors for benchmark inputs.
3. Calls the functions implemented in `hw1_task_impl.py`.
4. Collects timing, FLOP, bandwidth, and arithmetic-intensity measurements.
5. Saves the raw measurements to `hw1/results/roofline_data.json`.
6. Draws the roofline plot to `hw1/results/roofline.png`.

The main entrypoint `hw1_task.py` imports this module and orchestrates the full
run. In normal use, run `python3 hw1/hw1_task.py` from the repository root.
"""

import json
from pathlib import Path

import matplotlib
import numpy as np
import torch

# Use a non-interactive matplotlib backend so plotting works on remote machines
# and Colab runtimes without opening a GUI window.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Store generated artifacts next to this homework.
RESULTS_DIR = Path(__file__).parent / "results"

# Create the results directory if it does not already exist. This is why the
# script can write `roofline.png` and `roofline_data.json` on a fresh checkout.
RESULTS_DIR.mkdir(exist_ok=True)

# The roofline needs theoretical hardware ceilings. The keys are substrings we
# expect to find inside `torch.cuda.get_device_name(0)`.
GPU_SPECS = {
    "H100": {
        "label": "NVIDIA H100 80GB HBM3",
        # Peak FP32 throughput in FLOP/s, excluding Tensor Cores.
        "peak_flops": 67e12,
        # Peak memory bandwidth in byte/s.
        "peak_bw": 3.35e12,
    },
    "L40S": {
        "label": "NVIDIA L40S 48GB GDDR6",
        "peak_flops": 91.6e12,
        "peak_bw": 864e9,
    },
    "RTX PRO 6000 Blackwell": {
        "label": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
        "peak_flops": 120e12,
        "peak_bw": 1597e9,
    },
}


def _get_gpu_specs():
    """Return the theoretical roofline specs for the active CUDA device."""
    # The whole homework benchmark is CUDA-specific. If CUDA is not visible,
    # there is no GPU timeline to benchmark and no device name to match.
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please set the GPU roofline settings yourself.")

    # Ask PyTorch what GPU Colab/Nebius/local CUDA actually assigned.
    device_name = torch.cuda.get_device_name(0)

    # Match by substring rather than exact string because cloud providers often
    # append extra words, for example "Server Edition".
    for gpu_key, specs in GPU_SPECS.items():
        if gpu_key in device_name:
            return specs

    # If this happens, add a new GPU_SPECS entry with FP32 peak and bandwidth.
    raise RuntimeError(
        f"Unsupported GPU '{device_name}'. Please set the roofline GPU settings yourself."
    )


# Load GPU constants once when the module is imported. The plotting and printed
# header use these global values throughout the run.
GPU_INFO = _get_gpu_specs()
GPU_LABEL = GPU_INFO["label"]
PEAK_FLOPS = GPU_INFO["peak_flops"]
PEAK_BW = GPU_INFO["peak_bw"]

# The ridge point is where the roofline changes from memory-bound to
# compute-bound: bandwidth * arithmetic_intensity == peak_compute.
RIDGE_POINT = PEAK_FLOPS / PEAK_BW


def print_header():
    """Print the GPU roofline constants for this run."""
    print("=" * 70)
    print(f"HW1: GPU Roofline Model — {GPU_LABEL}")
    print("=" * 70)
    print("\nTheoretical specs (FP32):")
    print(f"  Peak compute:    {PEAK_FLOPS / 1e12:.0f} TFLOP/s")
    print(f"  Peak bandwidth:  {PEAK_BW / 1e12:.2f} TB/s")
    print(f"  Ridge point:     {RIDGE_POINT:.1f} FLOP/Byte")
    print()


def measure_roofline_points(
    lowest_ai_fn,
    make_compute_fn,
    benchmark_fn,
    compute_elementwise_metrics,
):
    """Run operations with varying arithmetic intensity and measure performance."""
    # Use a large vector so the kernels do enough work to measure reliably.
    # float32 has 4 bytes/element, so 64M elements is 256 MB of input data.
    n = 64 * 1024 * 1024  # 64M elements = 256 MB in float32

    # Allocate the benchmark input directly on the GPU. All measured functions
    # operate on this CUDA tensor.
    x = torch.randn(n, device="cuda", dtype=torch.float32)

    bytes_per_element = 4  # float32

    # For a simple copy-like operation, the idealized traffic is one read from x
    # and one write to the output.
    total_transfer_bytes = n * 2 * bytes_per_element  # read + write

    # Every measurement is stored as one dictionary. This list later becomes the
    # JSON file and also feeds the plotting function.
    results = []

    # Benchmark lowest-AI baseline function
    ms = benchmark_fn(lowest_ai_fn, x)

    # Bandwidth = bytes moved / seconds. `ms` is milliseconds, so convert to
    # seconds with `ms * 1e-3`.
    achieved_bw = total_transfer_bytes / (ms * 1e-3)

    # A true 0-FLOP point cannot be shown on a log-log plot because log(0) is
    # undefined. We place it at AI=0.01 as a visual baseline near the left edge.
    results.append(
        {
            "name": "lowest-AI (~0 FLOPs)",
            "series": "baseline",
            "variant": "baseline",
            "num_ops": 0,
            "arithmetic_intensity": 0.01,
            "achieved_flops": achieved_bw * 0.01,
            "achieved_bw": achieved_bw,
            "ms": ms,
        }
    )
    print(f"  lowest-AI:  {ms:.3f} ms | BW: {achieved_bw / 1e12:.2f} TB/s")

    # Benchmark compute functions with varying arithmetic intensity.
    # We compare eager and compiled versions to show how fusion changes the
    # measured roofline position.
    ops_list = [1, 2, 4, 8, 16, 32, 64, 128]
    for num_ops in ops_list:
        # Build two versions of the same mathematical operation:
        # eager_fn shows normal PyTorch execution,
        # compiled_fn shows the effect of torch.compile fusion.
        eager_fn = make_compute_fn(num_ops, compiled=False)
        compiled_fn = make_compute_fn(num_ops, compiled=True)

        for variant, compute_fn in [("eager", eager_fn), ("compiled", compiled_fn)]:
            # Time this particular function variant on the GPU.
            ms = benchmark_fn(compute_fn, x)

            # Convert runtime into roofline coordinates:
            # x-axis = arithmetic intensity, y-axis = achieved FLOP/s.
            total_flops, ai, achieved_flops = compute_elementwise_metrics(
                num_elements=n,
                num_ops=num_ops,
                bytes_per_element=bytes_per_element,
                ms=ms,
                variant=variant,
            )

            results.append(
                {
                    "name": f"{num_ops} ops",
                    "series": "elementwise",
                    "variant": variant,
                    "num_ops": num_ops,
                    "arithmetic_intensity": ai,
                    "achieved_flops": achieved_flops,
                    # Effective bandwidth derived from achieved FLOP/s and AI:
                    # FLOP/s divided by FLOP/byte equals byte/s.
                    "achieved_bw": achieved_flops / ai,
                    "ms": ms,
                }
            )
            print(
                f"  {num_ops:>3d} ops ({variant:>8}): {ms:.3f} ms | "
                f"AI: {ai:.3g} FLOP/B | {achieved_flops / 1e12:.2f} TFLOP/s"
            )

    # Benchmark matrix multiplication (very high arithmetic intensity)
    for m in [1024, 2048, 4096]:
        # Square matrices of increasing size give high-arithmetic-intensity
        # reference points using PyTorch's optimized matrix multiply.
        a = torch.randn(m, m, device="cuda", dtype=torch.float32)
        b = torch.randn(m, m, device="cuda", dtype=torch.float32)

        # Bind a and b as default arguments so benchmark_fn can call fn() with
        # no additional arguments.
        fn = lambda a=a, b=b: torch.mm(a, b)
        ms = benchmark_fn(fn, warmup=25, rep=100)

        # Dense matrix multiply does approximately 2*m^3 FLOPs:
        # one multiply and one add for each inner product term.
        total_flops = 2 * m * m * m

        # Minimal traffic model: read A, read B, write C.
        total_bytes_mm = (2 * m * m + m * m) * bytes_per_element
        ai = total_flops / total_bytes_mm
        achieved_flops = total_flops / (ms * 1e-3)

        results.append(
            {
                "name": f"n={m}",
                "series": "matmul",
                "variant": "library",
                "num_ops": -1,
                "arithmetic_intensity": ai,
                "achieved_flops": achieved_flops,
                "achieved_bw": total_bytes_mm / (ms * 1e-3),
                "ms": ms,
            }
        )
        print(
            f"  matmul {m}×{m}: {ms:.3f} ms | AI: {ai:.1f} FLOP/B | "
            f"{achieved_flops / 1e12:.2f} TFLOP/s"
        )

    return results


def plot_roofline(results):
    """Create a roofline diagram with theoretical ceilings and measured points."""
    # Create one matplotlib figure and axis.
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))

    # Arithmetic intensity values for drawing the theoretical curve. A log-spaced
    # range makes the roofline readable across many orders of magnitude.
    ai_range = np.logspace(-2, 4, 500)

    # Memory roof: performance grows linearly with AI while memory bandwidth is
    # the bottleneck. Units: byte/s * FLOP/byte = FLOP/s.
    mem_ceiling = PEAK_BW * ai_range

    # Compute roof: once arithmetic intensity is high enough, performance is
    # capped by peak FP32 throughput.
    compute_ceiling = np.full_like(ai_range, PEAK_FLOPS)

    # The actual roofline is the lower of the memory and compute ceilings.
    roofline = np.minimum(mem_ceiling, compute_ceiling)

    # Plot all three theoretical lines on log-log axes.
    ax.loglog(ai_range, roofline, "k-", linewidth=2.5, label="Roofline ceiling")
    ax.loglog(
        ai_range,
        mem_ceiling,
        "b--",
        linewidth=1,
        alpha=0.5,
        label=f"Memory BW ceiling ({PEAK_BW / 1e12:.2f} TB/s)",
    )
    ax.loglog(
        ai_range,
        compute_ceiling,
        "r--",
        linewidth=1,
        alpha=0.5,
        label=f"Compute ceiling ({PEAK_FLOPS / 1e12:.0f} TFLOP/s)",
    )

    # Split the raw result dictionaries into visual groups so each series gets
    # its own marker color and legend label.
    baseline_results = [r for r in results if r.get("series") == "baseline"]
    eager_results = [r for r in results if r.get("series") == "elementwise" and r.get("variant") == "eager"]
    compiled_results = [
        r for r in results if r.get("series") == "elementwise" and r.get("variant") == "compiled"
    ]
    mm_results = [r for r in results if r.get("series") == "matmul" or r["num_ops"] == -1]

    if baseline_results:
        # Plot the memory-only baseline. max(..., 1e6) keeps the point visible on
        # the log y-axis even if the fake FLOP/s value is tiny.
        ais = [r["arithmetic_intensity"] for r in baseline_results]
        flops = [max(r["achieved_flops"], 1e6) for r in baseline_results]
        ax.scatter(
            ais,
            flops,
            c="gray",
            s=80,
            zorder=5,
            edgecolors="black",
            linewidths=0.5,
            label="Lowest-AI baseline",
        )
        for r in baseline_results:
            # Add a small label near the point with the operation name and time.
            f = max(r["achieved_flops"], 1e6)
            label = f"{r['name']}\n{r['ms']:.3f} ms"
            ax.annotate(
                label,
                (r["arithmetic_intensity"], f),
                textcoords="offset points",
                xytext=(8, -5),
                fontsize=7,
                color="gray",
            )

    if eager_results:
        # Eager element-wise measurements usually cluster at low AI because each
        # PyTorch operation materializes intermediates in global memory.
        ais = [r["arithmetic_intensity"] for r in eager_results]
        flops = [max(r["achieved_flops"], 1e6) for r in eager_results]
        ax.scatter(
            ais,
            flops,
            c="orange",
            s=80,
            zorder=5,
            edgecolors="black",
            linewidths=0.5,
            label="Element-wise eager (estimated AI)",
        )
        # Alternate label offsets to reduce text overlap between neighboring
        # eager points.
        eager_offsets = [(0, 8), (0, -16)]
        for i, r in enumerate(eager_results):
            f = max(r["achieved_flops"], 1e6)
            label = f"{r['num_ops']} ops\n{r['ms']:.2f} ms"
            ax.annotate(
                label,
                (r["arithmetic_intensity"], f),
                textcoords="offset points",
                xytext=eager_offsets[i % len(eager_offsets)],
                ha="center",
                fontsize=6,
                color="orange",
            )

    if compiled_results:
        # Compiled element-wise measurements should move to the right as num_ops
        # increases because fusion keeps memory traffic roughly fixed while FLOPs
        # increase.
        ais = [r["arithmetic_intensity"] for r in compiled_results]
        flops = [max(r["achieved_flops"], 1e6) for r in compiled_results]
        ax.scatter(
            ais,
            flops,
            c="blue",
            s=80,
            zorder=5,
            edgecolors="black",
            linewidths=0.5,
            label="Element-wise compiled (fused AI)",
        )
        for r in compiled_results:
            f = max(r["achieved_flops"], 1e6)
            label = f"{r['num_ops']} ops\n{r['ms']:.3f} ms"
            ax.annotate(
                label,
                (r["arithmetic_intensity"], f),
                textcoords="offset points",
                xytext=(8, -10),
                fontsize=7,
                color="blue",
            )

    if mm_results:
        # Matrix multiply points are high-AI reference operations from PyTorch's
        # optimized library kernels.
        ais = [r["arithmetic_intensity"] for r in mm_results]
        flops = [r["achieved_flops"] for r in mm_results]
        ax.scatter(
            ais,
            flops,
            c="red",
            s=100,
            marker="D",
            zorder=5,
            edgecolors="black",
            linewidths=0.5,
            label="Matrix multiply",
        )
        for r in mm_results:
            label = f"{r['name']}\n{r['ms']:.3f} ms"
            ax.annotate(
                label,
                (r["arithmetic_intensity"], r["achieved_flops"]),
                textcoords="offset points",
                xytext=(8, -5),
                fontsize=7,
                color="red",
            )

    # Configure labels, ranges, legend, and grid for a readable roofline plot.
    ax.set_xlabel("Arithmetic Intensity (FLOP/Byte)", fontsize=12)
    ax.set_ylabel("Performance (FLOP/s)", fontsize=12)
    ax.set_title(f"Roofline Model — {GPU_LABEL} (FP32)", fontsize=14)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(1e-2, 1e4)
    ax.set_ylim(1e9, 2e14)
    ax.grid(True, which="both", alpha=0.2)

    # Save the PNG artifact under hw1/results.
    plt.tight_layout()
    path = RESULTS_DIR / "roofline.png"
    fig.savefig(path, dpi=150)
    print(f"\nRoofline plot saved to {path}")
    plt.close()


def save_roofline_data(results):
    """Write the raw benchmark result dictionaries to JSON."""
    # The JSON file is useful for inspecting exact numbers after the plot is
    # generated, and it is easier to paste/share than the PNG.
    with open(RESULTS_DIR / "roofline_data.json", "w") as f:
        json.dump(results, f, indent=2)
