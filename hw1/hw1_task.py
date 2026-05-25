"""Entrypoint for running the HW1 roofline benchmark.

This is the script you execute from the repository root:

    python3 hw1/hw1_task.py

It does not implement the homework logic itself. Instead, it connects two other
files:

- `hw1_task_impl.py`: your implementation of the benchmarked functions and
  measurement formulas.
- `hw1_runtime.py`: provided utilities that run the benchmark, save JSON data,
  and generate the roofline PNG plot.

When this script finishes successfully, it creates:

- `hw1/results/roofline_data.json`
- `hw1/results/roofline.png`
"""

# Import the runtime helpers that know how to benchmark, save, and plot results.
from hw1_runtime import (
    measure_roofline_points,
    plot_roofline,
    print_header,
    save_roofline_data,
)


def run_hw1(
    lowest_ai_fn,
    make_compute_fn,
    benchmark_fn,
    compute_elementwise_metrics,
):
    """Run the full HW1 benchmark pipeline.

    The four arguments are functions from `hw1_task_impl.py`. Passing them in as
    arguments keeps this entrypoint small and makes the runtime reusable for
    grading or testing.
    """
    # Print the detected GPU and theoretical roofline constants.
    print_header()

    # This line is a reminder that the first compiled benchmark may spend extra
    # time compiling kernels before the actual timed measurements happen.
    print("Running benchmarks (first run compiles kernels via torch.compile)...")

    # Measure all benchmark points:
    #   1. lowest-arithmetic-intensity baseline
    #   2. eager element-wise operations
    #   3. compiled element-wise operations
    #   4. matrix multiplication reference points
    results = measure_roofline_points(
        lowest_ai_fn=lowest_ai_fn,
        make_compute_fn=make_compute_fn,
        benchmark_fn=benchmark_fn,
        compute_elementwise_metrics=compute_elementwise_metrics,
    )

    # Save the raw result dictionaries to hw1/results/roofline_data.json.
    save_roofline_data(results)

    # Generate the visual roofline plot from the same result dictionaries.
    print("\nGenerating plots...")
    plot_roofline(results)

    print("\nDone! Check the results/ directory for plots.")


def main():
    """Import the student's implementations and run the benchmark."""
    # Import inside main() so this file can be imported without immediately
    # loading or executing the homework implementation.
    from hw1_task_impl import (
        benchmark_fn,
        compute_elementwise_metrics,
        lowest_ai_fn,
        make_compute_fn,
    )

    # Pass the implementation functions into the runtime pipeline.
    run_hw1(
        lowest_ai_fn=lowest_ai_fn,
        make_compute_fn=make_compute_fn,
        benchmark_fn=benchmark_fn,
        compute_elementwise_metrics=compute_elementwise_metrics,
    )


# Only run the benchmark when this file is executed as a script. This prevents
# accidental benchmark runs if another file imports `hw1_task.py`.
if __name__ == "__main__":
    main()
