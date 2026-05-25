import torch
from utils import (
    build_model,
    get_input_ids,
    slow_loop,
    time_generation,
    MODEL_NAME,
    PROFILE_STEPS,
    RESULTS_DIR,
)


def optimized_loop(model, input_ids, n_steps):
    """Generate tokens using the model's KV cache.

    The slow baseline feeds the full growing sequence back through the model on
    every step. This version does one full-prompt prefill, stores the returned
    key/value cache, and then feeds only the newest token on each decode step.
    """
    if n_steps <= 0:
        return []

    generated_tokens = []
    with torch.inference_mode():
        # Prefill: process the whole prompt once and keep its attention KV cache.
        outputs = model(input_ids=input_ids, use_cache=True)
        past_key_values = outputs.past_key_values
        next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        generated_tokens.append(next_token_id)

        # Decode: each later forward pass consumes only the previous token plus
        # the cached K/V tensors from all earlier positions.
        for _ in range(n_steps - 1):
            outputs = model(
                input_ids=next_token_id,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            generated_tokens.append(next_token_id)

    # Avoid `.item()` inside the loop because it synchronizes CPU and GPU every
    # step. Convert once at the end so time_generation can still print a preview.
    return torch.cat(generated_tokens, dim=1).squeeze(0).detach().cpu().tolist()


def profile(loop_fn, model, input_ids, trace_name: str):
    """Profile a short generation run and export a Chrome/Perfetto trace."""
    trace_path = RESULTS_DIR / trace_name
    activities = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]

    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        loop_fn(model, input_ids, PROFILE_STEPS)

    torch.cuda.synchronize()
    print(
        prof.key_averages().table(
            sort_by="cuda_time_total",
            row_limit=20,
        )
    )
    prof.export_chrome_trace(str(trace_path))
    print(f"Chrome trace written to {trace_path}")


def generate_optimized(optimized_trace_name: str) -> float:
    """Build the optimized model, profile it, and return timed latency."""
    # H100, L40S, and RTX PRO 6000 Blackwell all support bfloat16. It reduces
    # memory traffic and is a standard inference dtype on modern NVIDIA GPUs.
    model = build_model(torch.bfloat16)
    model.config.use_cache = True
    input_ids = get_input_ids()

    profile(optimized_loop, model, input_ids, optimized_trace_name)
    elapsed = time_generation(optimized_loop, model, input_ids, "Optimized")

    del model
    torch.cuda.empty_cache()
    return elapsed


def main():
    print("=" * 60)
    print("HW2: LLM Inference Optimization")
    print(f"Model: {MODEL_NAME}")
    print("=" * 60)

    print("\n--- Part 1: Slow baseline ---")
    model = build_model(torch.float32)
    input_ids = get_input_ids()
    profile(slow_loop, model, input_ids, "v0_slow_trace.json")
    slow_elapsed = time_generation(slow_loop, model, input_ids, "Slow")
    del model
    torch.cuda.empty_cache()

    print("\n--- Part 2: Optimized ---")
    optimized_elapsed = generate_optimized(optimized_trace_name="v1_optimized_trace.json")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if optimized_elapsed is None or optimized_elapsed <= 0:
        print("generate_optimized() did not return a positive elapsed time; "
              "cannot compute speedup.")
    else:
        speedup = slow_elapsed / optimized_elapsed
        print(f"  Slow:      {slow_elapsed:6.2f}s")
        print(f"  Optimized: {optimized_elapsed:6.2f}s")
        print(f"  Speedup:   {speedup:6.2f}x  (vs V0 slow baseline)")


if __name__ == "__main__":
    main()


# ============================================================================
# Writeup
# ============================================================================
#
# Changes made and speedup per fix:
#
#
# Biggest impact and why:
#
