# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
End-to-end test for K_CACHE/Q_ATTN Hadamard rotation with a real R3 checkpoint.

Uses nm-testing/Meta-Llama-3-8B-Instruct-spinquantR3, which has:
  - transform_config: hadamard, head_dim=128
  - apply: q_attn + k_cache, targets re:.*self_attn$

Verifies that the model loads and runs inference without error.
Per-layer flag dispatch (_kq_attn_transform) is covered by the unit
tests in test_hadamard_kv_dispatch.py.

Run with rotation (default):
  .venv/bin/python tests/quantization/test_hadamard_kq_r3_e2e.py

Run without rotation (baseline):
  .venv/bin/python tests/quantization/test_hadamard_kq_r3_e2e.py --no-rotation

Requires ~16GB VRAM (RTX 3090 or equivalent). Not suitable for CI.
"""

import argparse
import os
import time

MODEL = "nm-testing/Meta-Llama-3-8B-Instruct-spinquantR3"

PROMPTS = [
    "The capital of France is",
    "The theory of relativity states that",
    "In machine learning, a transformer model",
    "The largest ocean on Earth is",
    "To make a cup of tea, you need to",
]

MAX_TOKENS = 64


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-rotation",
        action="store_true",
        help="Disable K_CACHE/Q_ATTN Hadamard rotation (baseline run).",
    )
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available.")
        return

    from vllm import LLM, SamplingParams

    rotation = not args.no_rotation
    print(f"\n{'=' * 60}")
    print(f"Model : {MODEL}")
    print(f"Rotation : {'ENABLED' if rotation else 'DISABLED (baseline)'}")
    print(f"{'=' * 60}\n")

    # For the baseline run, load the model from a temp directory with
    # transform_config stripped from config.json so no rotation is applied.
    import json
    import tempfile

    from huggingface_hub import snapshot_download

    model_path = snapshot_download(MODEL)

    if not rotation:
        tmpdir = tempfile.mkdtemp()
        # Copy everything except config.json
        for item in os.listdir(model_path):
            src = os.path.join(model_path, item)
            dst = os.path.join(tmpdir, item)
            if item == "config.json":
                with open(src) as f:
                    config = json.load(f)
                config.get("quantization_config", {}).pop("transform_config", None)
                with open(dst, "w") as f:
                    json.dump(config, f, indent=2)
            else:
                os.symlink(src, dst)
        load_path = tmpdir
    else:
        load_path = model_path

    llm = LLM(
        model=load_path,
        dtype="bfloat16",
    )

    sampling_params = SamplingParams(max_tokens=MAX_TOKENS, temperature=0.0)

    # Warmup
    print("Warming up...")
    llm.generate(PROMPTS[:1], sampling_params)

    # Timed run
    print(f"Running {len(PROMPTS)} prompts x {MAX_TOKENS} tokens...\n")
    t0 = time.perf_counter()
    outputs = llm.generate(PROMPTS, sampling_params)
    elapsed = time.perf_counter() - t0

    total_output_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    throughput = total_output_tokens / elapsed

    print(f"{'=' * 60}")
    print(f"Throughput : {throughput:.1f} output tok/s")
    print(f"Total tokens : {total_output_tokens} in {elapsed:.2f}s")
    print(f"{'=' * 60}\n")

    print("Sample outputs:")
    for i, (prompt, output) in enumerate(zip(PROMPTS, outputs)):
        text = output.outputs[0].text
        print(f"  [{i + 1}] {prompt!r}")
        print(f"       -> {text!r}")
        print()


if __name__ == "__main__":
    main()
