# tools/inspect_checkpoint.py
import argparse
import os
import re
from collections import OrderedDict
from pathlib import Path

import torch


def inspect_checkpoint(
    ckpt_path: str, output_file: str | None = None, key_filter: str | None = None
):
    """Loads a checkpoint and prints/saves its state dict keys and tensor shapes."""
    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint file not found at {ckpt_path}")
        return

    print(f"--- Inspecting Checkpoint: {Path(ckpt_path).name} ---")
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")

        # Find the actual state dict
        state_dict = None
        potential_keys = ["model", "state_dict", "state_dict_ema"]
        if isinstance(ckpt, dict):
            for key in potential_keys:
                if key in ckpt and isinstance(ckpt[key], dict):
                    state_dict = ckpt[key]
                    print(f"Found state dict under key: '{key}'")
                    break
            if state_dict is None:
                # Check if the top level is the state dict
                if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
                    state_dict = ckpt
                    print("Checkpoint file root appears to be the state dict.")
                else:
                    print(
                        f"Error: Could not find a valid state dict under keys {potential_keys} or at the root."
                    )
                    print(f"Top-level keys found: {list(ckpt.keys())}")
                    return
        elif isinstance(ckpt, OrderedDict) or isinstance(
            ckpt, dict
        ):  # Handle raw state dict files
            state_dict = ckpt
            print("Checkpoint file appears to be a raw state dict.")
        else:
            print(
                f"Error: Checkpoint is not a dictionary or known state dict format (type: {type(ckpt)})."
            )
            return

        if not state_dict:
            print("Error: State dict is empty.")
            return

        print(f"Total keys found: {len(state_dict)}")

        lines_to_write = []
        filtered_count = 0
        pattern = None
        if key_filter:
            try:
                pattern = re.compile(key_filter)
                print(f"Filtering keys using regex: '{key_filter}'")
            except re.error as e:
                print(f"Error compiling regex '{key_filter}': {e}. No filter applied.")
                pattern = None

        # Sort keys for consistent output
        sorted_keys = sorted(state_dict.keys())

        for key in sorted_keys:
            # Apply filter if provided
            if pattern and not pattern.search(key):
                continue

            filtered_count += 1
            shape = tuple(state_dict[key].shape)
            dtype = state_dict[key].dtype
            numel = state_dict[key].numel()
            # Preserve the full key path including 'model.' prefix if it exists
            full_key = key
            if (
                "model" in potential_keys
                and "model" in ckpt
                and isinstance(ckpt["model"], dict)
            ):
                full_key = f"model.{key}"
            line = f"- {full_key:<80} | Shape: {str(shape):<25} | Dtype: {str(dtype):<15} | Numel: {numel}"
            lines_to_write.append(line)

        print(f"\n--- Filtered Keys ({filtered_count}/{len(state_dict)}) ---")
        for line in lines_to_write:
            print(line)

        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "a") as f:  # Changed to append mode
                f.write(f"# Checkpoint Inspection: {Path(ckpt_path).name}\n")
                f.write(f"# Total Keys: {len(state_dict)}\n")
                if pattern:
                    f.write(
                        f"# Filtered Keys ({filtered_count}) using regex: '{key_filter}'\n"
                    )
                else:
                    f.write(f"# Filtered Keys: {filtered_count}\n")
                f.write("-" * 120 + "\n")
                f.write("\n".join(lines_to_write))
                f.write("\n\n")  # Add spacing between different checkpoint inspections
            print(f"\nSaved inspection details to: {output_path}")

        print("-" * 40)

    except Exception as e:
        print(f"Error loading or inspecting checkpoint {ckpt_path}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inspect keys and shapes in PyTorch checkpoint files."
    )
    parser.add_argument(
        "checkpoint_paths", nargs="+", help="Path(s) to the checkpoint file(s)."
    )
    parser.add_argument(
        "-o", "--output", help="Optional path to save the inspection results to a file."
    )
    parser.add_argument("-f", "--filter", help="Optional regex pattern to filter keys.")

    # Add example paths for convenience
    base_path = "/path/to/checkpoints/mFormerV1"
    example_checkpoints = {
        "convnext_t": f"{base_path}/convnext_tiny_22k_1k_384.pth",
        "convnext_s": f"{base_path}/convnext_small_22k_1k_384.pth",
        "convnext_b": f"{base_path}/convnext_base_22k_1k_384.pth",
        "convnext_l": f"{base_path}/convnext_large_22k_1k_384.pth",
        "convnext_xl": f"{base_path}/convnext_xlarge_22k_1k_384_ema.pth",
        "rope_s": f"{base_path}/rope_mixed_deit_small_patch16_LS.pth",
        "rope_b": f"{base_path}/rope_mixed_deit_base_patch16_LS.pth",
        "rope_l": f"{base_path}/rope_mixed_deit_large_patch16_LS.pth",
    }
    parser.epilog = (
        "Example usage:\n"
        "  python tools/inspect_checkpoint.py path/to/ckpt1.pth path/to/ckpt2.pth -f 'attn|mlp' -o inspection.txt\n"
    )
    parser.epilog += "\nTarget Checkpoints for mFormerV1 Variants:\n"
    parser.epilog += f"  sm: {example_checkpoints['convnext_t']} AND {example_checkpoints['rope_s']}\n"
    parser.epilog += f"  md: {example_checkpoints['convnext_s']} AND {example_checkpoints['rope_s']}\n"
    parser.epilog += f"  lg: {example_checkpoints['convnext_l']} AND {example_checkpoints['rope_b']}\n"
    parser.epilog += f"  xl: {example_checkpoints['convnext_xl']} AND {example_checkpoints['rope_l']}\n"

    args = parser.parse_args()

    for path in args.checkpoint_paths:
        inspect_checkpoint(path, args.output, args.filter)

"""
python linnaeus/tools/inspect_checkpoints.py \
    /path/to/checkpoints/mFormerV1/convnext_tiny_22k_1k_384.pth \
    /path/to/checkpoints/mFormerV1/rope_mixed_deit_small_patch16_LS.pth \
    -o inspection_sm.txt

python linnaeus/tools/inspect_checkpoints.py \
    /path/to/checkpoints/mFormerV1/convnext_small_22k_1k_384.pth \
    /path/to/checkpoints/mFormerV1/rope_mixed_deit_small_patch16_LS.pth \
    -o inspection_md.txt

python linnaeus/tools/inspect_checkpoints.py \
    /path/to/checkpoints/mFormerV1/convnext_large_22k_1k_384.pth \
    /path/to/checkpoints/mFormerV1/rope_mixed_deit_base_patch16_LS.pth \
    -o inspection_lg.txt

python linnaeus/tools/inspect_checkpoints.py \
    /path/to/checkpoints/mFormerV1/convnext_xlarge_22k_1k_384_ema.pth \
    /path/to/checkpoints/mFormerV1/rope_mixed_deit_large_patch16_LS.pth \
    -o inspection_xl.txt

python linnaeus/tools/inspect_checkpoints.py \
    /path/to/checkpoints/mFormerV1/convnext_tiny_22k_1k_384.pth \
    -o inspection_convnext_tiny_22k_1k_384.txt
"""
