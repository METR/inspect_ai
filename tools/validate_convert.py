"""Validate that converting an .eval file preserves all resolved data.

Usage:
    # Compare two files (original vs converted):
    uv run python tools/validate_convert.py original.eval converted.eval

    # Convert and validate in one step:
    uv run python tools/validate_convert.py original.eval --output-dir /tmp/converted
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from inspect_ai.log._convert import convert_eval_logs
from inspect_ai.log._file import read_eval_log

HEADER_FIELDS = ["status", "eval", "plan", "results", "stats", "error", "invalidated"]


def deep_diff(a: object, b: object, path: str = "") -> list[str]:
    """Return list of human-readable difference descriptions."""
    diffs: list[str] = []

    if type(a) is not type(b):
        diffs.append(f"{path}: type mismatch: {type(a).__name__} vs {type(b).__name__}")
        return diffs

    if isinstance(a, dict):
        assert isinstance(b, dict)
        all_keys = set(a) | set(b)
        for k in sorted(all_keys):
            if k not in a:
                diffs.append(f"{path}.{k}: missing in original")
            elif k not in b:
                diffs.append(f"{path}.{k}: missing in converted")
            else:
                diffs.extend(deep_diff(a[k], b[k], f"{path}.{k}"))
    elif isinstance(a, list):
        assert isinstance(b, list)
        if len(a) != len(b):
            diffs.append(f"{path}: list length {len(a)} vs {len(b)}")
        for i, (ai, bi) in enumerate(zip(a, b)):
            diffs.extend(deep_diff(ai, bi, f"{path}[{i}]"))
    elif a != b:
        a_str = repr(a)
        b_str = repr(b)
        if len(a_str) > 120:
            a_str = a_str[:120] + "..."
        if len(b_str) > 120:
            b_str = b_str[:120] + "..."
        diffs.append(f"{path}: {a_str} != {b_str}")

    return diffs


def validate(original_path: str, converted_path: str) -> bool:
    """Compare two eval files after full resolution. Returns True if identical."""
    print(f"Reading original: {original_path}")
    orig = read_eval_log(original_path, resolve_attachments="full")

    print(f"Reading converted: {converted_path}")
    conv = read_eval_log(converted_path, resolve_attachments="full")

    ok = True

    # Compare header fields
    for field in HEADER_FIELDS:
        orig_val = getattr(orig, field)
        conv_val = getattr(conv, field)
        orig_json = json.loads(
            json.dumps(
                orig_val.model_dump() if hasattr(orig_val, "model_dump") else orig_val
            )
        )
        conv_json = json.loads(
            json.dumps(
                conv_val.model_dump() if hasattr(conv_val, "model_dump") else conv_val
            )
        )
        diffs = deep_diff(orig_json, conv_json, field)
        if diffs:
            ok = False
            for d in diffs:
                print(f"  DIFF: {d}")

    # Compare samples
    if orig.samples is None and conv.samples is None:
        pass
    elif orig.samples is None or conv.samples is None:
        print(
            f"  DIFF: samples: one is None (orig={orig.samples is not None}, conv={conv.samples is not None})"
        )
        ok = False
    elif len(orig.samples) != len(conv.samples):
        print(f"  DIFF: sample count: {len(orig.samples)} vs {len(conv.samples)}")
        ok = False
    else:
        for i, (os, cs) in enumerate(zip(orig.samples, conv.samples)):
            sample_id = f"sample[{i}] (id={os.id}, epoch={os.epoch})"
            orig_json = json.loads(json.dumps(os.model_dump()))
            conv_json = json.loads(json.dumps(cs.model_dump()))
            diffs = deep_diff(orig_json, conv_json, sample_id)
            if diffs:
                ok = False
                for d in diffs[:10]:
                    print(f"  DIFF: {d}")
                if len(diffs) > 10:
                    print(f"  ... and {len(diffs) - 10} more diffs in {sample_id}")

    if ok:
        print("OK: files are identical after resolution")
    else:
        print("FAIL: differences found")

    return ok


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate eval file conversion preserves data"
    )
    parser.add_argument("original", help="Path to original .eval file")
    parser.add_argument("converted", nargs="?", help="Path to converted .eval file")
    parser.add_argument(
        "--output-dir", help="Convert original to this dir, then validate"
    )
    args = parser.parse_args()

    if args.converted and args.output_dir:
        parser.error("Provide either a converted file or --output-dir, not both")

    if args.converted:
        converted_path = args.converted
    elif args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Converting {args.original} -> {args.output_dir}")
        convert_eval_logs(args.original, "eval", str(output_dir))
        converted_path = str(output_dir / Path(args.original).name)
    else:
        parser.error("Provide either a converted file path or --output-dir")

    ok = validate(args.original, converted_path)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
