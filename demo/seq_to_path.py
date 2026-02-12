import argparse
import os
from glob import glob

def _list_sequences(root_dir: str) -> list[str]:
    pattern = os.path.join(root_dir, "E*/S*/A*")
    return sorted(glob(pattern))

def _format_entry(index: int, path: str) -> str:
    parts = path.rstrip(os.sep).split(os.sep)
    action = parts[-1]
    subject = parts[-2]
    env = parts[-3]
    return f"{index}: {env}/{subject}/{action}"

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Map MMFi sequence index to dataset path."
    )
    parser.add_argument(
        "--indices",
        nargs="+",
        type=int,
        required=True,
        help="Sequence indices (0-based, sorted by path).",
    )
    parser.add_argument(
        "--root_dir",
        default="/home/ryan/MM-Fi/MMFi_Dataset",
        help="Dataset root directory.",
    )
    parser.add_argument(
        "--count",
        action="store_true",
        help="Print total sequences before mapping.",
    )
    args = parser.parse_args()

    sequences = _list_sequences(args.root_dir)
    if not sequences:
        print(f"No sequences found under: {args.root_dir}")
        return 1

    if args.count:
        print(f"total_sequences={len(sequences)}")

    for idx in args.indices:
        if idx < 0 or idx >= len(sequences):
            print(f"index={idx} out of range [0, {len(sequences) - 1}]")
            continue
        print(_format_entry(idx, sequences[idx]))

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
