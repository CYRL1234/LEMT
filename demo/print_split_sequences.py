import argparse
import pickle

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Print sequence indices for a split from an MMFi pkl."
    )
    parser.add_argument(
        "--pkl_path",
        required=True,
        help="Path to the dataset pkl file.",
    )
    parser.add_argument(
        "--split",
        required=True,
        help="Split name (e.g., test_rdn_p3).",
    )
    args = parser.parse_args()

    with open(args.pkl_path, "rb") as f:
        data = pickle.load(f)

    splits = data.get("splits", {})
    if args.split not in splits:
        available = ", ".join(sorted(splits.keys()))
        print(f"Split not found: {args.split}")
        print(f"Available splits: {available}")
        return 1

    indices = splits[args.split]
    print(indices)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
