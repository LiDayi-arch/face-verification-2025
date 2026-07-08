import argparse
import csv
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    with args.scores.open(newline="") as f:
        rows = list(csv.DictReader(f))

    with args.out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Index", "ID"])
        for row in rows:
            writer.writerow([row["Index"], 1 if float(row["score"]) >= args.threshold else 0])

    print(f"Wrote {args.out} from {args.scores} with threshold={args.threshold:.4f}")


if __name__ == "__main__":
    main()
