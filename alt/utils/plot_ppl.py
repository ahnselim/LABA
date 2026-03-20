"""
Plot perplexity or overlap-rate curves from a ``ppl_curve.csv`` file.

Usage:
    python plot_ppl.py /path/to/ppl_curve.csv --mode ppl
    python plot_ppl.py /path/to/ppl_curve.csv --mode overlap --save-dir /path/to/save_dir
    python utils/plot_ppl.py /ssd/ssd4/asl/LABA/alt/output/llama_3_8/output_step2_mc_greedy/2_25bit/step2_2/ppl_curve.csv --mode ppl overlap
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


MODE_CONFIG = {
    "ppl": {
        "column": "global_best_ppl",
        "ylabel": "Perplexity",
        "title": "Perplexity over generation",
        "color": "tab:blue",
        "filename": "ppl.png",
    },
    "overlap": {
        "column": "overlap10_rate",
        "ylabel": "Overlap rate",
        "title": "Overlap rate over generation",
        "color": "tab:orange",
        "filename": "overlap.png",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot perplexity or overlap rate over generation from a ppl_curve.csv file."
    )
    parser.add_argument("csv_path", type=Path, help="Path to ppl_curve.csv")
    parser.add_argument(
        "--mode",
        nargs="+",
        choices=sorted(MODE_CONFIG.keys()),
        required=True,
        help="One or more plot modes to draw: ppl overlap",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="Directory to save plotted figures. Defaults to <csv_parent>/plots.",
    )
    return parser.parse_args()


def plot_metric(df: pd.DataFrame, mode: str, save_dir: Path):
    config = MODE_CONFIG[mode]
    plt.figure(figsize=(8, 5))
    plt.plot(df["generation"], df[config["column"]], marker="o", color=config["color"])
    plt.xlabel("Generation")
    plt.ylabel(config["ylabel"])
    plt.title(config["title"])
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    save_path = save_dir / config["filename"]
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved {mode} plot to {save_path}")



def main():
    args = parse_args()
    df = pd.read_csv(args.csv_path)

    required_columns = {"generation"} | {MODE_CONFIG[mode]["column"] for mode in args.mode}
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns in CSV: {missing_columns}")

    save_dir = args.save_dir if args.save_dir is not None else args.csv_path.resolve().parent / "plots"
    save_dir.mkdir(parents=True, exist_ok=True)

    for mode in dict.fromkeys(args.mode):
        plot_metric(df, mode, save_dir)


if __name__ == "__main__":
    main()
