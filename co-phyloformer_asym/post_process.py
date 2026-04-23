import argparse
from pathlib import Path

from plot_from_csv import load_csv, plot_scatter, plot_density

EVENT_NAMES = ["Speciation", "HGT", "Loss", "Duplication"]


def main():
    parser = argparse.ArgumentParser("Plot predictions from saved CSVs (no inference)")
    parser.add_argument("--output-dir", required=True,
                        help="Directory to write plots into (and default location for CSVs)")
    parser.add_argument("--train-csv", default=None,
                        help="Path to train_predictions.csv (default: <output-dir>/train_predictions.csv)")
    parser.add_argument("--val-csv", default=None,
                        help="Path to val_predictions.csv (default: <output-dir>/val_predictions.csv)")
    parser.add_argument("--bins", type=int, default=40)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    train_csv  = Path(args.train_csv) if args.train_csv else output_dir / "train_predictions.csv"
    val_csv    = Path(args.val_csv)   if args.val_csv   else output_dir / "val_predictions.csv"

    if not train_csv.exists():
        raise FileNotFoundError(f"Train CSV not found: {train_csv}")
    if not val_csv.exists():
        raise FileNotFoundError(f"Val CSV not found: {val_csv}")

    print(f"[PostProcess] Loading train predictions: {train_csv}")
    train_preds, train_labels = load_csv(train_csv)
    print(f"[PostProcess] Loading val predictions:   {val_csv}")
    val_preds, val_labels = load_csv(val_csv)

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_scatter(train_preds, train_labels, val_preds, val_labels, output_dir)
    plot_density(train_preds, train_labels, val_preds, val_labels, output_dir, bins=args.bins)

    print(f"[PostProcess] Scatter plots: {output_dir / 'scatter_plots'}")
    print(f"[PostProcess] Density plots: {output_dir / 'density_plots'}")


if __name__ == "__main__":
    main()
