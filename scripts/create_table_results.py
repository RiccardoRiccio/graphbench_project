import json
import csv
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]

MODELS = ["GT", "GIN", "GCN", "GAT", "GTresidual"]
DATASETS = [
    "electronic_circuits_5_eff",
    "electronic_circuits_5_vout",
    "electronic_circuits_7_eff",
    "electronic_circuits_7_vout",
    "electronic_circuits_10_eff",
    "electronic_circuits_10_vout",
]

def get_best_test_rse(model, dataset):
    base = ROOT / f"results_{model.lower()}" / dataset
    if not base.exists():
        return None

    best = None

    for run_dir in base.iterdir():
        if not run_dir.is_dir():
            continue

        results_file = run_dir / "results.json"
        if not results_file.exists():
            continue

        try:
            with open(results_file, "r") as f:
                data = json.load(f)
            test_rse = data.get("test_rse")
            if test_rse is None:
                continue
            if best is None or test_rse < best:
                best = test_rse
        except Exception:
            continue

    return best

def build_table():
    rows = []
    for model in MODELS:
        row = [model]
        for dataset in DATASETS:
            val = get_best_test_rse(model, dataset)
            row.append(f"{val:.6f}" if val is not None else "NA")
        rows.append(row)
    return rows

def save_csv(header, rows, out_file):
    with open(out_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

def make_markdown_table(header, rows):
    lines = []
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)

def main():
    header = ["Model"] + DATASETS
    rows = build_table()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = ROOT / "scripts" / f"best_test_rse_table_{timestamp}.csv"
    md_path = ROOT / "scripts" / f"best_test_rse_table_{timestamp}.md"

    save_csv(header, rows, csv_path)

    md_table = make_markdown_table(header, rows)
    with open(md_path, "w") as f:
        f.write(md_table + "\n")

    print(md_table)
    print(f"\nSaved CSV to: {csv_path}")
    print(f"Saved Markdown to: {md_path}")

if __name__ == "__main__":
    main()