import os
import json
import pandas as pd

ROOTS = {
    "GAT": "./results_gat",
    "GCN": "./results_gcn",
    "GIN": "./results_gin",
    "GPS": "./results_gps",
    "GT": "./results_gt",
}

datasets = [
    "electronic_circuits_5_eff",
    "electronic_circuits_5_vout",
    "electronic_circuits_7_eff",
    "electronic_circuits_7_vout",
    "electronic_circuits_10_eff",
    "electronic_circuits_10_vout",
]

pretty_columns = {
    "electronic_circuits_5_eff": "5_eff",
    "electronic_circuits_5_vout": "5_vout",
    "electronic_circuits_7_eff": "7_eff",
    "electronic_circuits_7_vout": "7_vout",
    "electronic_circuits_10_eff": "10_eff",
    "electronic_circuits_10_vout": "10_vout",
}


def latest_run_dir(path: str):
    if not os.path.isdir(path):
        return None
    subdirs = [
        d for d in os.listdir(path)
        if os.path.isdir(os.path.join(path, d))
    ]
    if not subdirs:
        return None
    subdirs = sorted(subdirs)
    return os.path.join(path, subdirs[-1])


table = {}

for model_name, root in ROOTS.items():
    table[model_name] = {}

    for dataset in datasets:
        dataset_dir = os.path.join(root, dataset)

        if not os.path.isdir(dataset_dir):
            table[model_name][dataset] = None
            continue

        run_dir = latest_run_dir(dataset_dir)
        if run_dir is None:
            table[model_name][dataset] = None
            continue

        results_json = os.path.join(run_dir, "results.json")
        if not os.path.isfile(results_json):
            table[model_name][dataset] = None
            continue

        with open(results_json, "r") as f:
            results = json.load(f)

        table[model_name][dataset] = results.get("test_rse", None)

df = pd.DataFrame.from_dict(table, orient="index")
df = df[datasets]
df_pretty = df.rename(columns=pretty_columns)

print("\n" + "=" * 100)
print("TEST RSE RESULTS TABLE")
print("=" * 100)
print(df_pretty.to_string())
print("=" * 100)

# Save CSV
csv_path = "./summary_test_rse_selected_models.csv"
df_pretty.to_csv(csv_path)
print(f"\nCSV saved to: {csv_path}")

# Generate LaTeX
latex_table = df_pretty.round(4).to_latex(
    index=True,
    caption="Test RSE results on the Electronic Circuits datasets.",
    label="tab:selected_model_results",
    column_format="l" + "c" * len(df_pretty.columns),
    escape=False,
)

print("\n" + "=" * 100)
print("LATEX TABLE")
print("=" * 100)
print(latex_table)
print("=" * 100)

tex_path = "./summary_test_rse_selected_models.tex"
with open(tex_path, "w") as f:
    f.write(latex_table)

print(f"\nLaTeX saved to: {tex_path}")