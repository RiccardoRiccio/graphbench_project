import os
import json
import pandas as pd

ROOT = "./results_seed_42_all_graphbench_res_and_gatoriginal"

models = [
    # "GAToriginal",
    # "GATGraphBench",
    # "GCNGraphBench",
    # "GINGraphBench",
    # "GTGraphBench",
    # "PNAGraphBench",
    # "GPSGraphBench",
    "MLPGraphBench",
]

datasets = [
    "electronic_circuits_5_eff",
    "electronic_circuits_5_vout",
    "electronic_circuits_7_eff",
    "electronic_circuits_7_vout",
    "electronic_circuits_10_eff",
    "electronic_circuits_10_vout",
]

# folder naming used by your training script
model_to_folder = {
    # "GAToriginal":    "results_42_gatoriginal",
    # "GATGraphBench":  "results_42_gatgraphbench",
    # "GCNGraphBench":  "results_42_gcngraphbench",
    # "GINGraphBench":  "results_42_gingraphbench",
    # "GTGraphBench":   "results_42_gtgraphbench",
    # "PNAGraphBench":  "results_42_pnagraphbench",
    # "GPSGraphBench":  "results_42_gpsgraphbench",
    "MLPGraphBench":  "results_42_mlpgraphbench",
    
}

table = {}

for model in models:
    table[model] = {}
    model_folder = os.path.join(ROOT, model_to_folder[model])

    for dataset in datasets:
        dataset_folder = os.path.join(model_folder, dataset)

        if not os.path.isdir(dataset_folder):
            table[model][dataset] = None
            continue

        run_dirs = [
            d for d in os.listdir(dataset_folder)
            if os.path.isdir(os.path.join(dataset_folder, d))
        ]

        if len(run_dirs) == 0:
            table[model][dataset] = None
            continue

        # pick latest timestamped run
        run_dirs = sorted(run_dirs)
        latest_run = run_dirs[-1]

        results_json = os.path.join(dataset_folder, latest_run, "results.json")

        if not os.path.isfile(results_json):
            table[model][dataset] = None
            continue

        with open(results_json, "r") as f:
            results = json.load(f)

        table[model][dataset] = results.get("test_rse", None)

# build dataframe
df = pd.DataFrame.from_dict(table, orient="index")
df = df[datasets]

# optional shorter column names for display / latex
pretty_columns = {
    "electronic_circuits_5_eff": "5_eff",
    "electronic_circuits_5_vout": "5_vout",
    "electronic_circuits_7_eff": "7_eff",
    "electronic_circuits_7_vout": "7_vout",
    "electronic_circuits_10_eff": "10_eff",
    "electronic_circuits_10_vout": "10_vout",
}

df_pretty = df.rename(columns=pretty_columns)

# -------------------------
# PRINT TABLE
# -------------------------
print("\n" + "="*100)
print("TEST RSE RESULTS TABLE")
print("="*100)
print(df_pretty.to_string())
print("="*100)

# -------------------------
# SAVE CSV
# -------------------------
csv_path = os.path.join(ROOT, "summary_test_rse_graphbench_and_gatoriginal.csv")
df_pretty.to_csv(csv_path)
print(f"\nCSV saved to: {csv_path}")

# -------------------------
# GENERATE LATEX
# -------------------------
latex_table = df_pretty.round(4).to_latex(
    index=True,
    caption="Test RSE results for GraphBench-style baselines and GAToriginal on the Electronic Circuits datasets.",
    label="tab:graphbench_results",
    column_format="l" + "c"*len(df_pretty.columns),
    escape=False
)

print("\n" + "="*100)
print("LATEX TABLE")
print("="*100)
print(latex_table)
print("="*100)

latex_path = os.path.join(ROOT, "summary_test_rse_graphbench_and_gatoriginal.tex")
with open(latex_path, "w") as f:
    f.write(latex_table)

print(f"\nLaTeX saved to: {latex_path}")