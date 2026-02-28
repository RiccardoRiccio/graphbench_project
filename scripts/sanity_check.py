import os
import graphbench
import torch

root_dir = os.path.expanduser("~/graphbench_project/data")
# Keep the specific task for the loader
task_name = "electronic_circuits_5_eff"

print(f"Loading task: {task_name}")
loader = graphbench.Loader(root_dir, task_name)
dataset_list = loader.load()

# 1) Unpack the data
# Based on your output, it's a list containing a dict
task_data = dataset_list[0] 
print("\n--- Data Structure ---")
print(f"Splits available: {list(task_data.keys())}")

# 2) Inspect a sample from the training set
train_set = task_data['train']
sample = train_set[0]
print(f"Sample type: {type(sample)}")
print(f"Nodes (x): {sample.x.shape}")
print(f"Edges (edge_index): {sample.edge_index.shape}")
if hasattr(sample, 'edge_attr') and sample.edge_attr is not None:
    print(f"Edge features: {sample.edge_attr.shape}")
print(f"Target (y): {sample.y}")

# 3) Fix the Evaluator
# 3) Fixed Evaluator Probe
print("\n--- Evaluator Probe ---")
try:
    evaluator = graphbench.Evaluator("electronic_circuits")
except KeyError:
    # This prints the first 10 valid names in the library's CSV
    evaluator_instance = graphbench.Evaluator("socialnetwork") # 'socialnetwork' is a known default from repo
    valid_names = evaluator_instance.csv_info.index.tolist()
    print("Available names for Evaluator:", valid_names[:10])
    
    # Try the most likely alternative name
    alt_name = "electronic_circuits" if "electronic_circuits" in valid_names else "electronic_circuit"
    evaluator = graphbench.Evaluator(alt_name)

print(f"Evaluator successfully linked to: {evaluator.name}")