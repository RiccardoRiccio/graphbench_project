import graphbench
import torch
import os

# All tasks from the website screenshot
all_tasks = [
    "electronic_circuits_5_eff", "electronic_circuits_5_vout",
    "electronic_circuits_7_eff", "electronic_circuits_7_vout",
    "electronic_circuits_10_eff", "electronic_circuits_10_vout"
]

root_path = "./graphbench_data"

print(f"{'Dataset Name':<30} | {'Train':<8} | {'Valid':<8} | {'Test':<8} | {'Nodes':<6}")
print("-" * 75)

for task in all_tasks:
    try:
        loader = graphbench.Loader(root=root_path, dataset_names=[task])
        dataset = loader.load()[0]
        
        train_len = len(dataset["train"])
        val_len = len(dataset["valid"])
        test_len = len(dataset["test"])
        
        # Get node count from a sample
        sample = dataset["train"][0]
        nodes = sample.num_nodes
        
        print(f"{task:<30} | {train_len:<8} | {val_len:<8} | {test_len:<8} | {nodes:<6}")
    except Exception as e:
        print(f"Error loading {task}: {e}")