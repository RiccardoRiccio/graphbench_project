import graphbench
import torch
dataset_name = "electronic_circuits_5_eff"

Loader = graphbench.Loader(root="./graphbench_data", dataset_names=[dataset_name])
out = Loader.load()

splits = out[0]  # because out is a list of length 1
train_ds = splits["train"]
val_ds   = splits["valid"]   # <-- key is "valid"
test_ds  = splits["test"]

print("Split sizes:")
print("  train:", len(train_ds))
print("  valid:", len(val_ds))
print("  test: ", len(test_ds))

sample = train_ds[0]
print("\nSample:", sample)

print("x shape:", sample.x.shape if getattr(sample, "x", None) is not None else None)
print("edge_index shape:", sample.edge_index.shape if getattr(sample, "edge_index", None) is not None else None)
print("edge_attr:", None if getattr(sample, "edge_attr", None) is None else sample.edge_attr.shape)
print("y:", sample.y)
print("num_nodes:", sample.num_nodes)
print("num_edges:", sample.edge_index.size(1))

for i in range(20):
    g = train_ds[i]
    print(i, g.x.shape, g.y.item(), g.num_nodes, g.edge_index.size(1))


g = train_ds[0]

# squeeze to remove the middle dimension
x = g.x.squeeze(1)   # [num_nodes, 9]

print("x shape after squeeze:", x.shape)
print("First node feature vector:", x[0])
print("All node feature vectors:\n", x)
print("Unique values in x:", torch.unique(x))

import torch
import graphbench
import numpy as np

# 1. Setup the domain
# Note: "electroniccircuit" is the standard domain key for these tasks
dataset_name = "electronic_circuits_5_eff"
evaluator = graphbench.Evaluator("electroniccircuit")

print(f"--- Sanity Check: GraphBench Evaluator ---")

# 2. Simulate model outputs (y_pred) and targets (y_true)
# Let's simulate a batch of 10 samples
batch_size = 10
y_true = torch.randn(batch_size, 1)  # Real targets are usually [B, 1]
y_pred = y_true + torch.randn(batch_size, 1) * 0.1 # Simulated predictions with noise

print(f"y_true shape: {y_true.shape}")
print(f"y_pred shape: {y_pred.shape}")

# 3. Test the Evaluate function
# The evaluator expects (y_true, y_pred)
# Based on the library, these often need to be torch Tensors or numpy arrays
try:
    results = evaluator.evaluate(y_true, y_pred)
    print("\nEvaluator Results Structure:")
    print(results)
    
    # If it returns a dictionary, check the keys
    if isinstance(results, dict):
        for metric, value in results.items():
            print(f"Metric: {metric:<10} | Value: {value:.4f}")
except Exception as e:
    print(f"\nEvaluation failed! Error: {e}")

# 4. Check Loader internal structure again for multi-tasking
loader = graphbench.Loader(root="./graphbench_data", dataset_names=[dataset_name])
dataset_list = loader.load()
print(f"\nLoader returned list of length: {len(dataset_list)}")