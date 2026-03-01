import torch
import graphbench
from models.simple_mlp import SimpleMLP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset_name = "electronic_circuits_5_eff"
loader = graphbench.Loader(root="./graphbench_data", dataset_names=[dataset_name])
splits = loader.load()[0]
train_ds = splits["train"]

model = SimpleMLP().to(device)
evaluator = graphbench.Evaluator("electroniccircuit")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.MSELoss()

print("Starting minimal training check on 10 samples...")

model.train()
ys = []
preds = []

for i in range(10):
    data = train_ds[i].to(device)
    optimizer.zero_grad()

    pred = model(data).view(-1)          # [1]
    target = data.y.view(-1).to(device)  # [1]

    loss = criterion(pred, target)
    loss.backward()
    optimizer.step()

    ys.append(target.detach().cpu())
    preds.append(pred.detach().cpu())

    print(f"Sample {i:02d} | MSE loss: {loss.item():.6f}")

# Evaluate once on the 10-sample set (RSE is meaningful now)
y_true = torch.cat(ys)
y_pred = torch.cat(preds)

# 2. Reshape from 1D [10] to 2D [10, 1] as required by GraphBench
# The error was: "y_true and y_pred are supposed to be 2-dim arrays"
y_true = y_true.view(-1, 1)
y_pred = y_pred.view(-1, 1)
results = evaluator.evaluate(y_true, y_pred)

# In many versions of this library, results is a dictionary
if isinstance(results, dict):
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")
else:
    print(f"RSE: {results:.4f}")

print("\nGraphBench evaluation on these 10 samples:")
print(results)

print("\n✅ MLP forward/training/evaluator pipeline verified.")