import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.loader import DataLoader

import graphbench
from models.baseline_mlp import BaselineMLP

class FilteredDataset(torch.utils.data.Dataset):
    def __init__(self, base_ds, name="ds"):
        self.base = base_ds
        self.idx = []
        bad = 0
        for i in range(len(base_ds)):
            y = getattr(base_ds[i], "y", None)
            if y is None or torch.isnan(y).any() or torch.isinf(y).any():
                bad += 1
            else:
                self.idx.append(i)
        print(f"  {name}: kept {len(self.idx)}/{len(base_ds)}, removed {bad}")

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        return self.base[self.idx[i]]

@torch.no_grad()
def eval_rse(model, loader, evaluator, device):
    model.eval()
    preds, trues = [], []

    for batch in loader:
        batch = batch.to(device)

        # model returns [B] (you squeeze(-1) in BaselineMLP)
        y_pred = model(batch).view(-1, 1)  # (N,1) for Evaluator
        y_true = batch.y.view(-1, 1).float()

        preds.append(y_pred.cpu())
        trues.append(y_true.cpu())

    y_pred = torch.cat(preds, dim=0)
    y_true = torch.cat(trues, dim=0)

    return evaluator.evaluate(y_pred=y_pred, y_true=y_true)


def main():
    # ---- Config (Table 27 style; 5 epochs for sanity) ----
    dataset_name = "electronic_circuits_10_acc"   # you said you want this
    root = "./graphbench_data"

    epochs = 20
    batch_size = 512
    lr = 1e-3
    hidden_dim = 384

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # ---- Load splits (confirmed format from your sanity_check) ----
    Loader = graphbench.Loader(root=root, dataset_names=[dataset_name])
    out = Loader.load()
    splits = out[0]  # dict with keys train/valid/test

    # train_ds = splits["train"]
    # valid_ds = splits["valid"]
    # test_ds  = splits["test"]

    train_ds = FilteredDataset(splits["train"], "train")
    valid_ds = FilteredDataset(splits["valid"], "valid")
    test_ds  = FilteredDataset(splits["test"], "test")

    print("Split sizes:")
    print("  train:", len(train_ds))
    print("  valid:", len(valid_ds))
    print("  test: ", len(test_ds))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)
    

    # ---- Infer input_dim from x (x is [N,1,9] so squeeze) ----
    sample = train_ds[0]
    x = sample.x.squeeze(1) if sample.x.dim() == 3 else sample.x
    input_dim = x.size(-1)

    # ---- Model ----
    model = BaselineMLP(input_dim=input_dim, hidden_dim=hidden_dim, out_channels=1).to(device)
    opt = Adam(model.parameters(), lr=lr)

    # ---- Evaluator (master.csv maps "electroniccircuit" -> RSE) ----
    evaluator = graphbench.Evaluator("electroniccircuit")

    # ---- Train ----
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_graphs = 0

        for batch in train_loader:
            batch = batch.to(device)

            y_pred = model(batch).float()          # [B]
            y_true = batch.y.view(-1).float()      # [B]

            loss = F.mse_loss(y_pred, y_true)      # simple training objective
            opt.zero_grad()
            loss.backward()
            opt.step()

            bs = batch.num_graphs
            total_loss += loss.item() * bs
            total_graphs += bs

        train_mse = total_loss / max(total_graphs, 1)
        valid_rse = eval_rse(model, valid_loader, evaluator, device)

        print(f"Epoch {epoch:02d} | train MSE: {train_mse:.6f} | valid RSE: {valid_rse:.6f}")

    # ---- Final test ----
    test_rse = eval_rse(model, test_loader, evaluator, device)
    print(f"Test RSE: {test_rse:.6f}")


if __name__ == "__main__":
    main()