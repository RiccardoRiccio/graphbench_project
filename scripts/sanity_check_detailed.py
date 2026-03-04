import graphbench
import torch
from collections import Counter, defaultdict
import math

dataset_name = "electronic_circuits_5_eff"
root = "./graphbench_data"

Loader = graphbench.Loader(root=root, dataset_names=[dataset_name])
splits = Loader.load()[0]

train_ds = splits["train"]
valid_ds = splits["valid"]
test_ds  = splits["test"]

print("Split sizes:")
print("  train:", len(train_ds))
print("  valid:", len(valid_ds))
print("  test :", len(test_ds))


def safe_x(g):
    x = getattr(g, "x", None)
    if x is None:
        return None
    if x.dim() == 3 and x.size(1) == 1:
        x = x.squeeze(1)
    return x


def graph_stats(ds, split_name, max_graphs=None):
    num_graphs = len(ds) if max_graphs is None else min(len(ds), max_graphs)

    num_nodes_list = []
    num_edges_list = []
    densities = []
    y_list = []

    has_edge_attr = 0
    has_self_loops = 0
    undirected_count = 0
    duplicate_edge_graphs = 0

    x_shapes = Counter()
    x_unique_values = set()
    x_row_sums = Counter()
    node_type_counts = Counter()
    extra_fields = Counter()

    for i in range(num_graphs):
        g = ds[i]

        # record available fields
        for key in g.keys():
            if key not in ["x", "edge_index", "edge_attr", "y", "batch", "ptr"]:
                extra_fields[key] += 1

        x = safe_x(g)
        edge_index = g.edge_index
        y = g.y.view(-1).float()

        n = g.num_nodes
        e = edge_index.size(1)

        num_nodes_list.append(n)
        num_edges_list.append(e)
        y_list.append(y.mean().item())

        # density for directed graph interpretation
        if n > 1:
            densities.append(e / (n * (n - 1)))
        else:
            densities.append(0.0)

        # edge_attr presence
        if getattr(g, "edge_attr", None) is not None:
            has_edge_attr += 1

        # self-loops
        src, dst = edge_index
        if (src == dst).any().item():
            has_self_loops += 1

        # duplicate edges
        edge_tuples = list(zip(src.tolist(), dst.tolist()))
        if len(edge_tuples) != len(set(edge_tuples)):
            duplicate_edge_graphs += 1

        # undirectedness check: every (u,v) has (v,u)
        edge_set = set(edge_tuples)
        is_undirected = all((v, u) in edge_set for (u, v) in edge_set)
        if is_undirected:
            undirected_count += 1

        # node feature checks
        if x is not None:
            x_shapes[tuple(x.shape)] += 1
            x_unique_values.update(torch.unique(x).cpu().tolist())

            # row sums
            row_sums = x.sum(dim=-1)
            for val in row_sums.cpu().tolist():
                x_row_sums[float(val)] += 1

            # if one-hot, extract type id
            if x.dim() == 2:
                for row in x:
                    if torch.all((row == 0) | (row == 1)) and abs(row.sum().item() - 1.0) < 1e-6:
                        node_type = int(row.argmax().item())
                        node_type_counts[node_type] += 1

    def summarize(vals, name):
        vals_t = torch.tensor(vals, dtype=torch.float)
        print(f"{name}:")
        print(f"  mean={vals_t.mean().item():.4f} std={vals_t.std(unbiased=False).item():.4f} "
              f"min={vals_t.min().item():.4f} max={vals_t.max().item():.4f}")

    print(f"\n=== {split_name} split summary (checked {num_graphs} graphs) ===")
    summarize(num_nodes_list, "num_nodes")
    summarize(num_edges_list, "num_edges")
    summarize(densities, "density")
    summarize(y_list, "target y")

    print("\nEdge structure:")
    print(f"  graphs with edge_attr present: {has_edge_attr}/{num_graphs}")
    print(f"  graphs with self-loops:        {has_self_loops}/{num_graphs}")
    print(f"  graphs fully undirected:       {undirected_count}/{num_graphs}")
    print(f"  graphs with duplicate edges:   {duplicate_edge_graphs}/{num_graphs}")

    print("\nNode features:")
    print("  x shapes (top 10):", x_shapes.most_common(10))
    print("  unique values in x:", sorted(x_unique_values))
    print("  row sums in x (top 10):", x_row_sums.most_common(10))
    print("  node type counts:", dict(sorted(node_type_counts.items())))

    print("\nExtra Data fields:")
    print(" ", dict(extra_fields))


def corrcoef(a, b):
    a = torch.tensor(a, dtype=torch.float)
    b = torch.tensor(b, dtype=torch.float)
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.pow(2).sum().sqrt() * b.pow(2).sum().sqrt()).item()
    if denom == 0:
        return float("nan")
    return (a * b).sum().item() / denom


def simple_target_correlations(ds, split_name, max_graphs=5000):
    ys = []
    ns = []
    es = []
    type_count_vectors = []

    num_graphs = min(len(ds), max_graphs)

    for i in range(num_graphs):
        g = ds[i]
        x = safe_x(g)
        y = g.y.view(-1).float().mean().item()

        ys.append(y)
        ns.append(g.num_nodes)
        es.append(g.edge_index.size(1))

        if x is not None and x.dim() == 2 and x.size(-1) == 9:
            counts = x.sum(dim=0).cpu().tolist()  # counts per one-hot dim
            type_count_vectors.append(counts)

    print(f"\n=== {split_name} simple target correlations (first {num_graphs} graphs) ===")
    print("corr(y, num_nodes):", corrcoef(ys, ns))
    print("corr(y, num_edges):", corrcoef(ys, es))

    if type_count_vectors:
        type_count_vectors = torch.tensor(type_count_vectors, dtype=torch.float)
        ys_t = torch.tensor(ys, dtype=torch.float)
        for j in range(type_count_vectors.size(1)):
            print(f"corr(y, count_type_{j}):", corrcoef(ys_t.tolist(), type_count_vectors[:, j].tolist()))


# Inspect one sample in detail
g = train_ds[0]
x = safe_x(g)

print("\n=== One detailed sample ===")
print(g)
print("keys:", g.keys())
print("x shape:", None if x is None else tuple(x.shape))
print("edge_index shape:", tuple(g.edge_index.shape))
print("edge_attr:", None if getattr(g, "edge_attr", None) is None else tuple(g.edge_attr.shape))
print("y:", g.y)

if x is not None:
    print("first 5 node rows:\n", x[:5])
    if x.dim() == 2:
        print("row sums:", x.sum(dim=-1))
        if x.size(-1) == 9:
            print("argmax node types:", x.argmax(dim=-1))

print("first 20 edges:\n", g.edge_index[:, :20])

# Run summaries
graph_stats(train_ds, "train", max_graphs=5000)
graph_stats(valid_ds, "valid", max_graphs=2000)
simple_target_correlations(train_ds, "train", max_graphs=5000)