import torch, time

E, F, V = 696000, 300, 2094
edge_features = torch.randn(E, F, device='cuda')
dst_indices = torch.randint(0, V, (E,), device='cuda')

# Your current implementation
def scatter_max_manual(edge_features, dst_indices, num_vertices):
    out = torch.full((num_vertices, edge_features.shape[1]), -1e9,
                      dtype=edge_features.dtype, device=edge_features.device)
    idx = dst_indices.unsqueeze(1).expand(-1, edge_features.shape[1])
    out.scatter_reduce_(0, idx, edge_features, reduce='amax')
    return torch.where(out < -1e8, torch.zeros_like(out), out)

for _ in range(5):  # warmup
    scatter_max_manual(edge_features, dst_indices, V)
torch.cuda.synchronize()

t0 = time.perf_counter()
for _ in range(20):
    scatter_max_manual(edge_features, dst_indices, V)
torch.cuda.synchronize()
print(f"scatter_reduce_ manual: {(time.perf_counter()-t0)/20*1000:.2f}ms")

