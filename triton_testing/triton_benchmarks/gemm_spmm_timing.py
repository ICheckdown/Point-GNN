"""Profile current Point-GNN: TF32 GEMM, Triton/torch gathers, CSR Triton scatter.
Run alongside pointgnn_torch.py. Does not modify the model.
"""
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import time, argparse, functools, csv
import numpy as np
import torch
torch.set_float32_matmul_precision('high')
from tqdm import tqdm
from dataset.kitti_dataset import KittiDataset
from models.graph_gen import get_graph_generate_fn
from util.config_util import load_config
import pointgnn_torch as pg

parser = argparse.ArgumentParser()
parser.add_argument('checkpoint_path')
parser.add_argument('--dataset_root_dir', default='../kitti/')
parser.add_argument('--dataset_split_file', default='')
parser.add_argument('--weights_dir', default='triton_weights')
parser.add_argument('--n_frames', type=int, default=100)
parser.add_argument('--warmup', type=int, default=3)
parser.add_argument('--skip_frames', default='13')
parser.add_argument('--output_dir', default='triton_testing/stage1_profile_csr')
args = parser.parse_args()
if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
config = load_config(os.path.join(args.checkpoint_path, 'config'))
split = args.dataset_split_file or os.path.join(args.dataset_root_dir, '3DOP_splits/val.txt')
dataset = KittiDataset(os.path.join(args.dataset_root_dir, 'image/training/image_2'),
                       os.path.join(args.dataset_root_dir, 'velodyne/training/velodyne/'),
                       os.path.join(args.dataset_root_dir, 'calib/training/calib/'),
                       os.path.join(args.dataset_root_dir, 'labels/training/label_2'),
                       split, num_classes=config['num_classes'])
graph_generate_fn = get_graph_generate_fn(config['graph_gen_method'])

def get_input_features(cam, t):
    a = cam.attr
    if t == 'irgb': return a
    if t == '0rgb': return np.hstack((np.zeros((len(a), 1), dtype=a.dtype), a[:, 1:]))
    if t == '0000': return np.zeros_like(a)
    if t == 'i000': return np.hstack((a[:, [0]], np.zeros((len(a), 3), dtype=a.dtype)))
    if t == 'i': return a[:, [0]]
    if t == 'rgb': return a[:, 1:4]
    return np.zeros((len(a), 1), dtype=np.float32)

def build_inputs(i):
    cam = dataset.get_cam_points_in_image_with_rgb(i, config['downsample_by_voxel_size'])
    verts, _, edges = graph_generate_fn(cam.xyz, **config['runtime_graph_gen_kwargs'])
    return verts[0], get_input_features(cam, config['input_features']), verts[1], edges[0], edges[1]

class EventTimer:
    def __init__(self, max_pairs=256):
        self.starts = [torch.cuda.Event(enable_timing=True) for _ in range(max_pairs)]
        self.ends = [torch.cuda.Event(enable_timing=True) for _ in range(max_pairs)]
        self.max_pairs = max_pairs
        self.reset()

    def reset(self):
        self.used = 0
        self.records = []

    def run(self, category, fn, *a, **kw):
        if self.used >= self.max_pairs: raise RuntimeError('Increase EventTimer max_pairs')
        j = self.used
        self.used += 1
        start, end = self.starts[j], self.ends[j]
        start.record()
        result = fn(*a, **kw)
        end.record()
        self.records.append((category, j))
        return result

    def totals(self):
        names = ('GEMM', 'Gather', 'Scatter')
        totals = {name: 0.0 for name in names}
        calls = {name: 0 for name in names}
        for name, j in self.records:
            totals[name] += self.starts[j].elapsed_time(self.ends[j])
            calls[name] += 1
        return totals, calls

class Instrument:
    def __init__(self, timer):
        self.timer = timer
        self.saved = {}
        self.original_index_select = None

    def __enter__(self):
        for name, category in (('fc_relu', 'GEMM'), ('fc_no_relu', 'GEMM'),
                               ('scatter_max', 'Scatter'), ('triton_gather', 'Gather')):
            original = getattr(pg, name)
            self.saved[name] = original
            @functools.wraps(original)
            def wrapped(*a, __fn=original, __cat=category, **kw):
                return self.timer.run(__cat, __fn, *a, **kw)
            setattr(pg, name, wrapped)
        self.original_index_select = torch.Tensor.index_select
        original = self.original_index_select
        def wrapped_index_select(tensor, dim, index):
            return self.timer.run('Gather', original, tensor, dim, index)
        torch.Tensor.index_select = wrapped_index_select
        return self

    def __exit__(self, *_):
        for name, original in self.saved.items(): setattr(pg, name, original)
        if self.original_index_select is not None:
            torch.Tensor.index_select = self.original_index_select

skip = {int(x) for x in args.skip_frames.split(',') if x.strip()}
frames = [i for i in range(dataset.num_files) if i not in skip][:args.n_frames]
if not frames: raise RuntimeError('No frames selected')
model = pg.PointGNN(weights_dir=args.weights_dir, num_classes=dataset.num_classes, device='cuda')
print(f'GPU: {torch.cuda.get_device_name(0)} | frames: {len(frames)}')
print('Warming up...')
warm = build_inputs(frames[0])
for _ in range(args.warmup):
    result = model.forward(*warm)
    del result
torch.cuda.synchronize()
del warm
torch.cuda.reset_peak_memory_stats()
print('Warmup complete.\n')

timer = EventTimer()
frame_start = torch.cuda.Event(enable_timing=True)
frame_end = torch.cuda.Event(enable_timing=True)
rows = []
with Instrument(timer):
    for i in tqdm(frames, desc='Profiling CSR model'):
        inputs = build_inputs(i)
        timer.reset()
        torch.cuda.synchronize()
        wall0 = time.perf_counter()
        frame_start.record()
        logits, boxes = model.forward(*inputs)
        frame_end.record()
        torch.cuda.synchronize()
        wall_ms = (time.perf_counter() - wall0) * 1000.0
        cuda_ms = frame_start.elapsed_time(frame_end)
        totals, calls = timer.totals()
        gemm, gather, scatter = totals['GEMM'], totals['Gather'], totals['Scatter']
        spmm = gather + scatter
        other = cuda_ms - gemm - spmm
        rows.append(dict(frame=i, gemm_ms=gemm, gather_ms=gather, scatter_ms=scatter,
                         spmm_ms=spmm, other_ms=other, cuda_forward_ms=cuda_ms,
                         wall_forward_ms=wall_ms, gemm_calls=calls['GEMM'],
                         gather_calls=calls['Gather'], scatter_calls=calls['Scatter']))
        del logits, boxes, inputs

def avg(name): return float(np.mean([row[name] for row in rows]))
cuda_ms, wall_ms = avg('cuda_forward_ms'), avg('wall_forward_ms')
print('\n' + '=' * 73)
print(f'CURRENT PROFILE — TF32 GEMM + TRITON GATHER + CSR SCATTER | {len(rows)} frames')
print('=' * 73)
print(f"{'Category':<19}{'Mean ms':>14}{'Share':>12}{'Calls/frame':>15}")
for label, key, count in (('GEMM', 'gemm_ms', 'gemm_calls'),
                          ('Gather', 'gather_ms', 'gather_calls'),
                          ('CSR scatter-max', 'scatter_ms', 'scatter_calls'),
                          ('SpMM-like total', 'spmm_ms', None), ('Other', 'other_ms', None)):
    value = avg(key)
    c = f'{avg(count):.1f}' if count else '-'
    print(f'{label:<19}{value:>12.3f} ms{100 * value / cuda_ms:>11.2f}%{c:>15}')
print('-' * 73)
print(f"{'CUDA forward':<19}{cuda_ms:>12.3f} ms{100:>11.2f}%")
print(f"{'Wall forward':<19}{wall_ms:>12.3f} ms")
print(f"GEMM : SpMM-like ratio = {avg('gemm_ms') / avg('spmm_ms'):.3f}:1")
print('SpMM-like = all index_select + Triton feature gather + CSR scatter-max')
print('Other is the residual, including CPU launch gaps / copies; timings are approximate.')
print(f"CUDA memory: allocated={torch.cuda.memory_allocated()/1024**3:.2f} GB | "
      f"reserved={torch.cuda.memory_reserved()/1024**3:.2f} GB | "
      f"peak_alloc={torch.cuda.max_memory_allocated()/1024**3:.2f} GB | "
      f"peak_reserved={torch.cuda.max_memory_reserved()/1024**3:.2f} GB")
os.makedirs(args.output_dir, exist_ok=True)
csv_path = os.path.join(args.output_dir, 'stage1_csr_profile.csv')
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
print(f'Saved: {csv_path}')
