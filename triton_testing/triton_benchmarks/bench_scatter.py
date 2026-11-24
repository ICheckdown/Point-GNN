"""
check_scatter_structure.py

Loads Point-GNN graph inputs and checks whether destination indices are sorted/grouped,
plus the number of incoming edges per destination vertex.
"""

import os,argparse
import numpy as np

from dataset.kitti_dataset import KittiDataset
from models.graph_gen import get_graph_generate_fn
from util.config_util import load_config

parser=argparse.ArgumentParser()
parser.add_argument("checkpoint_path",type=str)
parser.add_argument("--dataset_root_dir",type=str,default="../kitti/")
parser.add_argument("--dataset_split_file",type=str,default="")
parser.add_argument("--n_frames",type=int,default=10)
parser.add_argument("--skip_frames",type=str,default="13")
args=parser.parse_args()

config_path=os.path.join(args.checkpoint_path,"config")
if not os.path.isfile(config_path): raise FileNotFoundError(config_path)
config=load_config(config_path)

split=args.dataset_split_file or os.path.join(args.dataset_root_dir,"3DOP_splits/val.txt")
dataset=KittiDataset(os.path.join(args.dataset_root_dir,"image/training/image_2"),os.path.join(args.dataset_root_dir,"velodyne/training/velodyne/"),os.path.join(args.dataset_root_dir,"calib/training/calib/"),os.path.join(args.dataset_root_dir,"labels/training/label_2"),split,num_classes=config["num_classes"])
graph_generate_fn=get_graph_generate_fn(config["graph_gen_method"])

skip={int(x) for x in args.skip_frames.split(",") if x.strip()}
frames=[i for i in range(dataset.num_files) if i not in skip][:args.n_frames]

print("\n"+"="*92)
print("POINT-GNN SCATTER STRUCTURE CHECK")
print("="*92)
print(f"{'Frame':>6} {'Layer':>6} {'E':>10} {'V':>7} {'Sorted':>8} {'Mean':>9} {'Median':>9} {'P90':>8} {'P99':>8} {'Max':>7}")
print("-"*92)

all_sorted=True
for i in frames:
    cam=dataset.get_cam_points_in_image_with_rgb(i,config["downsample_by_voxel_size"])
    vertex_coord_list,_,edges_list=graph_generate_fn(cam.xyz,**config["runtime_graph_gen_kwargs"])

    for layer,edges in enumerate(edges_list,1):
        dst=np.asarray(edges[:,1],dtype=np.int64)
        E=len(dst)
        if E==0:
            print(f"{i:>6} {layer:>6} {0:>10} {0:>7} {'N/A':>8}")
            continue

        V=int(dst.max())+1
        sorted_dst=bool(np.all(dst[:-1] <= dst[1:]))
        all_sorted &= sorted_dst

        counts=np.bincount(dst,minlength=V)
        nz=counts[counts>0]

        print(f"{i:>6} {layer:>6} {E:>10,d} {V:>7,d} {str(sorted_dst):>8} {nz.mean():>9.2f} {np.median(nz):>9.1f} {np.percentile(nz,90):>8.1f} {np.percentile(nz,99):>8.1f} {nz.max():>7,d}")

print("-"*92)
print(f"All destination arrays sorted/grouped: {all_sorted}")
print("="*92)
