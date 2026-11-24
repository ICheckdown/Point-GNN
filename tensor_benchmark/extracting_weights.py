"""
extract_weights.py
==================
Extracts weight matrices and biases for the 4 heavy MatMul ops
from the Point-GNN TF checkpoint and saves them as .npy files.

Run once:
  python3 extract_weights.py checkpoints/car_auto_T3_train/
"""

import os
import argparse
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from util.tf_compat import tf
from util.config_util import load_config
from models.models import get_model
from models.box_encoding import get_encoding_len

parser = argparse.ArgumentParser()
parser.add_argument('checkpoint_path', type=str)
parser.add_argument('--output_dir', type=str, default='triton_weights')
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

CONFIG_PATH = os.path.join(args.checkpoint_path, 'config')
config      = load_config(CONFIG_PATH)

# Build model to get variable names
if config['input_features'] in ('irgb', '0000', 'i000', '0rgb'):
    t_ivf = tf.placeholder(dtype=tf.float32, shape=[None, 4])
elif config['input_features'] == 'rgb':
    t_ivf = tf.placeholder(dtype=tf.float32, shape=[None, 3])
else:
    t_ivf = tf.placeholder(dtype=tf.float32, shape=[None, 1])

num_levels = len(config['runtime_graph_gen_kwargs']['level_configs'])
t_vcl  = [tf.placeholder(dtype=tf.float32, shape=[None, 3])
           for _ in range(num_levels + 1)]
t_el   = [tf.placeholder(dtype=tf.int32,   shape=[None, 2])
           for _ in range(num_levels)]
t_kil  = [tf.placeholder(dtype=tf.int32,   shape=[None, 1])
           for _ in range(num_levels)]
t_istr = tf.placeholder(dtype=tf.bool, shape=[])

model = get_model(config['model_name'])(
    num_classes=config['num_classes'],
    box_encoding_len=get_encoding_len(config['box_encoding_method']),
    mode='test',
    **config['model_kwargs'])

t_logits, t_pred_box = model.predict(
    t_ivf, t_vcl, t_kil, t_el, t_istr)

# Target ops — weight variable name pattern in TF
# Each FC layer has: weights:0 and biases:0
TARGET_SCOPES = [
    'layer1/extract_vertex_features/fully_connected_2',  # fc2: K=64,  N=128
    'layer1/extract_vertex_features/fully_connected_3',  # fc3: K=128, N=300
    'layer2/extract_vertex_features/fully_connected',    # L2fc0: K=303, N=300
    'layer2/extract_vertex_features/fully_connected_1',  # L2fc1: K=300, N=300
    # layers 3 and 4 share same weight names as layer2 in Point-GNN
    # (they are separate variables but same structure)
    'layer3/extract_vertex_features/fully_connected',
    'layer3/extract_vertex_features/fully_connected_1',
    'layer4/extract_vertex_features/fully_connected',
    'layer4/extract_vertex_features/fully_connected_1',
]

tf_graph = tf.get_default_graph()
saver    = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.variables_initializer(tf.global_variables()))
    saver.restore(sess, tf.train.latest_checkpoint(args.checkpoint_path))

    print("Extracting weights...\n")
    for scope in TARGET_SCOPES:
        # Try common TF variable name patterns
        for suffix in ['weights:0', 'kernel:0']:
            try:
                w_tensor = tf_graph.get_tensor_by_name(f'{scope}/{suffix}')
                w_val    = sess.run(w_tensor)
                fname    = scope.replace('/', '_') + '_weights.npy'
                np.save(os.path.join(args.output_dir, fname), w_val)
                print(f"  {scope}/weights  shape={w_val.shape}")
                break
            except KeyError:
                continue

        for suffix in ['biases:0', 'bias:0']:
            try:
                b_tensor = tf_graph.get_tensor_by_name(f'{scope}/{suffix}')
                b_val    = sess.run(b_tensor)
                fname    = scope.replace('/', '_') + '_biases.npy'
                np.save(os.path.join(args.output_dir, fname), b_val)
                print(f"  {scope}/biases   shape={b_val.shape}")
                break
            except KeyError:
                continue

print(f"\nWeights saved to: {args.output_dir}/")