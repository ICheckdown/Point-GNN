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
parser.add_argument('--output_dir',    type=str, default='triton_weights')
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)
config = load_config(os.path.join(args.checkpoint_path, 'config'))

# Build model to get variable names in graph
if config['input_features'] in ('irgb', '0000', 'i000', '0rgb'):
    t_ivf = tf.placeholder(dtype=tf.float32, shape=[None, 4])
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

model.predict(t_ivf, t_vcl, t_kil, t_el, t_istr)

# All scopes to extract
SCOPES = []

# Layer 1
for i in range(4):
    SCOPES.append(
        f'layer1/extract_vertex_features/fully_connected'
        f'{"" if i==0 else f"_{i}"}')
for i in range(2):
    SCOPES.append(
        f'layer1/combined_features/fully_connected'
        f'{"" if i==0 else f"_{i}"}')

# Layers 2-4
for layer in ['layer2', 'layer3', 'layer4']:
    # auto-offset
    for i in range(2):
        SCOPES.append(
            f'{layer}/fully_connected'
            f'{"" if i==0 else f"_{i}"}')
    # edge MLP
    for i in range(2):
        SCOPES.append(
            f'{layer}/extract_vertex_features/fully_connected'
            f'{"" if i==0 else f"_{i}"}')
    # combined MLP
    for i in range(2):
        SCOPES.append(
            f'{layer}/combined_features/fully_connected'
            f'{"" if i==0 else f"_{i}"}')

# Output head
SCOPES += [
    'output/predictor/cls/fully_connected',
    'output/predictor/cls/fully_connected_1',
]
for cls_i in range(4):
    for fc_i in range(3):
        SCOPES.append(
            f'output/predictor/loc/cls_{cls_i}/fully_connected'
            f'{"" if fc_i==0 else f"_{fc_i}"}')

g     = tf.get_default_graph()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.variables_initializer(tf.global_variables()))
    saver.restore(sess, tf.train.latest_checkpoint(args.checkpoint_path))

    print(f"Extracting {len(SCOPES)} FC layers...\n")
    saved = 0
    for scope in SCOPES:
        for w_suffix, f_suffix in [('weights', 'weights'),
                                    ('biases',  'biases')]:
            for tf_name in [f'{scope}/{w_suffix}:0',
                            f'{scope}/kernel:0',
                            f'{scope}/bias:0']:
                try:
                    val   = sess.run(g.get_tensor_by_name(tf_name))
                    fname = scope.replace('/', '_') + f'_{f_suffix}.npy'
                    np.save(os.path.join(args.output_dir, fname), val)
                    print(f"  {scope}/{w_suffix}  {val.shape}")
                    saved += 1
                    break
                except KeyError:
                    continue

print(f"\nSaved {saved} arrays to {args.output_dir}/")