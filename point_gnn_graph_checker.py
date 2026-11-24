#!/usr/bin/env python3
"""Fresh Point-GNN 0.4m vs 0.8m KITTI validation and evaluator negative control.

Run from repository root alongside downsampling_experiment.py.
Defaults to the COMPLETE KITTI val split. Never reuses previous predictions.
Requires the existing PyTorch/Triton runner on Jetson and a built native KITTI evaluator.
"""
import argparse
import csv
import datetime as dt
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_prediction(path):
    count = 0
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        cols = line.split()
        if len(cols) != 16:
            raise RuntimeError(f'{path}:{line_number}: expected 16 KITTI detection fields, found {len(cols)}')
        try:
            for v in cols[1:]:
                val = float(v)
                if not (float('-inf') < val < float('inf')):
                    raise ValueError('nonfinite')
        except ValueError as exc:
            raise RuntimeError(f'{path}:{line_number}: invalid numeric field') from exc
        count += 1
    return count


def corrupt_one(source, dest, shift=100000.0):
    """Move camera boxes and 2D rectangles far away, preserving confidence."""
    count = 0
    with source.open() as inp, dest.open('w') as out:
        for raw in inp:
            fields = raw.split()
            if not fields:
                out.write('\n')
                continue
            if len(fields) != 16:
                raise RuntimeError(f'Unexpected KITTI row in {source}')
            for ix in (4, 5, 6, 7, 11, 12, 13):
                fields[ix] = format(float(fields[ix]) + shift, '.8f')
            out.write(' '.join(fields) + '\n')
            count += 1
    return count


def parse_ap(text):
    parsed = {}
    for line in text.splitlines():
        m = re.match(r'^\s*(car_detection_(?:AP|BEV_AP|3D_AP))\s*:\s*(.*)$', line, re.I)
        if m:
            values = m.group(2).split()
            if len(values) < 3:
                continue
            try:
                parsed[m.group(1).lower()] = [float(s) for s in values[:3]]
            except ValueError:
                pass
    return parsed


def execute(command, logpath):
    print('\n$', ' '.join(map(str, command)), flush=True)
    with logpath.open('w') as log:
        proc = subprocess.Popen(list(map(str, command)), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            log.write(line)
            log.flush()
            if ('processing' in line or 'Finished' in line or 'RESULTS:' in line or
                    'Frames per condition:' in line or 'Voxel size' in line or 'WARNING' in line):
                print(line.rstrip(), flush=True)
        returncode = proc.wait()
    if returncode:
        raise RuntimeError(f'Command exited {returncode}. See {logpath}')


def evaluate(executable, gt_dir, prediction_dir, logpath):
    command = [str(executable), str(gt_dir), str(prediction_dir)]
    print('Evaluating:', ' '.join(command), flush=True)
    proc = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    logpath.write_text(proc.stdout)
    if proc.returncode:
        raise RuntimeError(f'Evaluator returned {proc.returncode}; see {logpath}')
    ap = parse_ap(proc.stdout)
    if len(ap) != 3:
        raise RuntimeError(f'Evaluator returned no complete Car AP metrics; see {logpath}')
    return ap


def flat(ap):
    result = {}
    for k, vals in ap.items():
        for difficulty, value in zip(('easy', 'moderate', 'hard'), vals):
            result[f'{k}_{difficulty}'] = value
    return result


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint_path')
    parser.add_argument('--dataset_root_dir', default='../kitti/')
    parser.add_argument('--dataset_split_file', default='')
    parser.add_argument('--weights_dir', default='triton_testing/torch_weights')
    parser.add_argument('--evaluator', default='kitti_native_evaluation/evaluate_object_offline')
    parser.add_argument('--runner', default='downsampling_experiment.py')
    parser.add_argument('--output-dir', default='results/final_graph_validation')
    parser.add_argument('--n-frames', type=int, default=None,
                        help='Diagnostic subset only. OMIT for full validation split and publishable AP.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--max-negative-ap', type=float, default=0.001,
                        help='Maximum allowed AP percentage for shifted-box control (default 0.001).')
    args = parser.parse_args()
    if args.n_frames is not None and args.n_frames <= 0:
        parser.error('--n-frames must be positive')
    runner = Path(args.runner).resolve()
    evaluator = Path(args.evaluator).resolve()
    checkpoint = Path(args.checkpoint_path).resolve()
    root = Path(args.dataset_root_dir).resolve()
    weights = Path(args.weights_dir).resolve()
    split = (Path(args.dataset_split_file).resolve() if args.dataset_split_file else
             root / '3DOP_splits/val.txt')
    for path in (runner, evaluator, checkpoint / 'config', split):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not weights.is_dir():
        raise FileNotFoundError(weights)

    out = Path(args.output_dir).resolve()
    if out.exists():
        raise FileExistsError(f'{out} already exists. Select a NEW --output-dir to rule out reuse.')
    out.mkdir(parents=True)
    status_file = out / 'status.json'
    def status(phase, **kwargs):
        status_file.write_text(json.dumps({'phase': phase, 'date': dt.datetime.now().isoformat(),
                                          **kwargs}, indent=2))

    print('Output:', out, flush=True)
    print('Mode:', 'FULL VALIDATION' if args.n_frames is None else
          f'{args.n_frames}-FRAME DIAGNOSTIC SUBSET (not official KITTI AP)', flush=True)
    status('running_inference')
    subprocess_args = [sys.executable, runner, checkpoint, '--dataset_root_dir', root,
                       '--dataset_split_file', split, '--weights_dir', weights,
                       '--voxel-sizes', '0.8', '--seed', str(args.seed),
                       '--warmup', str(args.warmup), '--output_dir', out / 'inference']
    if args.n_frames is None:
        subprocess_args.append('--all-frames')
    else:
        subprocess_args += ['--n_frames', str(args.n_frames)]
    execute(subprocess_args, out / 'inference.log')

    experiment = out / 'inference' / 'experiment1_voxel_detections'
    metadata = json.loads((experiment / 'metadata.json').read_text())
    if args.n_frames is None and not metadata.get('full_validation_split'):
        raise RuntimeError('Runner did not process the full validation split')
    indices = metadata['selected_frame_indices']
    if args.n_frames is not None and len(indices) != args.n_frames:
        raise RuntimeError('Incorrect frame count in runner metadata')
    # Use dataset's own filenames, NOT assumptions about numeric split indices.
    sys.path.insert(0, str(runner.parent))
    from dataset.kitti_dataset import KittiDataset
    from util.config_util import load_config
    cfg = load_config(str(checkpoint / 'config'))
    dataset = KittiDataset(str(root / 'image/training/image_2'),
                           str(root / 'velodyne/training/velodyne'),
                           str(root / 'calib/training/calib'),
                           str(root / 'labels/training/label_2'),
                           str(split), num_classes=cfg['num_classes'])
    expected = [dataset.get_filename(i) for i in indices]
    if len(expected) != len(set(expected)):
        raise RuntimeError('Duplicate frame names in dataset selection')
    expected_set = set(expected)
    condition_dirs = {s: experiment / f'voxel_{s}' / 'data' for s in ('0p4m', '0p8m')}
    rows = []
    counts = {}
    contents = {}
    status('verifying_predictions')
    gt_full = root / 'labels/training/label_2'
    for condition, data_dir in condition_dirs.items():
        actual_files = list(data_dir.glob('*.txt'))
        actual_set = {p.stem for p in actual_files}
        missing = sorted(expected_set - actual_set)
        extra = sorted(actual_set - expected_set)
        if missing or extra or len(actual_files) != len(expected):
            raise RuntimeError(f'{condition}: missing={missing[:10]} extra={extra[:10]} '
                               f'files={len(actual_files)} expected={len(expected)}')
        counts[condition] = {}
        contents[condition] = {}
        for frame in expected:
            gt = gt_full / f'{frame}.txt'
            if not gt.is_file():
                raise FileNotFoundError(f'Missing ground truth for selected frame: {gt}')
            pred = data_dir / f'{frame}.txt'
            num_boxes = check_prediction(pred)
            counts[condition][frame] = num_boxes
            contents[condition][frame] = sha(pred)
        print(f'{condition}: verified {len(expected)} fresh files, '
              f'{sum(counts[condition].values())} detections', flush=True)

    identical_nonempty = 0
    identical_empty = 0
    for frame in expected:
        same = contents['0p4m'][frame] == contents['0p8m'][frame]
        if same and counts['0p4m'][frame] + counts['0p8m'][frame] > 0:
            identical_nonempty += 1
        if same and counts['0p4m'][frame] == counts['0p8m'][frame] == 0:
            identical_empty += 1
        rows.append({'frame_id': frame, 'baseline_boxes': counts['0p4m'][frame],
                     'voxel_0p8m_boxes': counts['0p8m'][frame],
                     'identical_output': same, 'baseline_sha256': contents['0p4m'][frame],
                     'voxel_0p8m_sha256': contents['0p8m'][frame]})
    write_csv(out / 'frame_comparison.csv', rows)
    if sum(counts['0p4m'].values()) == 0 or sum(counts['0p8m'].values()) == 0:
        raise RuntimeError('Clean predictions are empty; AP experiment cannot validate accuracy')
    print(f'Identical nonempty outputs: {identical_nonempty}/{len(expected)}; '
          f'identical empty outputs: {identical_empty}/{len(expected)}', flush=True)

    # Isolate matching GT for partial subset, so unrelated labels cannot contaminate AP.
    if args.n_frames is None:
        gt_dir = gt_full
    else:
        gt_dir = out / 'ground_truth_subset' / 'label_2'
        gt_dir.mkdir(parents=True)
        for frame in expected:
            (gt_dir / f'{frame}.txt').symlink_to((gt_full / f'{frame}.txt').resolve())

    corrupt_dir = out / 'SHIFTED_BOXES_NEGATIVE_CONTROL' / 'data'
    corrupt_dir.mkdir(parents=True)
    altered = 0
    for frame in expected:
        altered += corrupt_one(condition_dirs['0p4m'] / f'{frame}.txt',
                               corrupt_dir / f'{frame}.txt')
    if altered != sum(counts['0p4m'].values()):
        raise RuntimeError('Corruption did not cover all detections')
    for frame in expected:
        if check_prediction(corrupt_dir / f'{frame}.txt') != counts['0p4m'][frame]:
            raise RuntimeError(f'Corrupted detection count differs for {frame}')
    print(f'Created shifted-box negative control with {altered} moved boxes', flush=True)

    status('evaluating')
    metrics = {}
    names = {'baseline_0p4m': condition_dirs['0p4m'].parent,
             'modified_0p8m': condition_dirs['0p8m'].parent,
             'shifted_negative_control': corrupt_dir.parent}
    for name, pred_root in names.items():
        metrics[name] = evaluate(evaluator, gt_dir, pred_root, out / f'evaluator_{name}.log')
        print(f"{name}: 3D AP (E/M/H) = {metrics[name]['car_detection_3d_ap']}", flush=True)

    errors = []
    for metric, vals in metrics['shifted_negative_control'].items():
        if any(abs(v) > args.max_negative_ap for v in vals):
            errors.append(f'Negative control produced nonzero {metric}: {vals}')
    if not any(v > args.max_negative_ap for vals in metrics['baseline_0p4m'].values() for v in vals):
        errors.append('Baseline AP is zero; evaluator or dataset invalid')
    if identical_nonempty == len(expected) and identical_empty == 0:
        errors.append('Every prediction file identical despite graph resolution change')

    results = [{'condition': name, 'scope': 'full_validation' if args.n_frames is None else
                f'diagnostic_subset_{len(expected)}', 'frames': len(expected),
                **flat(metrics[name])} for name in names]
    write_csv(out / 'ap_comparison.csv', results)
    report = {
        'scope': 'FULL_VALIDATION' if args.n_frames is None else 'DIAGNOSTIC_SUBSET_NOT_OFFICIAL',
        'frame_count': len(expected), 'runner': str(runner), 'evaluator': str(evaluator),
        'split_file_sha256': sha(split), 'weights_dir': str(weights), 'checkpoint': str(checkpoint),
        'total_detections': {x: sum(c.values()) for x, c in counts.items()},
        'identical_nonempty_files': identical_nonempty, 'identical_empty_files': identical_empty,
        'altered_detections': altered, 'metrics': metrics, 'negative_control_passed': not errors,
        'errors': errors,
        'note': ('Full split AP uses native evaluator. Any partial-subset metric is diagnostic '
                 'and should not be compared with published KITTI benchmark values.')}
    (out / 'report.json').write_text(json.dumps(report, indent=2))
    if errors:
        status('FAILED', errors=errors)
        raise RuntimeError('Validation FAILED: ' + '; '.join(errors))
    status('PASSED', ap_comparison=str(out / 'ap_comparison.csv'))
    print('\nPASS: clean AP evaluated and deliberately shifted boxes gave ~0 AP.', flush=True)
    print('Results:', out / 'ap_comparison.csv', flush=True)
    print('Frame checks:', out / 'frame_comparison.csv', flush=True)
    print('Full report:', out / 'report.json', flush=True)


if __name__ == '__main__':
    main()
