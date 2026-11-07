"""Test explicite de la CLI complète sur 20 images synthétiques temporaires."""
import csv
import os
from pathlib import Path
import subprocess
import sys
import tempfile

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_NUM_INTRAOP_THREADS'] = '2'
os.environ['TF_NUM_INTEROP_THREADS'] = '2'
import tensorflow as tf

repo = Path(__file__).resolve().parents[1]
python = sys.executable
with tempfile.TemporaryDirectory(prefix='clip-smoke-') as directory:
    root = Path(directory)
    rows = []
    for label, intensity in [('dog', 40), ('ball', 210)]:
        for i in range(10):
            path = root / f'{label}_{i}.jpg'
            tf.io.write_file(str(path), tf.io.encode_jpeg(tf.fill((12, 12, 3), tf.constant(intensity + i, tf.uint8))))
            rows.append({'image_path':path.name, 'label':label, 'caption':f'a {label} example number {i}'})
    csv_path = root / 'captions.csv'
    with csv_path.open('w') as f:
        writer = csv.DictWriter(f, fieldnames=['image_path', 'label', 'caption'])
        writer.writeheader()
        writer.writerows(rows)
    run = root / 'run'
    commands = [
        ['train', '--csv', str(csv_path), '--image-root', str(root), '--output', str(run), '--classifier-epochs', '1', '--clip-epochs', '1', '--batch-size', '2', '--sequence-length', '8'],
        ['index', '--csv', str(csv_path), '--image-root', str(root), '--run', str(run), '--batch-size', '2'],
        ['search', '--run', str(run), '--text', 'a dog', '--top-k', '2'],
        ['search', '--run', str(run), '--image', str(root / 'dog_0.jpg'), '--top-k', '2'],
        ['evaluate', '--run', str(run), '--batch-size', '2'],
    ]
    for command in commands:
        print('RUN', command[0], flush=True)
        result = subprocess.run([python, str(repo / 'main.py'), *command], cwd=repo, env=os.environ, capture_output=True, text=True)
        if result.returncode:
            print(result.stdout[-4000:], result.stderr[-6000:], flush=True)
            raise SystemExit(result.returncode)
        print('OK', command[0], result.stdout[-700:], flush=True)
    # Index invalidation must fail after configuration changes.
    import json
    config = json.loads((run / 'config.json').read_text())
    config['seed'] += 1
    (run / 'config.json').write_text(json.dumps(config))
    result = subprocess.run([python, str(repo / 'main.py'), 'search', '--run', str(run), '--text', 'dog'], cwd=repo, env=os.environ, capture_output=True, text=True)
    assert result.returncode == 1 and "index ne correspond plus" in result.stderr, result.stderr
    print('OK stale index rejected', flush=True)
