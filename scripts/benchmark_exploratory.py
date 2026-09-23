"""Run validation benchmarks sequentially with an explicit exploratory override."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from pic2word.data import load_cirr_split, load_fashioniq_split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cirr-root', type=Path, required=True)
    parser.add_argument('--fashioniq-root', type=Path, required=True)
    parser.add_argument('--b1-run', type=Path, required=True)
    parser.add_argument('--b4-run', type=Path, required=True)
    parser.add_argument('--b5-run', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--override-go', action='store_true', required=True)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    for name, dataset in [('CIRR', load_cirr_split(args.cirr_root)),
                          ('Fashion-IQ', load_fashioniq_split(args.fashioniq_root))]:
        missing = dataset.missing_images()
        if missing or not dataset.queries:
            raise ValueError(f'{name}: {len(missing)} missing images; {len(dataset.queries)} queries')
        print(f'{name}: {len(dataset.queries)} queries, all image files present', flush=True)
    runs = dict(B1=args.b1_run, B4=args.b4_run, B5=args.b5_run)
    checkpoints = {}
    for variant, run in runs.items():
        checkpoint = (run / 'best_checkpoint.pt').resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        with checkpoint.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        checkpoints[variant] = dict(path=str(checkpoint), sha256=digest)
    if args.preflight_only:
        print('Preflight passed. No evaluation started.')
        return 0
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    protocol = dict(status='running', exploratory=True, manual_go_override=True,
                    go_gate_passed=False, split='val', checkpoints=checkpoints,
                    cirr_root=str(args.cirr_root.resolve()),
                    fashioniq_root=str(args.fashioniq_root.resolve()),
                    note='User authorized evaluation despite unmet GO gate. No training or checkpoint changes.')
    protocol_path = output / 'benchmark_protocol.json'
    def save():
        protocol_path.write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    save()
    summary = {}
    try:
        for variant, checkpoint in checkpoints.items():
            summary[variant] = {}
            for name, script, root, metric, index_args in [
                ('CIRR', 'evaluate_cfpe_cirr.py', args.cirr_root, 'metrics.json',
                 ['--index', str(output / variant / 'indexes' / 'cirr.pt')]),
                ('Fashion-IQ', 'evaluate_cfpe_fashioniq.py', args.fashioniq_root, 'fashioniq_metrics.json',
                 ['--index-dir', str(output / variant / 'indexes')]),
            ]:
                destination = output / variant / name
                destination.mkdir(parents=True)
                command = [sys.executable, '-u', str(project / 'scripts' / script),
                           '--dataset-root', str(root.resolve()), '--checkpoint', checkpoint['path'],
                           '--run-dir', str(destination), '--device', 'cuda', '--top-k', '100', *index_args]
                print(f'Starting {variant} / {name}', flush=True)
                with (destination / 'evaluation.log').open('w', encoding='utf-8') as log:
                    process = subprocess.Popen(command, cwd=project, stdout=subprocess.PIPE,
                                               stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
                    for line in process.stdout:
                        log.write(line)
                        log.flush()
                        print(line, end='', flush=True)
                    if process.wait():
                        raise RuntimeError(f'{variant} / {name} failed; see {destination / "evaluation.log"}')
                summary[variant][name] = json.loads((destination / metric).read_text(encoding='utf-8'))
                (output / 'comparison.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        protocol['status'] = 'completed'
    except BaseException as error:
        protocol.update(status='failed', error=str(error))
        raise
    finally:
        save()
    print(f'Completed: {output}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
