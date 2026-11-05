"""Write TensorBoard scalars, including backfilling existing JSON run records."""
import argparse
import json
from pathlib import Path


def write_run(path):
    from torch.utils.tensorboard import SummaryWriter
    path = Path(path)
    updates = 0
    with SummaryWriter(str(path / 'tensorboard')) as writer:
        for directory in sorted(path.glob('step-*')):
            step = int(directory.name.split('-')[-1])
            baseline = directory / 'baseline.json'
            if baseline.exists():
                for suite, value in json.loads(baseline.read_text())['scores'].items():
                    writer.add_scalar(f'baseline/{suite}', value, step)
            candidates = directory / 'candidates.json'
            if candidates.exists():
                records = json.loads(candidates.read_text())
                rewards = [r['reward'] for r in records]
                if rewards:
                    mean = sum(rewards) / len(rewards)
                    writer.add_scalar('reward/mean', mean, step)
                    writer.add_scalar('reward/max', max(rewards), step)
                    writer.add_scalar('reward/std', (sum((r-mean)**2 for r in rewards) / len(rewards))**.5, step)
                    writer.add_scalar('edit/invalid_rate', sum(bool(r.get('error')) for r in records) / len(records), step)
                valid = [r for r in records if 'edited' in r]
                if valid:
                    writer.add_scalar('edit/changed_elements', sum(r['changed'] for r in valid) / len(valid), step)
                    writer.add_scalar('edit/preservation_pass_rate', sum(r['passed'] for r in valid) / len(valid), step)
                    for suite in valid[0]['edited']:
                        writer.add_scalar(f'edited/{suite}', sum(r['edited'][suite] for r in valid) / len(valid), step)
                        writer.add_scalar(f'delta/{suite}', sum(r['edited'][suite]-r['base'][suite] for r in valid) / len(valid), step)
            update_file = directory / 'update.json'
            if update_file.exists():
                update = json.loads(update_file.read_text())
                updates += int(update['updated'])
                writer.add_scalar('train/loss', update['loss'], step)
                writer.add_scalar('train/updated', int(update['updated']), step)
                writer.add_scalar('train/update_count', updates, step)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', default='runs')
    args = parser.parse_args()
    for run in Path(args.runs).iterdir():
        if run.is_dir():
            write_run(run)
