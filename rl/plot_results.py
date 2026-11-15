"""Create a small, reproducible chart from the benchmark JSON."""
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', type=Path, default=Path('rl/results/g995-final.json'))
    args = parser.parse_args()
    results = json.loads(args.json.read_text())
    labels = list(results)
    values = [100 * results[key]['summary']['completion_rate'] for key in labels]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, values, color=['#2979b8', '#969696', '#238b57'][:len(labels)])
    ax.bar_label(bars, fmt='%.1f%%')
    ax.set(ylim=(0, 105), ylabel='Completed courses (%)', title='Performance on unseen courses')
    fig.tight_layout()
    fig.savefig(args.json.with_suffix('.png'), dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
