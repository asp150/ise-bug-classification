"""
Experiment Runner
- Compares three models: NB+TF-IDF (baseline), LR+TF-IDF bigrams, LinearSVM+TF-IDF bigrams
- 30 repeats, 70/30 stratified train/test split
- Metrics: Precision, Recall, F1
- Statistical tests: Wilcoxon signed-rank test (significance) + A12 effect size (magnitude)
- Saves results to results/
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from baseline import load_data, run_baseline
from improved import run_improved
from logistic import run_logistic

PROJECTS = {
    'TensorFlow': 'data/tensorflow.csv',
    'PyTorch':    'data/pytorch.csv',
    'Keras':      'data/keras.csv',
    'MXNet':      'data/incubator-mxnet.csv',
    'Caffe':      'data/caffe.csv',
}

N_REPEATS = 30
METRICS = ['Precision', 'Recall', 'F1']
os.makedirs('results', exist_ok=True)


def wilcoxon_test(a, b):
    """Returns p-value from Wilcoxon signed-rank test. Returns 1.0 if identical."""
    if np.allclose(a, b):
        return 1.0
    _, p = wilcoxon(a, b)
    return p


def a12(a, b):
    """
    Vargha-Delaney A12 effect size: probability that a value from 'a' exceeds one from 'b'.
    A12 = 0.5 means no difference; >0.5 means 'a' tends to be larger.
    Thresholds: negligible <0.56, small 0.56-0.64, medium 0.64-0.71, large >0.71
    """
    m, n = len(a), len(b)
    more = sum(1.0 for x in a for y in b if x > y)
    tie  = sum(1.0 for x in a for y in b if x == y)
    return (more + 0.5 * tie) / (m * n)


def effect_label(a12_val):
    diff = abs(a12_val - 0.5)
    if diff < 0.06:
        return 'negligible'
    elif diff < 0.14:
        return 'small'
    elif diff < 0.21:
        return 'medium'
    else:
        return 'large'


def fmt(arr, sig=''):
    return f"{arr.mean():.3f} ± {arr.std():.3f}{sig}"


def run_all():
    summary_rows = []

    for project, filepath in PROJECTS.items():
        print(f"\n--- {project} ---")
        X, y = load_data(filepath)

        b_prec, b_rec, b_f1   = run_baseline(X, y, n_repeats=N_REPEATS)
        lr_prec, lr_rec, lr_f1 = run_logistic(X, y, n_repeats=N_REPEATS)
        s_prec, s_rec, s_f1   = run_improved(X, y, n_repeats=N_REPEATS)

        baseline_scores = [b_prec,  b_rec,  b_f1]
        lr_scores       = [lr_prec, lr_rec, lr_f1]
        svm_scores      = [s_prec,  s_rec,  s_f1]

        row = {'Project': project}
        for m, b, lr, svm in zip(METRICS, baseline_scores, lr_scores, svm_scores):
            p_lr  = wilcoxon_test(b, lr)
            p_svm = wilcoxon_test(b, svm)
            a_lr  = a12(lr,  b)
            a_svm = a12(svm, b)

            sig_lr  = '*' if p_lr  < 0.05 else ''
            sig_svm = '*' if p_svm < 0.05 else ''

            row[f'NB {m}']       = fmt(b)
            row[f'LR {m}']       = fmt(lr,  sig_lr)
            row[f'SVM {m}']      = fmt(svm, sig_svm)
            row[f'LR p({m})']    = f"{p_lr:.4f}"
            row[f'SVM p({m})']   = f"{p_svm:.4f}"
            row[f'LR A12({m})']  = f"{a_lr:.3f} ({effect_label(a_lr)})"
            row[f'SVM A12({m})'] = f"{a_svm:.3f} ({effect_label(a_svm)})"

            print(f"  {m}: NB={b.mean():.3f} | LR={lr.mean():.3f} (p={p_lr:.4f}, A12={a_lr:.3f}) | "
                  f"SVM={svm.mean():.3f} (p={p_svm:.4f}, A12={a_svm:.3f})")

        summary_rows.append(row)

        raw = pd.DataFrame({
            'nb_precision': b_prec,   'nb_recall': b_rec,   'nb_f1': b_f1,
            'lr_precision': lr_prec,  'lr_recall': lr_rec,  'lr_f1': lr_f1,
            'svm_precision': s_prec,  'svm_recall': s_rec,  'svm_f1': s_f1,
        })
        raw.to_csv(f'results/raw_{project.lower()}.csv', index=False)

    df = pd.DataFrame(summary_rows)
    df.to_csv('results/summary_table.csv', index=False)
    print("\nSaved results/summary_table.csv")

    plot_results(summary_rows)


def plot_results(summary_rows):
    projects = [r['Project'] for r in summary_rows]
    x = np.arange(len(projects))
    width = 0.26

    for metric in METRICS:
        nb_means  = [float(r[f'NB {metric}'].split(' ')[0])                          for r in summary_rows]
        lr_means  = [float(r[f'LR {metric}'].replace('*', '').split(' ')[0])         for r in summary_rows]
        svm_means = [float(r[f'SVM {metric}'].replace('*', '').split(' ')[0])        for r in summary_rows]

        fig, ax = plt.subplots(figsize=(10, 4.5))
        fig.patch.set_facecolor('#f9f9f9')
        ax.set_facecolor('#f9f9f9')

        bars1 = ax.bar(x - width, nb_means,  width, label='NB + TF-IDF (Baseline)',      color='#4e79a7', edgecolor='white', linewidth=0.8)
        bars2 = ax.bar(x,         lr_means,  width, label='LR + TF-IDF bigrams',          color='#f28e2b', edgecolor='white', linewidth=0.8)
        bars3 = ax.bar(x + width, svm_means, width, label='LinearSVM + TF-IDF bigrams',   color='#e15759', edgecolor='white', linewidth=0.8)

        for bars in (bars1, bars2, bars3):
            for bar in bars:
                h = bar.get_height()
                if h > 0.01:
                    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.012,
                            f'{h:.2f}', ha='center', va='bottom', fontsize=7, color='#333333')

        ax.set_ylabel(metric, fontsize=11, labelpad=8)
        ax.set_title(f'{metric} Comparison: NB vs LR vs LinearSVM Across Projects',
                     fontsize=12, fontweight='bold', pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(projects, fontsize=10)
        ax.set_ylim(0, 1.15)
        ax.yaxis.set_tick_params(labelsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#cccccc')
        ax.spines['bottom'].set_color('#cccccc')
        ax.grid(axis='y', linestyle=':', alpha=0.6, color='#aaaaaa')
        ax.legend(frameon=True, framealpha=0.9, fontsize=9, loc='upper right')
        plt.tight_layout()
        plt.savefig(f'results/plot_{metric.lower()}.png', dpi=150)
        plt.close()
        print(f"Saved results/plot_{metric.lower()}.png")


if __name__ == '__main__':
    run_all()
