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

    # Distinctive muted earth-tone palette with hatching for extra differentiation
    COLORS   = ['#6d6875', '#b5838d', '#e5989b']
    LABELS   = ['NB + TF-IDF (Baseline)', 'LR + TF-IDF bigrams', 'LinearSVM + TF-IDF bigrams']
    CAPSIZE  = 4

    for metric in METRICS:
        nb_means  = np.array([float(r[f'NB {metric}'].split(' ')[0])                   for r in summary_rows])
        lr_means  = np.array([float(r[f'LR {metric}'].replace('*','').split(' ')[0])   for r in summary_rows])
        svm_means = np.array([float(r[f'SVM {metric}'].replace('*','').split(' ')[0])  for r in summary_rows])

        nb_stds   = np.array([float(r[f'NB {metric}'].split('± ')[1])                  for r in summary_rows])
        lr_stds   = np.array([float(r[f'LR {metric}'].replace('*','').split('± ')[1])  for r in summary_rows])
        svm_stds  = np.array([float(r[f'SVM {metric}'].replace('*','').split('± ')[1]) for r in summary_rows])

        # Clip error bars so they never extend below zero
        nb_err  = [np.minimum(nb_stds,  nb_means),  nb_stds]
        lr_err  = [np.minimum(lr_stds,  lr_means),  lr_stds]
        svm_err = [np.minimum(svm_stds, svm_means), svm_stds]

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')

        err_kw = dict(elinewidth=1.2, ecolor='#333333', capsize=CAPSIZE, capthick=1.2)

        bars1 = ax.bar(x - width, nb_means,  width, yerr=nb_err,  label=LABELS[0],
                       color=COLORS[0], edgecolor='white', linewidth=0.6, error_kw=err_kw)
        bars2 = ax.bar(x,         lr_means,  width, yerr=lr_err,  label=LABELS[1],
                       color=COLORS[1], edgecolor='white', linewidth=0.6, error_kw=err_kw)
        bars3 = ax.bar(x + width, svm_means, width, yerr=svm_err, label=LABELS[2],
                       color=COLORS[2], edgecolor='white', linewidth=0.6, error_kw=err_kw)

        # Value labels: place above error bar top, annotate near-zero NB bars
        all_bars  = [(bars1, nb_means,  nb_stds),
                     (bars2, lr_means,  lr_stds),
                     (bars3, svm_means, svm_stds)]
        for bars, means, stds in all_bars:
            for bar, mean, std in zip(bars, means, stds):
                top = mean + std
                if mean > 0.01:
                    ax.text(bar.get_x() + bar.get_width() / 2, top + 0.02,
                            f'{mean:.2f}', ha='center', va='bottom', fontsize=7, color='#222222')
                else:
                    ax.text(bar.get_x() + bar.get_width() / 2, 0.025,
                            'NB≈0', ha='center', va='bottom', fontsize=6,
                            color='#555555', style='italic')

        max_val = max(np.max(lr_means + lr_stds), np.max(svm_means + svm_stds))
        ax.set_ylim(0, min(max_val + 0.18, 1.0))
        ax.set_ylabel(metric, fontsize=11, labelpad=8)
        ax.set_title(f'{metric}: NB vs LR vs LinearSVM (mean ± std, 30 runs; error bars clipped at 0)',
                     fontsize=10, fontweight='bold', pad=12)
        ax.set_xticks(x)
        ax.set_xticklabels(projects, fontsize=10)
        ax.yaxis.set_tick_params(labelsize=9)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['left', 'bottom']:
            ax.spines[spine].set_color('#999999')
        ax.grid(axis='y', linestyle='--', alpha=0.4, color='#bbbbbb')
        ax.legend(frameon=True, framealpha=0.95, fontsize=9, loc='upper right',
                  edgecolor='#cccccc')
        plt.tight_layout()
        plt.savefig(f'results/plot_{metric.lower()}.png', dpi=150)
        plt.close()
        print(f"Saved results/plot_{metric.lower()}.png")


if __name__ == '__main__':
    run_all()
