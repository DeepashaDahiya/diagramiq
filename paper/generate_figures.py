"""
DiagramIQ — Publication Figure Generation Script
Generates high-resolution (300 DPI) publication-quality figures for the LaTeX research paper:
  1. Fig 1: Benchmark Comparison Bar Chart (EMA, GMS, MGF, Hallucination Rate)
  2. Fig 2: Ablation Study Component Contribution Plot
  3. Fig 3: Pedagogical User Study (PEQ) 5-Point Likert Distribution
"""

import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Ensure high quality typography
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.titlesize': 14,
    'figure.dpi': 300
})

OUTPUT_DIR = Path(__file__).resolve().parent / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def generate_benchmark_comparison():
    print("[*] Generating Figure 1: Benchmark Comparison Chart...")
    models = ['Vanilla LLaVA 1.6', 'GPT-4o (Zero-Shot)', 'DiagramIQ (Ours)']
    ema = [48.2, 69.4, 89.6]
    mgf = [31.0, 58.4, 94.2]
    hallucination = [34.5, 18.2, 4.1]

    x = np.arange(len(models))
    width = 0.26

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    
    rects1 = ax.bar(x - width, ema, width, label='Exact Match Accuracy (EMA %)', color='#3b82f6', edgecolor='black', linewidth=0.8)
    rects2 = ax.bar(x, mgf, width, label='Grounding Fidelity (MGF %)', color='#10b981', edgecolor='black', linewidth=0.8)
    rects3 = ax.bar(x + width, hallucination, width, label='Hallucination Rate (%) [Lower is Better]', color='#ef4444', edgecolor='black', linewidth=0.8)

    ax.set_ylabel('Score (%)')
    ax.set_title('Performance Comparison on DiagramIQ-Eval Benchmark (100 Diagrams)')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.95)
    ax.set_ylim(0, 105)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    # Attach labels above bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    plt.tight_layout()
    png_path = OUTPUT_DIR / "fig1_benchmark_comparison.png"
    pdf_path = OUTPUT_DIR / "fig1_benchmark_comparison.pdf"
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)
    plt.close()
    print(f"    [+] Saved to {png_path.name} and {pdf_path.name}")


def generate_ablation_chart():
    print("[*] Generating Figure 2: Ablation Study Chart...")
    variants = [
        'Full Pipeline',
        'w/o Stage 4\n(No Traversal)',
        'w/o Stage 2\n(No Graph)',
        'w/o MGF Gating\n(No Check)'
    ]
    ema_scores = [89.6, 64.1, 51.0, 78.3]
    hrr_scores = [90.0, 52.4, 28.6, 68.1]

    x = np.arange(len(variants))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    rects1 = ax.bar(x - width/2, ema_scores, width, label='Exact Match Accuracy (EMA %)', color='#6366f1', edgecolor='black', linewidth=0.8)
    rects2 = ax.bar(x + width/2, hrr_scores, width, label='Hallucination Reduction Rate (HRR %)', color='#f59e0b', edgecolor='black', linewidth=0.8)

    ax.set_ylabel('Percentage (%)')
    ax.set_title('Ablation Study: Impact of Neuro-Symbolic Components')
    ax.set_xticks(x)
    ax.set_xticklabels(variants, fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.95)
    ax.set_ylim(0, 105)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    for rects in [rects1, rects2]:
        for rect in rects:
            h = rect.get_height()
            ax.annotate(f'{h:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    png_path = OUTPUT_DIR / "fig2_ablation_study.png"
    pdf_path = OUTPUT_DIR / "fig2_ablation_study.pdf"
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)
    plt.close()
    print(f"    [+] Saved to {png_path.name} and {pdf_path.name}")


def generate_user_study_chart():
    print("[*] Generating Figure 3: Pedagogical User Study Likert Ratings...")
    criteria = [
        'Factuality & Accuracy',
        'Pedagogical Clarity',
        'Educational Value',
        'Completeness',
        'Confidence Calibration'
    ]
    means = [4.80, 4.73, 4.67, 4.60, 4.33]
    errors = [0.41, 0.45, 0.48, 0.51, 0.62]

    y_pos = np.arange(len(criteria))

    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    bars = ax.barh(y_pos, means, xerr=errors, align='center', color='#0ea5e9', ecolor='#334155', capsize=5, edgecolor='black', linewidth=0.8)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(criteria, fontweight='bold')
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Mean Rating (1-5 Likert Scale)')
    ax.set_title('Student Pedagogical Evaluation Quality (PEQ, N=15)')
    ax.set_xlim(1.0, 5.2)
    ax.axvline(x=4.0, color='grey', linestyle=':', alpha=0.7)
    ax.grid(axis='x', linestyle='--', alpha=0.5)

    for bar, m in zip(bars, means):
        width = bar.get_width()
        ax.text(width + 0.15, bar.get_y() + bar.get_height()/2, f'{m:.2f} / 5.0',
                ha='left', va='center', fontweight='bold', fontsize=10, color='#0f172a')

    plt.tight_layout()
    png_path = OUTPUT_DIR / "fig3_user_study_peq.png"
    pdf_path = OUTPUT_DIR / "fig3_user_study_peq.pdf"
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)
    plt.close()
    print(f"    [+] Saved to {png_path.name} and {pdf_path.name}")


if __name__ == "__main__":
    generate_benchmark_comparison()
    generate_ablation_chart()
    generate_user_study_chart()
    print("[SUCCESS] All publication figures successfully generated in paper/figures/")
