"""
DiagramIQ — Complete Publication Figure Generation Script
Generates 4 high-resolution (300 DPI) publication-quality figures:
  1. Fig 1: End-to-End Neuro-Symbolic Pipeline Architecture (Vector graphic)
  2. Fig 2: Benchmark Performance Comparison (EMA, GMS, MGF, Hallucination)
  3. Fig 3: Ablation Study Component Contributions
  4. Fig 4: Pedagogical User Study (PEQ, N=15) 5-Point Likert Distribution
"""

import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Publication typography
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 13,
    'figure.dpi': 300
})

OUTPUT_DIR = Path(__file__).resolve().parent / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def generate_pipeline_architecture():
    print("[*] Generating Figure 1: Pipeline Architecture Diagram...")
    fig, ax = plt.subplots(figsize=(10.5, 3.8))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 38)
    ax.axis('off')

    stages = [
        {"title": "Input", "sub": "Diagram Image\n+ Query (q)", "x": 2, "w": 12, "color": "#f1f5f9", "edge": "#64748b"},
        {"title": "Stage 1", "sub": "Perception\nEasyOCR + LLaVA\n(Entity & BBoxes)", "x": 17, "w": 15, "color": "#e0f2fe", "edge": "#0284c7"},
        {"title": "Stage 2", "sub": "Symbolic Graph\nNetworkX DiGraph\n[MGF Gate >= 0.6]", "x": 35, "w": 15, "color": "#ede9fe", "edge": "#7c3aed"},
        {"title": "Stage 3", "sub": "Classification\nDomain Heuristics\n(UML/Arch/ER)", "x": 53, "w": 14, "color": "#fef3c7", "edge": "#d97706"},
        {"title": "Stage 4", "sub": "Graph Traversal\nSPOFs / Dijkstra\nFailure Cascade", "x": 70, "w": 14, "color": "#fee2e2", "edge": "#dc2626"},
        {"title": "Stage 5 & 6", "sub": "Grounded Output\nClaims + BBoxes\nCanonical JSON", "x": 87, "w": 12, "color": "#dcfce7", "edge": "#16a34a"},
    ]

    for s in stages:
        box = patches.FancyBboxPatch((s["x"], 6), s["w"], 24, boxstyle="round,pad=0.8",
                                     facecolor=s["color"], edgecolor=s["edge"], linewidth=1.8)
        ax.add_patch(box)
        ax.text(s["x"] + s["w"]/2, 25, s["title"], ha='center', va='center', fontweight='bold', fontsize=11, color='#0f172a')
        ax.text(s["x"] + s["w"]/2, 14, s["sub"], ha='center', va='center', fontsize=8.5, color='#334155', linespacing=1.2)

    # Draw arrows between stages
    arrow_props = dict(facecolor='#475569', edgecolor='#475569', width=1.5, headwidth=6, headlength=5, shrink=0.05)
    for i in range(len(stages) - 1):
        x_start = stages[i]["x"] + stages[i]["w"] + 0.8
        x_end = stages[i+1]["x"] - 0.8
        ax.annotate('', xy=(x_end, 18), xytext=(x_start, 18), arrowprops=arrow_props)

    # Title label at top
    ax.text(50, 35, "DiagramIQ 6-Stage Neuro-Symbolic Multimodal Reasoning Pipeline",
            ha='center', va='center', fontweight='bold', fontsize=12, color='#1e293b')

    plt.tight_layout()
    png_path = OUTPUT_DIR / "fig1_pipeline_architecture.png"
    pdf_path = OUTPUT_DIR / "fig1_pipeline_architecture.pdf"
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    print(f"    [+] Saved to {png_path.name} and {pdf_path.name}")


def generate_benchmark_comparison():
    print("[*] Generating Figure 2: Benchmark Comparison Chart...")
    models = ['Vanilla LLaVA 1.6', 'GPT-4o (Zero-Shot)', 'DiagramIQ (Ours)']
    ema = [48.2, 69.4, 89.6]
    mgf = [31.0, 58.4, 94.2]
    hallucination = [34.5, 18.2, 4.1]

    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    
    rects1 = ax.bar(x - width, ema, width, label='Exact Match Accuracy (EMA %)', color='#3b82f6', edgecolor='#1d4ed8', linewidth=1.0)
    rects2 = ax.bar(x, mgf, width, label='Grounding Fidelity (MGF %)', color='#10b981', edgecolor='#047857', linewidth=1.0)
    rects3 = ax.bar(x + width, hallucination, width, label='Hallucination Rate (%) [Lower is Better]', color='#ef4444', edgecolor='#b91c1c', linewidth=1.0)

    ax.set_ylabel('Percentage Score (%)', fontweight='bold')
    ax.set_title('Evaluation on DiagramIQ-Eval Benchmark (100 Software Diagrams)', fontweight='bold', pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.95)
    ax.set_ylim(0, 108)
    ax.grid(axis='y', linestyle='--', alpha=0.4)

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
    png_path = OUTPUT_DIR / "fig2_benchmark_comparison.png"
    pdf_path = OUTPUT_DIR / "fig2_benchmark_comparison.pdf"
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    print(f"    [+] Saved to {png_path.name} and {pdf_path.name}")


def generate_ablation_chart():
    print("[*] Generating Figure 3: Ablation Study Chart...")
    variants = [
        'Full Pipeline\n(DiagramIQ)',
        'Variant A\n(w/o Traversal)',
        'Variant B\n(w/o Graph)',
        'Variant C\n(w/o MGF Gating)'
    ]
    ema_scores = [89.6, 64.1, 51.0, 78.3]
    hrr_scores = [90.0, 52.4, 28.6, 68.1]

    x = np.arange(len(variants))
    width = 0.32

    fig, ax = plt.subplots(figsize=(7.8, 4.0))
    rects1 = ax.bar(x - width/2, ema_scores, width, label='Exact Match Accuracy (EMA %)', color='#6366f1', edgecolor='#4338ca', linewidth=1.0)
    rects2 = ax.bar(x + width/2, hrr_scores, width, label='Hallucination Reduction Rate (HRR %)', color='#f59e0b', edgecolor='#b45309', linewidth=1.0)

    ax.set_ylabel('Percentage Score (%)', fontweight='bold')
    ax.set_title('Ablation Study: Contribution of Individual Components', fontweight='bold', pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(variants, fontweight='bold', fontsize=9.5)
    ax.legend(loc='upper right', framealpha=0.95)
    ax.set_ylim(0, 108)
    ax.grid(axis='y', linestyle='--', alpha=0.4)

    for rects in [rects1, rects2]:
        for rect in rects:
            h = rect.get_height()
            ax.annotate(f'{h:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    png_path = OUTPUT_DIR / "fig3_ablation_study.png"
    pdf_path = OUTPUT_DIR / "fig3_ablation_study.pdf"
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    print(f"    [+] Saved to {png_path.name} and {pdf_path.name}")


def generate_user_study_chart():
    print("[*] Generating Figure 4: Pedagogical User Study Likert Ratings...")
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

    fig, ax = plt.subplots(figsize=(7.8, 3.8))
    bars = ax.barh(y_pos, means, xerr=errors, align='center', color='#0284c7', ecolor='#1e293b', capsize=5, edgecolor='#0369a1', linewidth=1.0)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(criteria, fontweight='bold', fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel('Mean Score (1 to 5 Likert Scale)', fontweight='bold')
    ax.set_title('Student Pedagogical Evaluation Quality (PEQ, N=15)', fontweight='bold', pad=10)
    ax.set_xlim(1.0, 5.25)
    ax.axvline(x=4.0, color='#94a3b8', linestyle=':', linewidth=1.5)
    ax.grid(axis='x', linestyle='--', alpha=0.4)

    for bar, m in zip(bars, means):
        width = bar.get_width()
        ax.text(width + 0.12, bar.get_y() + bar.get_height()/2, f'{m:.2f} / 5.0',
                ha='left', va='center', fontweight='bold', fontsize=9.5, color='#0f172a')

    plt.tight_layout()
    png_path = OUTPUT_DIR / "fig4_user_study_peq.png"
    pdf_path = OUTPUT_DIR / "fig4_user_study_peq.pdf"
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    print(f"    [+] Saved to {png_path.name} and {pdf_path.name}")


if __name__ == "__main__":
    generate_pipeline_architecture()
    generate_benchmark_comparison()
    generate_ablation_chart()
    generate_user_study_chart()
    print("[SUCCESS] All 4 publication figures generated successfully in paper/figures/")
