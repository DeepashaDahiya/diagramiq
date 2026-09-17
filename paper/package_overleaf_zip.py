"""
DiagramIQ — Overleaf Package Bundler
Creates a 1-click ready-to-upload zip archive for Overleaf.
"""

import os
import zipfile
from pathlib import Path

PAPER_DIR = Path(__file__).resolve().parent
ZIP_PATH = PAPER_DIR / "diagramiq_paper_overleaf.zip"

def create_overleaf_zip():
    print(f"[*] Packaging Overleaf ready archive: {ZIP_PATH.name}...")
    
    files_to_zip = [
        PAPER_DIR / "main.tex",
        PAPER_DIR / "references.bib",
    ]

    figures_dir = PAPER_DIR / "figures"
    
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zipf:
        for f in files_to_zip:
            if f.exists():
                zipf.write(f, arcname=f.name)
                print(f"    [+] Added {f.name}")
        
        if figures_dir.exists():
            for fig in figures_dir.glob("*.*"):
                arcname = f"figures/{fig.name}"
                zipf.write(fig, arcname=arcname)
                print(f"    [+] Added {arcname}")

        # Add a helpful README.txt for Overleaf
        readme_content = """DiagramIQ — Springer LNCS Research Paper Source Bundle
======================================================

How to view & compile this paper on Overleaf:
1. Go to https://www.overleaf.com and log in.
2. Click "New Project" -> "Upload Project".
3. Select this zip file ("diagramiq_paper_overleaf.zip").
4. Once loaded, ensure "main.tex" is selected as the main document.
5. Click "Recompile" (or Ctrl+S / Cmd+S).
6. The formatted camera-ready PDF will appear in the right-hand preview panel.

Authors: Deepasha, [Faculty Advisor / Mentor]
Title: DiagramIQ: A Neuro-Symbolic Multimodal Reasoning System for Structured Software Design Diagram Interpretation in Computing Education
"""
        zipf.writestr("README_OVERLEAF.txt", readme_content)
        print("    [+] Added README_OVERLEAF.txt")

    print(f"\n[SUCCESS] Overleaf bundle created: {ZIP_PATH}")
    print(f"    Size: {ZIP_PATH.stat().st_size / 1024:.2f} KB")

if __name__ == "__main__":
    create_overleaf_zip()
