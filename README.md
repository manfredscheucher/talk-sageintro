# Solving Math Problems with SageMath — an Introduction

A Jupyter Notebook slideshow introducing SageMath: logic, optimization, graph
theory and more. The talk (`talk_en.ipynb`) is organized into 4 parts as natural
stopping points: **1** What is Sage + Basics · **2** Math Applications ·
**3** Geometry & Graphs · **4** Optimization & Solvers.

## Install SageMath

**macOS**
```bash
brew install --cask sage  # or: download from sagemath.org/download.html
```

**Linux** (Ubuntu/Debian)
```bash
sudo apt install sagemath
# or via conda-forge:
conda create -n sage sage python=3.12
```

**Windows** (WSL2 recommended)
```bash
# install WSL2 with Ubuntu, then follow the Linux steps
sudo apt install sagemath
```

More info: [doc.sagemath.org/html/en/installation](https://doc.sagemath.org/html/en/installation/)

## Run the notebook

```
sage --notebook jupyter
```

## Export slides

Generate the slides as HTML:
```bash
sage -sh -c "jupyter nbconvert --to slides talk_en.ipynb"
```

Open interactively in the browser (arrow keys / spacebar):
```
talk_en.slides.html
```

Print to PDF: open `talk_en.slides.html?print-pdf` in the browser → Cmd+P → Save as PDF.

## Export as PDF

Requires pandoc:
```bash
brew install pandoc      # macOS
sudo apt install pandoc  # Linux
```

Then: `sage --notebook jupyter` → in the browser: **File → Save and Export Notebook As → PDF**
