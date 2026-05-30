<h1 align="center">FMLLM Evaluation Framework for Model Inversion Attack</h1>

<p align="center">
    <b>Revisiting Model Inversion Evaluation:</b><br>
<b>From Misleading Standards to Reliable Privacy Assessment (CVPR 2026 Findings)</b>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2505.03519">
    <img src="https://img.shields.io/badge/Paper-arXiv-b31b1b.svg" alt="Paper">
  </a>
  <a href="https://hosytuyen.github.io/projects/FMLLM/index.html">
    <img src="https://img.shields.io/badge/Code-GitHub-181717.svg" alt="Code">
  </a>
  <a href="https://huggingface.co/datasets/hosytuyen/MI-Reconstruction-Collection">
    <img src="https://img.shields.io/badge/Dataset-HuggingFace-ffcc4d.svg" alt="Dataset">
  </a>
  <a href="https://github.com/hosytuyen/MI-Eval-MLLM">
    <img src="https://img.shields.io/badge/Project-Page-2ea44f.svg" alt="Project">
  </a>
</p>

## Overview

Abstract: Model Inversion (MI) attacks aim to reconstruct information from private training data by exploiting access to a target model. Nearly all recent MI studies evaluate attack success using a standard framework that computes attack accuracy through a secondary evaluation model trained on the same private data and task design as the target model.

In this paper, we present the first in-depth analysis of this dominant evaluation framework and reveal a fundamental issue: many reconstructions deemed "successful" are in fact false positives that do not capture the visual identity of the target individual. We show these MI false positives satisfy the same formal conditions as Type I adversarial examples, and demonstrate extremely high false-positive transferability.

To address this, we introduce a new evaluation framework FMLLM based on Multimodal Large Language Models, whose general-purpose visual reasoning avoids the shared-task vulnerability. We reassess 27 MI attack setups and find consistently high false-positive rates under the conventional approach — calling for a reevaluation of progress in MI research.

This repository includes:

- notebook-based code for generating MLLM evaluation queries
- unified wrappers for Gemini, Qwen-VL-style endpoints, and ChatGPT-based evaluators
- analysis utilities for comparing FMLLM labels against conventional MI evaluation outputs


## Download MI-Reconstruction-Collection

The released reconstruction collection is hosted on Hugging Face:

```bash
git clone https://huggingface.co/datasets/hosytuyen/MI-Reconstruction-Collection AttackSamples
```

Alternatively, you can download the dataset manually from the dataset page and place it under `AttackSamples/`.

The downloaded collection should contain MI reconstructions across multiple private datasets, public datasets, attack methods, and target models. The code in this repo expects the dataset to live under `AttackSamples/`.

Each evaluation setup should contain:

- `all-images/`: reconstructed images to evaluate
- `private-data/`: class-organized reference images used to build `Set B`
- result CSV files such as `gemini_results_<model>.csv` or `prediction.csv`

Example structure:

```text
AttackSamples/
├── Facescrub/
│   ├── IFGMI/
│   │   ├── FFHQ/
│   │   │   └── Resnet18/
│   │   │       ├── all-images/
│   │   │       ├── private-data/
│   │   │       └── ...
│   │   └── ...
├── CelebA/
├── Cifa100/
└── Stanford_Dogs/
```

## Environment Setup

Create a clean environment and install dependencies:

```bash
conda create -n mi-eval python=3.9 -y
conda activate mi-eval
pip install -r requirements.txt
```

The main dependencies are:

- `torch`, `torchvision` for image preprocessing
- `pillow` for composing evaluation panels
- `google-generativeai` for Gemini-based evaluation
- `gradio-client` for Qwen-VL-style remote evaluators
- `openai<1.0.0` for ChatGPT-based evaluation helpers
- `pandas` for analysis

## Basic Usage

### Step 1: Create MLLM evaluation queries

Open `evaluation.ipynb` and configure:

- `setup_folder`, for example `AttackSamples/CelebA/PLGMI/CelebA/VGG16`
- your API key
- the evaluator model name, for example `gemini-2.0-flash`

The notebook uses `create_concatenated_images_with_labels(...)` in `utility.py` to build evaluation panels where:

- `Image A` is the reconstructed MI sample
- `Set B` is a small reference set from the candidate private class

The generated query images are written to a `user-study/` directory next to the original `all-images/` folder.

### Step 2: Evaluate the generated queries

After query generation, run the evaluator on the generated images. For example:

```python
from utility import *

processor = GeminiProcessor(GEMINI_API_KEY, MODEL_NAME)
evaluate_images_in_directory_unified(
    directory_path=output_folder,
    processor=processor,
    model_name=MODEL_NAME,
    max_images=10000,
    output_prefix="gemini",
)
```

This produces a CSV file beside the evaluated folder.

## Analysis

Use `analysis.ipynb` to compare FMLLM ground-truth labels against predictions from conventional MI evaluation pipelines and compute false positive statistics.

Example:

```python
base_path = "AttackSamples/Cifa100/PPA/Cifar10/Resnet18"
ground_truth_path = os.path.join(base_path, "gemini_results_gemini-2.0-flash.csv")
prediction_path = os.path.join(base_path, "prediction.csv")

merged = compute_metrics(ground_truth_path, prediction_path)
```

This reproduces the core measurement used in the paper: how often the standard evaluation pipeline counts a reconstruction as successful when the MLLM-based evaluation rejects it.


## Citation

If you use this repository, please cite:

```bibtex
@article{ho2025revisiting,
  title     = {Revisiting Model Inversion Evaluation: From Misleading Standards to Reliable Privacy Assessment},
  author    = {Ho, Sy-Tuyen and Koh, Jun Hao and Nguyen, Ngoc-Bao and Binder, Alexander and Cheung, Ngai-Man},
  journal   = {arXiv preprint arXiv:2505.03519},
  year      = {2025}
}
```
