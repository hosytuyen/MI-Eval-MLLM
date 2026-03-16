## Official Implementation of FMLLM (CVPR 2026 Findings)

**TL;DR:** Current model inversion evaluation framework is misleading, with up to 90% false positives. Our proposed multimodal evaluation framework FMLLM mitigates this.

**Abstract:** We include the supplementary code for our research on Model Inversion (MI) evaluation methods. We propose a novel framework for assessing MI attacks using Machine Learning Language Models (MLLMs) and demonstrate the limitations of conventional MI evaluation approaches.

We include a dataset of comprehensive collection of Model Inversion (MI) attack reconstructions evaluated using our proposed MLLM-based evaluation framework. It contains reconstructed images from SOTA and common MI attacks spanning different private dataset, public dataset, MI attack, and target model T.

## Environment Setup

### Using Conda
```bash
# Create and activate conda environment
conda create -n mi-eval python=3.8
conda activate mi-eval

# MI-Eval-MLLM

Reproducible MLLM-based evaluation for Model Inversion (MI) attacks (CVPR 2026 Findings supplemental code).

Quickstart

1. Clone the repo and download the dataset (Kaggle: https://www.kaggle.com/datasets/hosytuyen/mi-reconstruction-collection) into `AttackSamples/`.

2. Create environment and install dependencies:

```bash
conda create -n mi-eval python=3.9 -y
conda activate mi-eval
pip install -r requirements.txt
```

Data layout

Place reconstructed images under `AttackSamples/`.

```
./AttackSamples/                              # Dataset Folder
├── Facescrub/                              # Private Datasets
│   ├── IFGMI/                              # MI Attack 
│       ├── FFHQ/                           # Public dataset
|       |   |───Resnet18                    # T
|       |       |────all-images.zip         # Collention of reconstructed images
|       |       |────gemini_results.csv     # Label
│       └── MetFaces/
│
├── CelebA/
│   
│
└── Stanford_Dogs/
|
├── utility.py              # Core utilities for image processing and evaluation
├── gemini_evaluation.ipynb # MLLM-based evaluation implementation
└── analysis.ipynb          # Analysis of traditional MI evaluation limitations
```

Run evaluation

- Open [gemini_evaluation.ipynb](gemini_evaluation.ipynb) to configure your MLLM API key and run the evaluation.
- Use helper routines in `utility.py` for batch evaluation and CSV export.

Notebooks

- `gemini_evaluation.ipynb` — MLLM evaluation walkthrough
- `analysis.ipynb` — experiments and analysis


