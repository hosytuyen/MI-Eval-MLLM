## Authors
- Sy-Tuyen Ho¹*
- Koh Jun Hao¹*
- Ngoc-Bao Nguyen¹*
- Alexander Binder²
- Ngai-Man Cheung¹

¹ Singapore University of Technology and Design (SUTD)  
² Leipzig University  
\* Equal contribution

## Overview
This repository contains the supplementary code for our research on Model Inversion (MI) evaluation methods. We propose a novel framework for assessing MI attacks using Machine Learning Language Models (MLLMs) and demonstrate the limitations of conventional MI evaluation approaches.

## Project Structure

Download AttackSamples from here and extract. 

```
.
├── AttackSamples/          # Contains reconstructed images from 26 MI attack setups
│   └── CelebA/
│       └── KEDMI/
│           └── FFHQ/
│               └── FaceNet64/
│                   └── all-images/
│                       ├── 0/     # Class 0 reconstructed images
│                       ├── 1/     # Class 1 reconstructed images
│                       └── 2/     # Class 2 reconstructed images
├── utility.py              # Core utilities for image processing and evaluation
├── gemini_evaluation.ipynb # MLLM-based evaluation implementation
└── analysis.ipynb          # Analysis of traditional MI evaluation limitations
```

## Environment Setup

### Using Conda
```bash
# Create and activate conda environment
conda create -n mi-eval python=3.8
conda activate mi-eval

# Install required packages
pip install torch torchvision
pip install pillow
pip install google-generativeai
pip install jupyter notebook
pip install pathlib
```

## Data Preparation
1. For evaluating a new MI attack, prepare your reconstructed images in the following structure:
   ```
   AttackSamples/
   └── your_private_dataset/
       └── your_public_attack/
           └── your_target_model/
               └── all-images/
                   ├── 0/    # Class 0 reconstructed images
                   ├── 1/    # Class 1 reconstructed images
                   └── 2/    # Class 2 reconstructed images
   ```
2. Each subfolder (0, 1, 2) should contain the reconstructed .png/.jpg/.jpeg images for that class
3. We provide examples of reconstructed images from 26 different MI setups in the `AttackSamples` directory

## MLLM-based MI Evaluation Framework

1. Get a Gemini API key from Google AI Studio
2. Set up your environment as described above
3. Follow the step-by-step instructions in `gemini_evaluation.ipynb` for the evaluation


## (Optional) Additional Analysis
We provide implementation to investigate the limitations of common standard MI evaluation frameworks in `analysis.ipynb`. 

