# CTCM-Net

### Cross-Task Consistency Learning for Multimodal Visual Understanding

CTCM-Net is a research framework for robust visual understanding in
complex environments. It explores how **RGB and thermal (RGB-T)
information** can support multiple complementary computer-vision tasks
within a unified system: **semantic segmentation, crowd counting, and
human action recognition**.

The framework is designed around multimodal feature learning and
cross-task consistency, with the goal of using complementary visual cues
to improve understanding in challenging conditions such as low
illumination and visually complex scenes.

> **Project status:** Research / academic project. Update this README to
> reflect the modules, experiments, and results that are implemented in
> the current repository.

------------------------------------------------------------------------

## Table of Contents

-   [Motivation](#motivation)
-   [Objectives](#objectives)
-   [Tasks](#tasks)
-   [Architecture Overview](#architecture-overview)
-   [Datasets](#datasets)
-   [Repository Structure](#repository-structure)
-   [Environment Setup](#environment-setup)
-   [Data Preparation](#data-preparation)
-   [Training and Evaluation](#training-and-evaluation)
-   [Evaluation Metrics](#evaluation-metrics)
-   [Experiments and Results](#experiments-and-results)
-   [Reproducibility](#reproducibility)
-   [Limitations and Future Work](#limitations-and-future-work)
-   [References](#references)
-   [License](#license)

## Motivation

RGB cameras capture rich colour and texture information, while thermal
sensors can provide useful heat-based cues when visible-light imagery is
degraded. However, the modalities have different characteristics, and a
model must combine them carefully.

CTCM-Net investigates a multi-task approach in which related visual
tasks can benefit from multimodal representations and consistency
constraints. The intended system answers three complementary questions:

-   **Where?** Semantic segmentation identifies the class of each pixel.
-   **How many?** Crowd counting estimates the number of people in a
    scene.
-   **What are they doing?** Human action recognition predicts an action
    category.

## Objectives

-   Explore RGB-T feature fusion for challenging visual environments.
-   Support task-specific prediction heads for segmentation, crowd
    counting, and action recognition.
-   Investigate cross-task consistency signals between related
    predictions.
-   Compare RGB-only, thermal-only, and RGB-T configurations where
    supported by the implementation.
-   Evaluate performance using task-appropriate metrics and reproducible
    experiment settings.

## Tasks

  -----------------------------------------------------------------------
  Task              Input             Output            Typical
                                                        evaluation
  ----------------- ----------------- ----------------- -----------------
  Semantic          RGB and/or        Pixel-wise class  Mean Intersection
  segmentation      thermal image     mask              over Union
                                                        (mIoU), pixel
                                                        accuracy

  Crowd counting    RGB and/or        Count estimate or Mean Absolute
                    thermal image     density map       Error (MAE), Mean
                                                        Squared Error
                                                        (MSE)

  Human action      RGB-T image       Action class      Accuracy,
  recognition       sequence or                         precision,
                    clips, depending                    recall, F1-score
                    on dataset                          
  -----------------------------------------------------------------------

The exact input format, prediction heads, and metrics should match the
current implementation and dataset protocols.

## Architecture Overview

At a conceptual level, the framework can be organised as follows:

1.  **Multimodal input:** Receive paired RGB and thermal data when
    available.
2.  **Preprocessing and alignment:** Validate modality pairs and apply
    compatible resizing and normalization.
3.  **Feature extraction and fusion:** Extract modality-specific
    features and combine complementary RGB-T information.
4.  **Task-specific heads:** Generate segmentation, crowd-counting, and
    action-recognition predictions.
5.  **Cross-Task Consistency Module (CTCM):** Apply consistency
    objectives between relevant task predictions or representations,
    where implemented.
6.  **Training and evaluation:** Optimise the configured objectives and
    report task-specific metrics.

``` text
                 RGB Input          Thermal Input
                      \                /
                       \              /
                    Preprocessing & Alignment
                              |
                    RGB-T Feature Extraction
                              |
                       Multimodal Fusion
                              |
              +---------------+---------------+
              |               |               |
         Segmentation     Crowd Counting   Action Recognition
              |               |               |
              +---------------+---------------+
                              |
                  Cross-Task Consistency
                    (where implemented)
                              |
                     Joint Training Loss
```

This is a **conceptual overview**, not a claim that every component is
already implemented. Use the repository's actual model diagram and loss
definitions when available.

## Datasets

The project workflow has been planned around the following datasets:

-   **MFNet** --- RGB-thermal urban-scene semantic segmentation.
-   **RGBT-CC** --- RGB-thermal crowd counting.
-   **DarkAct / ARIM** --- action-recognition data for challenging or
    low-light scenes, depending on the selected dataset version and
    split.

Follow each dataset's official instructions for access, usage
permissions, annotations, and train/validation/test splits. Dataset
files are not included in this repository unless explicitly stated.

### Suggested data layout

Adapt this layout to your actual downloaded files:

``` text
datasets/
├── mfnet/
├── RGBT-CC-CVPR2021/
└── darkact_datasets/
    └── ARIM_v1/
```

Do not commit large datasets, private data, or files whose licences do
not permit redistribution. Consider using `.gitignore` to exclude local
data and experiment outputs.

## Repository Structure

A suggested structure is shown below. Adjust it to match the files
currently present in the repository.

``` text
CTCM-Net/
├── configs/                 # Experiment configurations
├── datasets/                # Local datasets (usually git-ignored)
├── models/                  # Encoders, fusion blocks, task heads
├── scripts/                 # Training and evaluation scripts
├── notebooks/               # Exploratory and experiment notebooks
├── results/                 # Metrics, plots, and qualitative outputs
├── checkpoints/             # Saved model weights (usually git-ignored)
├── requirements.txt         # Python dependencies
├── .gitignore
└── README.md
```

## Environment Setup

The commands below provide a typical PyTorch setup. Choose the PyTorch
build that matches your Python version and available CPU/GPU
environment.

``` bash
git clone <YOUR_REPOSITORY_URL>
cd CTCM-Net

python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `requirements.txt` has not been created yet, install the project's
verified dependencies and record their versions before running
experiments. For GPU training, install a PyTorch build compatible with
the available CUDA runtime; a CUDA-capable GPU is not required for
CPU-based code checks.

## Data Preparation

Before training:

1.  Download each dataset from its official source and review its
    licence.
2.  Arrange the files according to the dataset loader's expected
    directory structure.
3.  Verify that RGB and thermal samples correspond to the same scene or
    timestamp where paired data is required.
4.  Check image dimensions, labels, annotations, and missing files.
5.  Keep train, validation, and test splits separate.
6.  Compute normalization statistics from the training split only, where
    applicable.
7.  Record the split definitions and preprocessing settings used in each
    experiment.

Do not normalize segmentation label masks as if they were ordinary image
channels.

## Training and Evaluation

Training commands depend on the scripts and configuration files included
in the repository. Replace the example paths below with the actual entry
points.

``` bash
# Example only — use the real script and arguments from this repository
python scripts/train.py --config configs/experiment.yaml
python scripts/evaluate.py --config configs/experiment.yaml
```

If separate task-specific pipelines are used, document their actual
commands individually. Record random seeds, hyperparameters, dataset
splits, checkpoint paths, and software versions so that results can be
reproduced.

## Evaluation Metrics

Use the metrics appropriate to each task and report the evaluation
protocol alongside the results.

-   **Semantic segmentation:** mIoU and pixel accuracy; per-class IoU is
    useful for identifying class-level weaknesses.
-   **Crowd counting:** MAE and MSE between predicted and ground-truth
    counts, using the dataset's prescribed protocol.
-   **Action recognition:** Top-1 accuracy and, where useful, precision,
    recall, and F1-score.

Compare modality settings only when the data splits, training budget,
and evaluation procedure are comparable.

## Experiments and Results

Add measured results here once the corresponding experiments have
completed. Do not report placeholder values as experimental findings.

  -----------------------------------------------------------------------
  Task              Configuration     Metric            Result
  ----------------- ----------------- ----------------- -----------------
  Semantic          RGB-only          mIoU              To be reported
  segmentation                                          

  Semantic          RGB-T             mIoU              To be reported
  segmentation                                          

  Crowd counting    RGB-only / RGB-T  MAE               To be reported

  Action            RGB-only / RGB-T  Accuracy          To be reported
  recognition                                           

  Cross-task        Enabled /         Task-specific     To be reported
  consistency       disabled ablation metrics           
  -----------------------------------------------------------------------

Recommended figures include qualitative segmentation masks, predicted
versus ground-truth crowd counts, action-recognition examples, and
ablation comparisons. Include the checkpoint, dataset split, and
configuration used to produce each result.

## Reproducibility

For every reported experiment, aim to record:

-   Dataset version and split
-   Model and configuration name
-   Random seed
-   Optimizer, learning rate, batch size, and number of epochs
-   Preprocessing and augmentation settings
-   Checkpoint selection criterion
-   Software and hardware environment
-   Evaluation script and metric definitions

## Limitations and Future Work

Potential areas for further investigation include:

-   Improving robustness when one modality is noisy, missing, or
    misaligned.
-   Studying alternative adaptive fusion strategies.
-   Measuring whether cross-task consistency improves generalisation.
-   Conducting ablation studies for each loss and model component.
-   Evaluating generalisation across datasets and environmental
    conditions.
-   Improving inference efficiency for practical deployment.

These are research directions, not claims of completed functionality.

## References

Add the exact papers, dataset publications, and implementation sources
used in this repository. Relevant starting points for the planned
project include work on MFNet, multispectral semantic segmentation,
RGB-T crowd counting, and low-light action recognition. Verify
bibliographic details and cite the exact methods actually used.

## License

No licence is specified here. Add a `LICENSE` file before distributing
the repository, and ensure that the chosen licence is compatible with
the code and any third-party components. Dataset licences may be
separate from the repository's software licence.

------------------------------------------------------------------------

**Maintainer:** `<Mounika, Rounak, Arun, Mashub>`\
**Project:** CTCM-Net --- Cross-Task Consistency for Multimodal Visual
Understanding