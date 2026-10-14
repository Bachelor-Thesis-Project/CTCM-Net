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
6.  **Training and evaluation:** 


## Datasets

The project workflow has been planned around the following datasets:


### Suggested data layout


## Environment Setup

## Data Preparation

## Training and Evaluation

## Evaluation Metrics


## Experiments and Results

## Reproducibility

## Limitations and Future Work

## References

## License

**Maintainer:** `<Mounika, Rounak, Arun, Mashub>`\
**Project:** CTCM-Net --- Cross-Task Consistency for Multimodal Visual
Understanding