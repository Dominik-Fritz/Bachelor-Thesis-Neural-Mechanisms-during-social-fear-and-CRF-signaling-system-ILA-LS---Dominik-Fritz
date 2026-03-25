
# Neural Mechanisms during Social Fear and CRF Signaling in the ILA–LS Circuit

This repository contains all scripts used for data processing, analysis, and figure generation for the study:

"Neural Mechanisms of Social Fear Extinction in Mice: Role of Lateral Septal CRF Signaling and State-Dependent c-Fos Brain Mapping."

The repository is organized to mirror the structure of the analyses presented in the manuscript in order to facilitate transparency, reproducibility, and reuse of the analysis pipeline.

---

Repository Structure

classifier_and_model_validation/
Scripts used for validation of segmentation models and classifiers.

image_prep/
ImageJ macros used for preprocessing of microscopy images.

models/
Pre-trained models used for nucleus segmentation and classification.

qupath_scripts/
Scripts used for automated image analysis and signal quantification within QuPath.

statistical_analysis/
Python scripts used for statistical analysis of the c-Fos whole-brain mapping dataset.

schematic_figures/
Scripts used to generate schematic illustrations explaining experimental design and figure generation workflows.

---

Analysis Pipeline Overview

The analysis pipeline implemented in this repository follows the workflow used in the manuscript.

1. Image Preparation

Raw microscopy images are first prepared using ImageJ macros contained in the image_prep/ directory.

Typical preprocessing steps include:
- channel separation
- Z-projection of image stacks
- contrast normalization
- format conversion for downstream analysis

These steps standardize image input for segmentation and downstream analysis.

2. Model Preparation

Segmentation and classification models used in the analysis are stored in the models/ directory.

Two different segmentation approaches were used depending on the dataset:

StarDist – used for segmentation of nuclei in RNAscope datasets
Cellpose – used for segmentation of nuclei in c-Fos whole-brain mapping datasets

The prepared models are loaded and applied within QuPath using the scripts provided in the repository.

3. Image Analysis in QuPath

Image analysis is performed using scripts located in:

qupath_scripts/

These scripts apply the trained segmentation models within QuPath and perform automated object detection and classification.

Detected objects are quantified and exported as CSV files containing raw measurement data, which serve as the basis for downstream analyses.

4. Model and Classifier Validation

Validation of segmentation models and classification pipelines is performed using scripts contained in:

classifier_and_model_validation/

These scripts compute evaluation metrics including:
- precision
- recall
- F1 score

Validation ensures robust segmentation and classification performance across datasets.

5. Statistical Analysis

Statistical analyses are performed using Python scripts located in:

statistical_analysis/

These scripts are used specifically for the c-Fos whole-brain mapping dataset, where regional activity values are aggregated and statistical comparisons between experimental groups are performed.

6. Figure Generation and Schematics

Scripts used to generate schematic illustrations explaining experimental workflows and neural circuitry are located in:

schematic_figures/

These scripts reproduce schematic figures used in the manuscript and document how the visualizations were generated.

---

Reproducibility

All scripts required to reproduce the analyses and figures reported in the manuscript are provided in this repository.

The folder structure reflects the organization of the analyses described in the manuscript and allows straightforward navigation of the workflow.

Each script contains comments describing its purpose and expected input format.

---

Data Organization

The analyses in this repository are based on processed datasets derived from histological imaging and behavioral experiments described in the manuscript.

Due to file size limitations and institutional data policies, raw microscopy images and full intermediate datasets are not included in this repository.

Instead, the repository contains:
- analysis scripts used for processing and statistical analysis
- trained segmentation and classification models
- example data structures illustrating the format expected by the analysis scripts
- scripts used for generating figures included in the manuscript

Users wishing to reproduce the analysis pipeline should adapt the input paths within the scripts to match their own local data structure.

---

External Software

The analysis pipeline relies on several external software tools commonly used in histological image analysis:

QuPath – digital pathology and automated image quantification
ImageJ / Fiji + ABBA – image preprocessing and macro-based processing
Cellpose – deep learning–based cell segmentation (c-Fos datasets)
StarDist – nucleus segmentation for RNAscope datasets
Python – statistical analysis and figure generation

Users reproducing the pipeline should ensure that compatible versions of these tools are installed.

---

Requirements

The analyses were performed using Python 3.11 (except Cellpose, which was run using Python 3.8).

Required Python packages may include:
- numpy
- pandas
- matplotlib
- scipy
- seaborn

Additional dependencies are specified within the individual scripts.

---

Usage

Scripts can be executed individually depending on the analysis step.

Typical usage follows the pipeline described above:

1. Prepare microscopy images using macros in image_prep/
2. Load trained segmentation models from models/
3. Apply segmentation and classification using qupath_scripts/
4. Export quantified measurements as CSV files
5. Validate models using classifier_and_model_validation/
6. Perform statistical analyses using statistical_analysis/
7. Generate schematic figures using schematic_figures/

---

Code Availability

All scripts used for data processing, analysis, and figure generation are publicly available in this repository.

The repository is intended to provide full transparency for the analysis pipeline used in the study and to facilitate reproducibility of the reported results.

---

Data Availability

Due to the size of the raw microscopy datasets, raw images are not included in this repository.

Example data structures and analysis scripts are provided to illustrate the analysis workflow.

Additional data may be made available upon reasonable request.

---

Citation

If you use the scripts or models provided in this repository, please cite the associated study:

Fritz, Dominik
Neural Mechanisms of Social Fear Extinction in Mice: Role of Lateral Septal CRF Signaling and State-Dependent c-Fos Brain Mapping.

---

Contact

For questions regarding the code or analyses, please contact:

Dominik.Fritz@stud.uni-regensburg.de
