# Neural Mechanisms during Social Fear and CRF Signaling in the ILA–LS Circuit

This repository contains all scripts used for data processing, analysis, and figure generation for the study:

**"Neural Mechanisms of Social Fear Extinction in Mice: Role of Lateral Septal CRF Signaling and State-Dependent c-Fos Brain Mapping."**

The repository is organized to mirror the structure of the analyses presented in the manuscript in order to facilitate **transparency**, **reproducibility**, and **reuse of the analysis pipeline**.

---

## Repository Structure

The scripts are organized according to the analyses described in the manuscript.

**classifier_and_model_validation/**  
Scripts used for validation of segmentation models and classifiers.

**image_prep/**  
ImageJ macros used for preprocessing of microscopy images.

**models/**  
Trained models used for segmentation and classification.

**qupath_scripts/**  
Scripts used for automated image quantification within QuPath.

**statistical_analysis/**  
Scripts used for statistical analysis of behavioral and histological data.

**schematic_figures/**  
Scripts used to generate schematic illustrations of the experimental design and neural circuitry presented in the manuscript.

---

## Analysis Pipeline Overview

The analysis pipeline implemented in this repository follows the workflow used in the manuscript.

### 1. Image Preparation

Raw microscopy images are first prepared using ImageJ macros contained in the **image_prep/** directory.

Typical preprocessing steps include:

- channel separation  
- Z-projection of image stacks  
- contrast normalization  
- format conversion for downstream analysis  

These steps standardize image input for segmentation and classification.

---

### 2. Nuclei Segmentation

Segmentation of cell nuclei is performed using the trained models provided in the **models/** directory.

The segmentation models were trained using **Cellpose** and applied to detect nuclei across whole-brain sections and RNAscope images.

Segmentation produces object masks that are subsequently used for downstream classification and quantification.

---

### 3. Cell Classification

Following segmentation, detected nuclei are classified using trained classifiers implemented in **QuPath**.

Classifier scripts and workflows are located in:

**qupath_scripts/**

These classifiers distinguish signal-positive cells from background based on morphological and fluorescence features.

---

### 4. Model and Classifier Validation

Validation of segmentation models and classification pipelines is performed using scripts contained in:

**classifier_and_model_validation/**

These scripts compute evaluation metrics including:

- precision  
- recall  
- F1 score  

Validation ensures robust detection and classification performance across images.

---

### 5. Statistical Analysis

Statistical analyses are performed using Python scripts located in:

**statistical_analysis/**

These scripts implement statistical testing and data aggregation used to generate the quantitative results presented in the manuscript.

Typical analyses include:

- group comparisons  
- summary statistics  
- dataset preprocessing  

---

### 6. Image Quantification and Figure Generation

Final quantification of detected signals and generation of schematic figures are performed using scripts located in:

**qupath_scripts/**  
**schematic_figures/**

These scripts produce the quantitative outputs and schematic illustrations used in the manuscript.

---

## Reproducibility

All scripts required to reproduce the analyses and figures reported in the manuscript are provided in this repository.

The folder structure reflects the organization of the analyses described in the manuscript and allows straightforward navigation of the workflow.

Each script contains comments describing its purpose and expected input format.

---

## Data Organization

The analyses in this repository are based on processed datasets derived from histological imaging and behavioral experiments described in the manuscript.

Due to file size limitations and institutional data policies, **raw microscopy images and full intermediate datasets are not included in this repository**.

Instead, the repository contains:

- analysis scripts used for processing and statistical analysis  
- trained segmentation and classification models  
- example data structures illustrating the format expected by the analysis scripts  
- scripts used for generating figures included in the manuscript  

Users wishing to reproduce the analysis pipeline should adapt the input paths within the scripts to match their own local data structure.

---

## External Software

The analysis pipeline relies on several external software tools commonly used in histological image analysis:

- **QuPath** – digital pathology and automated image quantification  
- **ImageJ / Fiji** – image preprocessing and macro-based processing  
- **Cellpose** – deep learning–based cell segmentation  
- **Python** – statistical analysis and figure generation  

Users reproducing the pipeline should ensure that compatible versions of these tools are installed.

---

## Requirements

The analyses were performed using **Python 3.11**.

Required Python packages may include:

- numpy  
- pandas  
- matplotlib  
- scipy  
- seaborn  

Additional dependencies are specified within the individual scripts.

---

## Usage

Scripts can be executed individually depending on the analysis step.

Typical usage follows the pipeline described above:

1. Prepare images using macros in **image_prep/**  
2. Perform segmentation using models in **models/**  
3. Run classification using scripts in **qupath_scripts/**  
4. Validate models using scripts in **classifier_and_model_validation/**  
5. Perform statistical analyses using scripts in **statistical_analysis/**  
6. Generate schematic figures using scripts in **schematic_figures/**  

---

## Code Availability

All scripts used for data processing, analysis, and figure generation are publicly available in this repository.

The repository is intended to provide full **transparency for the analysis pipeline** used in the study and to facilitate **reproducibility of the reported results**.

---

## Data Availability

Due to the size of the raw microscopy datasets, raw images are not included in this repository.

Example data structures and analysis scripts are provided to illustrate the analysis workflow.

Additional data may be made available upon reasonable request.

---

## Citation

If you use the scripts or models provided in this repository, please cite the associated study:

**Fritz, Dominik**  
Neural Mechanisms of Social Fear Extinction in Mice: Role of Lateral Septal CRF Signaling and State-Dependent c-Fos Brain Mapping.

---

## Contact

For questions regarding the code or analyses, please contact:

Dominik.Fritz@stud.uni-regensburg.de
