# Arabic Handwritten Text Identification Using Local Feature Extraction Techniques

This project focuses on identifying Arabic handwritten text using local feature extraction methods like SIFT (Scale-Invariant Feature Transform) and ORB (Oriented FAST and Rotated BRIEF). The implementation evaluates the performance of these techniques based on accuracy, efficiency, and robustness.

---

## Table of Contents

- [Objective](#objective)
- [Dataset](#dataset)
- [Features](#features)
- [Installation](#installation)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Conclusion](#conclusion)
- [Acknowledgments](#acknowledgments)

---

## Objective

The goal is to explore and implement local feature extraction methods for identifying handwritten Arabic text and compare their performance. Key objectives include:

- Extracting features using SIFT and ORB.
- Using the Bag of Words (BoW) model to convert descriptors into feature vectors.
- Training a classifier to identify handwritten text based on extracted features.
- Comparing the performance of SIFT and ORB algorithms under various conditions.

---

## Dataset

The **AHAWP dataset** (Arabic Handwritten Automatic Word Processing) is used for this project. It contains:

- **10 unique Arabic words.**
- Handwritten samples by **82 individuals**, with 10 samples per word.
- A total of **8,144 grayscale images**.

Preprocessing steps include grayscale conversion, dataset augmentation (Gaussian noise, rotation, scaling, and illumination changes), and splitting into:

- **Training set (60%)**: 4,886 images.
- **Validation set (20%)**: 1,629 images.
- **Testing set (20%)**: 1,629 images.

The dataset can be downloaded from the [AHAWP dataset page](https://data.mendeley.com/datasets/2h76672znt/1/files/9031138a-b812-433e-a704-8acb1707936e).

---

## Features

- **Feature Extraction**:
  - SIFT: 128-dimensional descriptors for robust, detailed feature extraction.
  - ORB: Binary 32-dimensional descriptors for faster computation.

- **Feature Vector Formation**:
  - Visual Bag of Words (BoW) model.
  - Clustering using FAISS for GPU acceleration.
  - TF-IDF transformation for fixed-length vectors.

- **Classification**:
  - Logistic Regression for feature vector classification.

- **Robustness Testing**:
  - Augmentation with Gaussian noise, rotation, scaling, and illumination changes.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://https://github.com/Mohammed-Abed-Alkareem/Arabic-Handwriting-Identification-Local-Feature-Extraction
   cd Arabic-Handwritten-Text-Identification
   ```
---



## Evaluation Metrics

The following metrics are used to compare SIFT and ORB:

- **Accuracy**: Percentage of correctly classified samples.
- **Efficiency**: Time taken for feature extraction and vector formation.
- **Robustness**: Performance on augmented test sets with transformations.
- **Key Points**: Average key points detected per image.

---

## Results

### Key Findings:

- **Efficiency**:
  - ORB is significantly faster than SIFT but generates less detailed features.
  - ORB achieves feature extraction in **0.00135s per image**, compared to SIFT’s **0.01072s**.

- **Accuracy**:
  - SIFT achieves better training and validation accuracy.
  - Validation accuracy for SIFT peaks at **24%**, while ORB stagnates at **9%**.

- **Robustness**:
  - SIFT demonstrates higher resilience to transformations (noise, rotation, scaling, illumination changes).
  - ORB struggles under all augmented conditions, performing poorly in noisy or altered environments.

---

## Conclusion

- **SIFT**: Best for tasks requiring high accuracy and robustness. However, it is computationally intensive, making it less suitable for real-time applications.
- **ORB**: Efficient and lightweight, ideal for resource-constrained tasks but less reliable in challenging conditions.

### Recommendations:
- Use SIFT for detailed, robust analysis.
- Opt for ORB in time-sensitive or computationally limited scenarios.

---

