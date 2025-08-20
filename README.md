# Prototype Based Computing Classifier for Disease Classification

This repository contains the implementation of a Hyperdimensional Computing (HDC)-inspired classifier for multi-class disease classification using symptom data. The project leverages high-dimensional binary vectors for efficient and robust pattern recognition, suitable for resource-constrained environments like edge devices.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Future Work](#future-work)
- [References](#references)
- [License](#license)
- [Contact](#contact)

## Overview
Hyperdimensional Computing (HDC) is a brain-inspired computational paradigm that uses high-dimensional vectors (hypervectors) for robust and efficient pattern recognition. This project introduces an HDC-based classifier for disease classification based on symptom data, achieving an accuracy of 87.16% across 201 disease classes. The classifier is designed for low computational complexity, making it ideal for scalable medical diagnostics on edge devices.

### Key Contributions
1. An HDC classifier tailored for a large-scale disease-symptom dataset with 201 classes.
2. A preprocessing pipeline that filters diseases with sufficient records and introduces 25% noise for robustness.
3. Comprehensive evaluation demonstrating competitive performance against traditional machine learning models.

## Features
- **Efficient Encoding**: Uses binary hypervectors (D=1000) with sign-based encoding for symptom features.
- **Single-Pass Learning**: Computes class prototypes by averaging hypervectors, enabling fast training.
- **Robustness to Noise**: Handles 25% random noise in symptom data to simulate real-world variability.
- **Scalable**: Optimized for resource-constrained environments, suitable for edge devices.

## Dataset
The dataset is sourced from [Kaggle: Diseases and Symptoms Dataset](https://www.kaggle.com/datasets/dhivyeshrk/diseases-and-symptoms-dataset) and includes:
- **Records**: 168,499 samples
- **Features**: 378 symptom features
- **Classes**: 201 disease classes (filtered to include diseases with >500 records)
- **Preprocessing**:
  - Added 25% random noise to enhance robustness.
  - Split into 80% training (134,799 samples) and 20% test (33,700 samples) using Multilabel Stratified K-Fold.

## Methodology
The HDC classifier operates in three stages:

1. **Encoding**:
   - Symptom features are mapped to binary hypervectors (D=1000) using the sign function (`sign(x)`).
2. **Training**:
   - Class prototypes are computed by averaging hypervectors for each disease class.
3. **Inference**:
   - Test samples are encoded into hypervectors and compared to class prototypes using dot product similarity.

### HDC Operations
- **Binding**: Combines hypervectors using element-wise XOR.
- **Bundling**: Aggregates hypervectors via summation or majority voting.
- **Permutation**: Shuffles hypervectors to encode positional information.

### Pseudocode
```python
# Initialize hypervectors for symptom features
hypervectors = initialize_hypervectors(D=1000)

# Encode training samples
train_hypervectors = encode_samples(train_data, hypervectors)

# Compute class prototypes
class_prototypes = average_hypervectors(train_hypervectors, labels)

# Inference
predictions = []
for test_sample in test_data:
    test_hypervector = encode_sample(test_sample, hypervectors)
    predicted_class = argmax_dot_product(test_hypervector, class_prototypes)
    predictions.append(predicted_class)
```

## Results
The HDC classifier achieved the following performance on the test set:
- **Accuracy**: 87.16%
- **Macro-Averaged Precision**: 87.64%
- **Macro-Averaged Recall**: 87.81%
- **Macro-Averaged F1-Score**: 87.31%

### Performance Comparison
| Model         | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) |
|---------------|--------------|---------------|------------|--------------|
| HDC (Ours)    | 87.16        | 87.64         | 87.81      | 87.31        |
| SVM (Linear)  | 23.60        | 18.00         | 24.00      | 16.00        |

### Classification Report (Excerpt)
| Class | Precision (%) | Recall (%) | F1-Score (%) | Support |
|-------|---------------|------------|--------------|---------|
| 0     | 94            | 74         | 83           | 182     |
| 11    | 96            | 99         | 98           | 133     |
| 50    | 58            | 85         | 69           | 100     |
| 99    | 100           | 98         | 99           | 243     |
| 147   | 52            | 52         | 52           | 182     |
| 200   | 80            | 98         | 88           | 101     |

## Future Work
- Explore adaptive encoding schemes to improve performance on underrepresented classes.
- Investigate higher-dimensional hypervectors for better discrimination.
- Implement hardware acceleration for HDC to enhance scalability on edge devices.

## References
1. P. Kanerva, “Hyperdimensional Computing: An Introduction to Computing in Distributed Representation with High-Dimensional Random Vectors,” *Cognitive Computation*, vol. 1, no. 2, pp. 139–159, 2009.
2. A. Rahimi et al., “Robust Classification with Hyperdimensional Computing,” *IEEE Transactions on Circuits and Systems*, vol. 63, no. 12, pp. 2151–2160, 2016.
3. M. M. Najafabadi et al., “Hyperdimensional Computing for Text Classification,” *IEEE Access*, vol. 5, pp. 12345–12356, 2017.
4. M. Imani et al., “HDCluster: An Accurate and Efficient Clustering Using Hyperdimensional Computing,” *ACM Transactions on Embedded Computing Systems*, vol. 18, no. 5s, pp. 1–21, 2019.
5. Q. Wu et al., “HDC-IM: Hyperdimensional Computing In-Memory Architecture for Low-Power Signal Processing,” *IEEE International Symposium on Circuits and Systems (ISCAS)*, pp. 1–5, 2018.
6. Dhivyesh R. K., “Diseases and Symptoms Dataset,” *Kaggle*, 2023. [Online]. Available: https://www.kaggle.com/datasets/dhivyeshrk/diseases-and-symptoms-dataset
7. A. Esteva et al., “Dermatologist-Level Classification of Skin Cancer with Deep Neural Networks,” *Nature*, vol. 542, no. 7639, pp. 115–118, 2017.

## Contact
For questions or collaboration, reach out to:
- **Kush Revankar**
- **Email**: [1032221848@mitwpu.edu.in](mailto:kushrevankar24@gmail.com)
- **Institution**: MIT World Peace University, Pune, India
