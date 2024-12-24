# **Spred with SGD for Structured Sparsity in CNNs**

This repository contains the implementation and experimental results of applying the **Spred algorithm** to induce **structured sparsity** in Convolutional Neural Networks (CNNs). The work is an extension of the original research paper, focusing on improving computational efficiency while maintaining model performance through structured sparsity.

---

## **Overview**

Modern deep learning models often involve overparameterized architectures, leading to high computational and memory costs. This project tackles this issue by applying **structured sparsity** (e.g., filter or channel-level pruning) to CNNs using the **Spred algorithm**, combined with **Stochastic Gradient Descent (SGD)** and other optimizers. Structured sparsity offers practical benefits over unstructured sparsity, such as hardware compatibility and significant speed-ups during inference.

---

## **Key Features**
1. **Structured Sparsity**:
   - Induces sparsity at the **filter/channel level**, enabling efficient deployment on hardware accelerators.
2. **Reparameterization**:
   - Implements the Spred algorithm to reparameterize sparse weights as products of auxiliary variables, $U$ and $W$, for smooth optimization.
3. **Optimizers**:
   - Experiments conducted with **SGD**, **Adam**, **AdamW**, and **Adagrad**.
4. **Comparison**:
   - Detailed comparison of **structured vs. unstructured sparsity** across different CNN architectures and datasets.
5. **Evaluation Metrics**:
   - Model accuracy, training time, inference speed, and memory efficiency analyzed at various sparsity levels (20%, 40%, 60%, 80%).

---

## **Project Structure**

```plaintext
spred_structured_sparsity_project/
├── data/
│   ├── cifar10/           # CIFAR-10 dataset
│   └── cifar100/          # CIFAR-100 dataset
├── models/
│   ├── base_cnn.py        # Basic CNN architecture (ResNet18)
│   └── spred_structured.py # Spred implementation for structured sparsity
├── utils/
│   ├── data_loader.py     # Dataset loading and preprocessing
│   ├── loss_functions.py  # Loss functions with sparsity regularization
│   ├── metrics.py         # Evaluation metrics
│   ├── sparsity_utils.py  # Helper functions for sparsity operations
├── scripts/
│   ├── train.py           # Main training script
│   ├── evaluate.py        # Model evaluation
│   ├── sparsity_experiments.py # Sparsity level comparisons
│   └── visualize_results.py # Plot and analyze results
├── results/
│   ├── logs/              # Training logs
│   └── figures/           # Plots and visualizations
├── requirements.txt       # Python dependencies
├── README.md              # Documentation
```

---

## **Implementation Details**

1. **Reparameterization**:
   - Sparse parameters ($V_s$) are decomposed as:
     $V_s = U \times W\$
   - Penalties are applied to $U$ and $W$ to promote sparsity, transforming non-differentiable $L_1$ penalties into smooth $L_2$ norm penalties:
   
     $Penalty = \lambda (\|U\|_2^2 + \|W\|_2^2 )$

2. **Structured Sparsity**:
   - Targets **entire filters or channels** in CNNs, reducing computational cost while preserving network structure.

3. **Optimizers**:
   - Experiments conducted with **SGD**, **Adam**, **AdamW**, and **Adagrad** to analyze the impact of optimizers on sparsity and model performance.

4. **Datasets**:
   - Used **CIFAR-10** and **CIFAR-100** for training and evaluation.

5. **Evaluation**:
   - Metrics analyzed:
     - Accuracy vs. sparsity levels.
     - Training and inference times.
     - Model size and memory usage.

---

## **Experimental Results**

- **Structured vs. Unstructured Sparsity**:
  - Structured sparsity retained better accuracy at higher sparsity levels (e.g., 80%) compared to unstructured sparsity.
  - Structured sparsity also reduced inference time by up to **25%** due to hardware alignment.

- **Optimizer Comparison**:
  - **Adagrad** consistently outperformed other optimizers, especially in sparse gradient settings, achieving the highest accuracy across datasets and sparsity levels.

- **Example Result Summary (CIFAR-100, Structured Sparsity)**:
  | **Optimizer** | **Sparsity Level** | **Accuracy (%)** | **Training Time (s)** | **Inference Time (s)** |
  |---------------|--------------------|------------------|-----------------------|------------------------|
  | SGD           | 20%               | 45.46            | 417.34               | 0.89                  |
  | Adam          | 40%               | 44.88            | 428.70               | 0.80                  |
  | Adagrad       | 60%               | 48.71            | 395.38               | 0.88                  |

---

#### Link to the paper [Spred paper](https://proceedings.mlr.press/v202/ziyin23a/ziyin23a.pdf)
