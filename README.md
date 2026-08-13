# Reproducibility and Benchmarking of Parameter-Efficient Fine-Tuning Methods for Transformer Models

## Central Research Question

> **How should practitioners choose among parameter-efficient fine-tuning (PEFT) methods under different computational constraints?**

While numerous PEFT methods have been proposed to reduce the cost of fine-tuning large transformer models, existing evaluations are often conducted under different experimental settings, making direct comparison difficult. This project aims to provide a **controlled, reproducible benchmark** where all methods are evaluated under identical conditions, enabling practical recommendations based on both predictive performance and computational efficiency.

---

# Project Objectives

- Reproduce influential PEFT algorithms using their official or widely adopted implementations.
- Benchmark each method under an identical experimental setup.
- Compare predictive performance, computational efficiency, and resource requirements.
- Provide evidence-based recommendations for selecting a PEFT method under different hardware constraints.
- Release a fully reproducible benchmarking pipeline.

---

# Models Evaluated

The following pretrained transformer models will be fine-tuned:

| Model | Purpose |
|--------|---------|
| BERT-base | Standard encoder baseline |
| DistilBERT | Lightweight distilled transformer |
| RoBERTa-base *(optional extension)* | Stronger encoder baseline |

---

# Algorithms Evaluated

## Baseline

### Full Fine-Tuning

- All model parameters are updated during training.
- Serves as the upper-bound baseline for performance.

---

## Parameter-Efficient Fine-Tuning Methods

### LoRA
**Paper:** *LoRA: Low-Rank Adaptation of Large Language Models*

Key idea:
- Inject trainable low-rank matrices into attention layers while freezing the original model weights.

---

### AdaLoRA
**Paper:** *AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning*

Key idea:
- Dynamically reallocates adaptation rank during training to improve parameter efficiency.

---

### Prefix Tuning
**Paper:** *Prefix-Tuning: Optimizing Continuous Prompts for Generation*

Key idea:
- Learns trainable prefix vectors while keeping pretrained weights frozen.

---

### IA³
**Paper:** *Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning*

Key idea:
- Learns lightweight multiplicative scaling vectors applied to transformer activations.

---

# Datasets

The benchmark will use standard NLP datasets that cover multiple task types.

| Dataset | Task | Primary Metric |
|---------|------|----------------|
| SST-2 | Sentiment Classification | Accuracy |
| MRPC | Paraphrase Detection | Accuracy, F1 |
| RTE | Textual Entailment | Accuracy |
| AG News *(optional extension)* | News Topic Classification | Accuracy |

---

# Experimental Setup

## Framework

- PyTorch
- Hugging Face Transformers
- Hugging Face PEFT
- Hugging Face Datasets
- Accelerate
- Evaluate

---

## Hardware

Example benchmark hardware:

- NVIDIA RTX 3050 Laptop GPU (6 GB VRAM)
- Intel i5-13420H
- 16 GB RAM

---

## Controlled Variables

All methods will be trained under identical settings:

- Same pretrained model
- Same dataset splits
- Same optimizer
- Same scheduler
- Same learning rate
- Same number of epochs
- Same batch size
- Same random seed(s)
- Same evaluation pipeline

This ensures differences in performance arise solely from the fine-tuning method.

---

# Benchmarks

Each combination of:

- Model
- Dataset
- Fine-tuning method

constitutes one benchmark experiment.

Example benchmark matrix:

| Model | Dataset | Full FT | LoRA | AdaLoRA | Prefix | IA³ |
|--------|----------|---------|-------|----------|---------|------|
| BERT | SST-2 | ✓ | ✓ | ✓ | ✓ | ✓ |
| BERT | MRPC | ✓ | ✓ | ✓ | ✓ | ✓ |
| BERT | RTE | ✓ | ✓ | ✓ | ✓ | ✓ |
| DistilBERT | SST-2 | ✓ | ✓ | ✓ | ✓ | ✓ |
| DistilBERT | MRPC | ✓ | ✓ | ✓ | ✓ | ✓ |
| DistilBERT | RTE | ✓ | ✓ | ✓ | ✓ | ✓ |

---

# Measured Outcomes

The benchmark evaluates multiple dimensions rather than focusing solely on predictive accuracy.

---

## 1. Predictive Performance

Measures task effectiveness.

| Metric | Description |
|----------|-------------|
| Accuracy | Classification accuracy |
| Precision | Positive prediction quality |
| Recall | Positive class coverage |
| F1 Score | Harmonic mean of precision and recall |

---

## 2. Training Efficiency

Measures computational cost during optimization.

| Metric | Description |
|----------|-------------|
| Training Time | Total fine-tuning duration |
| Epoch Time | Average time per epoch |
| Samples per Second | Training throughput |
| Steps per Second | Optimization throughput |

---

## 3. Memory Efficiency

Measures hardware resource consumption.

| Metric | Description |
|----------|-------------|
| Peak GPU Memory | Maximum VRAM usage |
| Average GPU Memory | Mean VRAM consumption |
| CPU RAM Usage | Host memory consumption |

---

## 4. Parameter Efficiency

Measures adaptation complexity.

| Metric | Description |
|----------|-------------|
| Trainable Parameters | Number of updated parameters |
| Percentage Trainable | Trainable parameters relative to total model size |
| Checkpoint Size | Storage required for learned weights |

---

## 5. Inference Efficiency

Measures deployment performance.

| Metric | Description |
|----------|-------------|
| Inference Latency | Time per prediction |
| Throughput | Samples processed per second |

---

## 6. Optimization Behaviour

Measures learning dynamics.

| Metric | Description |
|----------|-------------|
| Training Loss | Optimization progress |
| Validation Loss | Generalization performance |
| Convergence Epoch | Epoch where performance stabilizes |

---

## 7. Stability

Measures reproducibility.

| Metric | Description |
|----------|-------------|
| Mean Accuracy | Average across multiple random seeds |
| Standard Deviation | Performance variability |
| Confidence Interval | Statistical reliability |

---

# Expected Deliverables

- Reproducible PyTorch implementation
- Unified benchmarking pipeline
- Benchmark results for all PEFT methods
- Performance comparison tables
- GPU memory comparison
- Training speed comparison
- Inference latency comparison
- Parameter efficiency analysis
- Reproducible experimental configuration
- Technical report documenting methodology and findings

---

# Expected Outcome

Rather than proposing a new fine-tuning algorithm, this project aims to produce a reproducible empirical evaluation of existing PEFT methods under standardized conditions. The final outcome is a reproducible benchmarking framework and a practical guide that helps researchers and practitioners choose the most suitable parameter-efficient fine-tuning method based on available computational resources, memory constraints, and desired predictive performance.

---

# Benchmark Results & Key Findings

Below are the aggregated empirical findings across **BERT-base** and **DistilBERT** on GLUE classification tasks (**SST-2**, **MRPC**, **RTE**) averaged over 3 random seeds (`42`, `43`, `44`).

## Empirical Results Summary Table

| Model | Dataset | Method | Accuracy ($\mu \pm \sigma$) | Peak VRAM | Training Time | Trainable Params (% Total) | Checkpoint Size |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **DistilBERT** | **SST-2** | Full FT | 0.8998 ± 0.0026 | 1459.8 MB | 782.5 s | 66,955,010 (100.00%) | 2557.44 MB |
| **DistilBERT** | **SST-2** | LoRA | 0.8735 ± 0.0083 | 781.4 MB | 412.9 s | 739,586 (1.09%) | 31.14 MB |
| **DistilBERT** | **SST-2** | AdaLoRA | 0.8352 ± 0.0024 | 814.2 MB | 647.9 s | 813,458 (1.20%) | 34.18 MB |
| **DistilBERT** | **SST-2** | Prefix Tuning | *N/A (Skipped)* | *N/A* | *N/A* | *N/A* | *N/A* |
| **DistilBERT** | **SST-2** | IA³ | 0.8356 ± 0.0013 | 979.4 MB | 587.4 s | 619,778 (0.92%) | 26.74 MB |
| **DistilBERT** | **MRPC** | Full FT | 0.8489 ± 0.0028 | 1843.2 MB | 86.2 s | 66,955,010 (100.00%) | 2557.23 MB |
| **DistilBERT** | **MRPC** | LoRA | 0.6838 ± 0.0000 | 1115.9 MB | 64.8 s | 739,586 (1.09%) | 31.12 MB |
| **DistilBERT** | **MRPC** | AdaLoRA | 0.6838 ± 0.0000 | 1118.6 MB | 66.8 s | 813,458 (1.20%) | 33.97 MB |
| **DistilBERT** | **MRPC** | Prefix Tuning | *N/A (Skipped)* | *N/A* | *N/A* | *N/A* | *N/A* |
| **DistilBERT** | **MRPC** | IA³ | 0.6838 ± 0.0000 | 1374.1 MB | 64.5 s | 619,778 (0.92%) | 26.53 MB |
| **DistilBERT** | **RTE** | Full FT | 0.5860 ± 0.0290 | 2090.7 MB | 81.8 s | 66,955,010 (100.00%) | 2557.23 MB |
| **DistilBERT** | **RTE** | LoRA | 0.5343 ± 0.0201 | 1305.0 MB | 61.2 s | 739,586 (1.09%) | 31.11 MB |
| **DistilBERT** | **RTE** | AdaLoRA | 0.5319 ± 0.0240 | 1307.8 MB | 62.9 s | 813,458 (1.20%) | 33.96 MB |
| **DistilBERT** | **RTE** | Prefix Tuning | *N/A (Skipped)* | *N/A* | *N/A* | *N/A* | *N/A* |
| **DistilBERT** | **RTE** | IA³ | 0.5223 ± 0.0240 | 1625.3 MB | 61.2 s | 619,778 (0.92%) | 26.52 MB |
| **BERT-base** | **SST-2** | Full FT | 0.9243 ± 0.0041 | 2580.6 MB | 1567.8 s | 109,483,778 (100.00%) | 4180.07 MB |
| **BERT-base** | **SST-2** | LoRA | 0.8956 ± 0.0040 | 1129.6 MB | 1166.9 s | 296,450 (0.27%) | 14.49 MB |
| **BERT-base** | **SST-2** | AdaLoRA | 0.6636 ± 0.0184 | 956.0 MB | 1061.3 s | 444,194 (0.40%) | 20.41 MB |
| **BERT-base** | **SST-2** | Prefix Tuning | 0.9025 ± 0.0020 | 1055.1 MB | 1125.3 s | 14,781,698 (11.90%) | 347.25 MB |
| **BERT-base** | **SST-2** | IA³ | 0.8039 ± 0.0064 | 1143.2 MB | 1022.7 s | 76,034 (0.07%) | 6.34 MB |
| **BERT-base** | **MRPC** | Full FT | 0.8676 ± 0.0088 | 2321.0 MB | 225.9 s | 109,483,778 (100.00%) | 4179.87 MB |
| **BERT-base** | **MRPC** | LoRA | 0.6838 ± 0.0000 | 1270.7 MB | 140.6 s | 296,450 (0.27%) | 14.29 MB |
| **BERT-base** | **MRPC** | AdaLoRA | 0.6838 ± 0.0000 | 1273.8 MB | 126.3 s | 444,194 (0.40%) | 19.99 MB |
| **BERT-base** | **MRPC** | Prefix Tuning | 0.7312 ± 0.0102 | 1314.1 MB | 123.3 s | 14,781,698 (11.90%) | 346.83 MB |
| **BERT-base** | **MRPC** | IA³ | 0.6838 ± 0.0000 | 1564.4 MB | 121.3 s | 76,034 (0.07%) | 5.92 MB |
| **BERT-base** | **RTE** | Full FT | 0.6570 ± 0.0108 | 2546.0 MB | 171.6 s | 109,483,778 (100.00%) | 4179.86 MB |
| **BERT-base** | **RTE** | LoRA | 0.4753 ± 0.0326 | 1454.2 MB | 125.1 s | 296,450 (0.27%) | 14.28 MB |
| **BERT-base** | **RTE** | AdaLoRA | 0.4585 ± 0.0108 | 1457.5 MB | 131.3 s | 444,194 (0.40%) | 19.98 MB |
| **BERT-base** | **RTE** | Prefix Tuning | 0.5199 ± 0.0253 | 1468.2 MB | 127.4 s | 14,781,698 (11.90%) | 346.82 MB |
| **BERT-base** | **RTE** | IA³ | 0.4705 ± 0.0398 | 1805.1 MB | 129.3 s | 76,034 (0.07%) | 5.91 MB |

---

## Technical Insights & Empirical Findings

### 1. Unexecuted Configurations (9 Runs)
Out of the 90 targeted run executions ($2 \text{ models} \times 3 \text{ datasets} \times 5 \text{ methods} \times 3 \text{ seeds}$), 81 runs were completed and **9 runs were skipped**:
- **Skipped**: `DistilBERT` + `Prefix Tuning` across `SST-2`, `MRPC`, and `RTE` (seeds 42, 43, 44).
- **Reason**: In Hugging Face PEFT, Prefix Tuning requires injecting past key-value (`past_key_values`) caches into multi-head attention. `DistilBertModel` is a lightweight distilled encoder architecture that does not implement KV caching or past key-values.

### 2. Majority Class Collapse on MRPC (Exact 68.38% Accuracy)
On MRPC, low-rank adapters (**LoRA**, **AdaLoRA**, **IA³**) achieved an exact accuracy of **$0.6838 \pm 0.0000$** across all random seeds. 
- **Cause**: The MRPC validation set consists of 408 samples, of which 279 belong to Class 1 ("equivalent paraphrase"). $\frac{279}{408} = 0.6838235... \approx \mathbf{68.38\%}$.
- **Mechanism**: Under fixed default hyperparameters without task-specific learning rate search, updating $<1\%$ of model parameters resulted in the optimizer collapsing to predict Class 1 for 100% of validation samples (confusion matrix: $\begin{bmatrix} 0 & 129 \\ 0 & 279 \end{bmatrix}$). Full Fine-Tuning (updating 100% parameters) escaped this local minimum to reach **84.89%** (DistilBERT) and **86.76%** (BERT-base).

### 3. AdaLoRA SVD Rank Allocation Dynamics
AdaLoRA on BERT-base / SST-2 converged smoothly ($\text{Loss: } 1.90 \rightarrow 0.61$, $\text{Val Loss: } 0.66 \rightarrow 0.62$) to $66.36\% \pm 1.84\%$ across all seeds.
- **Diagnostic Finding**: In Hugging Face PEFT, AdaLoRA requires calling `model.base_model.update_and_allocate(global_step)` during step execution to dynamically prune and allocate rank budgets. When using standard `transformers.Trainer` without a custom `AdaLoraStepCallback`, `update_and_allocate` is not invoked automatically, causing AdaLoRA to operate as an unallocated static SVD adapter.

