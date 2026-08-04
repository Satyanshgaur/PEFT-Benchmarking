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

Rather than proposing a new fine-tuning algorithm, this project aims to produce a rigorous empirical evaluation of existing PEFT methods under standardized conditions. The final outcome will be a reproducible benchmarking framework and a practical guide that helps researchers and practitioners choose the most suitable parameter-efficient fine-tuning method based on available computational resources, memory constraints, and desired predictive performance.
