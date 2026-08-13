# Reproducibility and Benchmarking of Parameter-Efficient Fine-Tuning Methods for Transformer Models

> A reproducible empirical benchmark of Full Fine-Tuning, LoRA, AdaLoRA, Prefix Tuning, and IA³ across BERT-base and DistilBERT on three GLUE classification tasks.

While numerous PEFT methods have been proposed to reduce the cost of fine-tuning large transformer models, existing evaluations are often conducted under different experimental settings, making direct comparison difficult. This project aims to provide a **controlled, reproducible benchmark** where all methods are evaluated under identical conditions, enabling practical recommendations based on both predictive performance and computational efficiency.

The central question is:

> **How should practitioners choose among parameter-efficient fine-tuning methods under different computational constraints?**

This is a **practical benchmark, not a new PEFT method or a formal research study**. The goal is to provide transparent measurements and useful observations rather than claim state-of-the-art results.

---

# Table of Contents

- [1. Project Objectives](#1-project-objectives)
- [2. Models Evaluated](#2-models-evaluated)
- [3. Algorithms Evaluated](#3-algorithms-evaluated)
- [4. Datasets](#4-datasets)
- [5. Experimental Setup](#5-experimental-setup)
- [6. Measured Outcomes](#6-measured-outcomes)
- [7. Benchmark Results & Summary Table](#7-benchmark-results-summary-table)
- [8. What Do the Results Suggest?](#8-what-do-the-results-suggest)
  - [8.1 There is no universal PEFT winner](#81-there-is-no-universal-peft-winner)
  - [8.2 LoRA provides a strong practical trade-off on SST-2](#82-lora-provides-a-strong-practical-trade-off-on-sst-2)
  - [8.3 Extreme parameter efficiency comes with trade-offs](#83-extreme-parameter-efficiency-comes-with-trade-offs)
  - [8.4 MRPC exposes a failure mode of the low-parameter methods](#84-mrpc-exposes-a-failure-mode-of-the-low-parameter-methods)
  - [8.5 RTE shows another task-dependent degradation](#85-rte-shows-another-task-dependent-degradation)
- [9. An Interesting AdaLoRA Observation](#9-an-interesting-adalora-observation)
- [10. Prefix Tuning and DistilBERT](#10-prefix-tuning-and-distilbert)
- [11. What Should a Practitioner Take Away?](#11-what-should-a-practitioner-take-away)
- [12. Limitations](#12-limitations)
- [13. Reproducibility](#13-reproducibility)
- [14. Conclusion](#14-conclusion)
  - [Project Artifacts](#project-artifacts)

---

<a id="1-project-objectives"></a>
# 1. Project Objectives

- Reproduce influential PEFT algorithms using their official or widely adopted implementations.
- Benchmark each method under an identical experimental setup.
- Compare predictive performance, computational efficiency, and resource requirements.
- Provide evidence-based recommendations for selecting a PEFT method under different hardware constraints.
- Release a fully reproducible benchmarking pipeline.

---

<a id="2-models-evaluated"></a>
# 2. Models Evaluated

The following pretrained transformer models will be fine-tuned:

| Model | Purpose |
|--------|---------|
| BERT-base | Standard encoder baseline |
| DistilBERT | Lightweight distilled transformer |

---

<a id="3-algorithms-evaluated"></a>
# 3. Algorithms Evaluated

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

<a id="4-datasets"></a>
# 4. Datasets

The benchmark will use standard NLP datasets that cover multiple task types.

| Dataset | Task | Primary Metric |
|---------|------|----------------|
| SST-2 | Sentiment Classification | Accuracy |
| MRPC | Paraphrase Detection | Accuracy, F1 |
| RTE | Textual Entailment | Accuracy |

---

<a id="5-experimental-setup"></a>
# 5. Experimental Setup

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

<a id="6-measured-outcomes"></a>
# 6. Measured Outcomes

Following records are measured: 

### Predictive performance

- Accuracy
- Precision
- Recall
- F1
- Confusion matrices
- Calibration metrics

### Training efficiency

- Total training time
- Epoch time
- Training throughput
- Optimization steps

### Memory efficiency

- Peak GPU memory
- Average GPU memory
- CPU RAM usage
- GPU utilization

### Parameter efficiency

- Trainable parameter count
- Percentage of model parameters updated
- Fine-tuned checkpoint size

### Inference

- Inference latency
- Throughput

### Reproducibility

Every run records:

- random seed
- experiment ID
- Git commit
- Python/PyTorch/Transformers versions
- CUDA/GPU information
- tokenizer information
- dataset fingerprint
- experiment configuration

---

<a id="7-benchmark-results-summary-table"></a>
# 7. Benchmark Results & Summary Table

Below are the aggregated empirical findings across **BERT-base** and **DistilBERT** on GLUE classification tasks (**SST-2**, **MRPC**, **RTE**) averaged over 3 random seeds (`42`, `43`, `44`).

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

<a id="8-what-do-the-results-suggest"></a>
# 8. What Do the Results Suggest?

<a id="81-there-is-no-universal-peft-winner"></a>
## 8.1 There is no universal PEFT winner

The benchmark does not produce one method that dominates every metric.

Instead, the methods occupy different points in an accuracy–efficiency trade-off.

For example, on BERT/SST-2:

- Full FT gives the highest accuracy: **92.43%**
- Prefix Tuning reaches **90.25%** with much lower VRAM
- LoRA reaches **89.56%** with only **0.27%** of parameters trainable
- IA³ uses only **0.07%** trainable parameters, but accuracy falls to **80.39%**

This illustrates the main practical point of the benchmark:

> **Parameter efficiency, memory efficiency, training speed, and predictive performance are related, but they are not the same objective.**

---

<a id="82-lora-provides-a-strong-practical-trade-off-on-sst-2"></a>
## 8.2 LoRA provides a strong practical trade-off on SST-2

LoRA is particularly competitive on SST-2.

For BERT:

| | Full FT | LoRA |
|---|---:|---:|
| Accuracy | 92.43% | 89.56% |
| Peak VRAM | 2580.6 MB | 1129.6 MB |
| Training time | 1567.8 s | 1166.9 s |
| Trainable parameters | 109.48M | 296.5K |
| Checkpoint | 4.18 GB | 14.49 MB |

LoRA gives up around three percentage points of accuracy while substantially reducing memory, trainable parameters, and checkpoint footprint.

On DistilBERT/SST-2, the trade-off is similarly favorable:

> **87.35% accuracy with 781 MB peak VRAM and a 31 MB checkpoint**, compared with 89.98% accuracy and 1.46 GB peak VRAM for Full FT.

Under this benchmark configuration, LoRA is therefore a strong general-purpose choice when the practitioner wants to reduce resource usage without giving up most of the predictive performance.

---

<a id="83-extreme-parameter-efficiency-comes-with-trade-offs"></a>
## 8.3 Extreme parameter efficiency comes with trade-offs

IA³ is the most parameter-efficient method in the BERT experiments:

> **76,034 trainable parameters — only 0.07% of BERT-base.**

Its checkpoint is also only about **6 MB**.

However, on BERT/SST-2 it reaches 80.39%, substantially below Full FT, LoRA, and Prefix Tuning.

This demonstrates why simply minimizing the number of trainable parameters is not enough to choose an adaptation method.

---

<a id="84-mrpc-exposes-a-failure-mode-of-the-low-parameter-methods"></a>
## 8.4 MRPC exposes a failure mode of the low-parameter methods

On MRPC, LoRA, AdaLoRA, and IA³ all achieve exactly:

> **68.38 ± 0.00%**

This is not a meaningful learned-performance score by itself.

The MRPC validation set contains 408 examples:

- 279 positive examples
- 129 negative examples

Therefore, predicting the positive class for every example gives:

\[
\frac{279}{408}=68.38\%.
\]

Inspection of the saved predictions confirms that these methods predicted the positive class for all 408 validation examples.

The important conclusion is therefore:

> **Under the standardized training configuration, LoRA, AdaLoRA, and IA³ failed to learn a useful decision boundary on MRPC and converged to the validation-set majority-class prediction.**

This should not be interpreted as evidence that these PEFT methods are inherently incapable of solving MRPC. The benchmark does not perform task-specific hyperparameter optimization.

---

<a id="85-rte-shows-another-task-dependent-degradation"></a>
## 8.5 RTE shows another task-dependent degradation

The RTE results show a similar, although less extreme, pattern.

For BERT:

- Full FT: **65.70%**
- Prefix Tuning: **51.99%**
- LoRA: **47.53%**
- AdaLoRA: **45.85%**
- IA³: **47.05%**

For DistilBERT:

- Full FT: **58.60%**
- LoRA: **53.43%**
- AdaLoRA: **53.19%**
- IA³: **52.23%**

This suggests that the practical value of PEFT can be strongly task-dependent, particularly for smaller or more difficult classification datasets.

Again, these numbers describe performance **under the fixed benchmark configuration**, rather than the best achievable performance after hyperparameter tuning.

---

<a id="9-an-interesting-adalora-observation"></a>
# 9. An Interesting AdaLoRA Observation

AdaLoRA produced an unexpectedly low **66.36 ± 1.84%** on BERT/SST-2.

Importantly, the training logs show that the model was actually learning:

- Training loss decreased from approximately **1.90 to 0.61**
- Validation loss decreased from approximately **0.66 to 0.62**
- The behavior was consistent across all three seeds

The saved confusion matrices also show that this was not simply an all-one-class prediction. Instead, the model showed a strong class imbalance in its predictions, with substantially higher recall for the positive class than for the negative class.

This makes the result useful diagnostically: **loss reduction did not translate into balanced classification performance.**

There is also an implementation detail worth documenting. The current benchmark uses the Hugging Face PEFT AdaLoRA configuration within the benchmark training loop, and AdaLoRA's adaptive rank allocation depends on updating its internal allocation state during training. Therefore, the current results should be treated as results for the **implemented benchmark configuration**, not as a definitive evaluation of optimally configured AdaLoRA.

This is an important limitation rather than something to hide.

---

<a id="10-prefix-tuning-and-distilbert"></a>
# 10. Prefix Tuning and DistilBERT

Nine targeted runs were not executed:

> DistilBERT + Prefix Tuning across SST-2, MRPC, and RTE for seeds 42–44.

The reason is an architectural compatibility limitation in the current Hugging Face PEFT/Transformers setup: Prefix Tuning requires past-key-value support that the DistilBERT implementation used in this benchmark does not provide.

Therefore, these configurations are reported as **N/A / skipped**, rather than assigning them a fabricated score.

Prefix Tuning was successfully evaluated with BERT-base.

---

<a id="11-what-should-a-practitioner-take-away"></a>
# 11. What Should a Practitioner Take Away?

The benchmark does not support a single ranking such as "LoRA > AdaLoRA > IA³."

A more useful interpretation is:

| Priority | What the benchmark suggests |
|---|---|
| Maximum predictive performance | Full Fine-Tuning is the strongest baseline |
| Strong accuracy with substantially lower resources | LoRA is particularly competitive |
| Competitive BERT/SST-2 performance with lower VRAM | Prefix Tuning is interesting |
| Extremely small adapter/checkpoint | IA³ is the most aggressive option |
| Adaptive low-rank allocation | AdaLoRA requires careful interpretation under this standardized setup |
| Very small/difficult datasets | Validate PEFT performance rather than assuming parameter efficiency will transfer directly |

The practical choice therefore depends on the constraint:

> **If accuracy is the primary objective and resources permit it, Full Fine-Tuning remains a strong baseline. If memory, storage, or trainable parameter count matter substantially, LoRA provides a particularly attractive compromise in these experiments. More aggressive parameter reduction can produce much smaller adapters, but the benchmark shows that this can come with substantial task-dependent performance degradation.**

---

<a id="12-limitations"></a>
# 12. Limitations

This benchmark should be interpreted as a **controlled practical comparison**, not a comprehensive evaluation of PEFT.

### Fixed hyperparameters

The methods were intentionally evaluated under a common training configuration. Individual methods may perform substantially better after method-specific hyperparameter tuning.

### Small task set

Only three GLUE classification tasks were evaluated. The results should not be generalized to all NLP workloads.

### Single hardware platform

Resource measurements were collected on an RTX 3050 Laptop GPU. Training times and memory behavior will differ across hardware.

### Approximate FLOPs

The benchmark's analytical FLOPs values are estimates rather than hardware-level measurements of actual executed FLOPs.

### AdaLoRA configuration

The current AdaLoRA implementation requires careful handling of its adaptive rank allocation mechanism. The reported results therefore describe the benchmark implementation and configuration rather than an optimized AdaLoRA study.

### Prefix Tuning compatibility

Prefix Tuning was not evaluated on DistilBERT because of the attention/past-key-value compatibility limitation described above.

### No hyperparameter search

The benchmark asks how methods compare under a standardized protocol. It does not answer how well each method performs after independent optimization.

---

<a id="13-reproducibility"></a>
# 13. Reproducibility

Each experiment is stored independently with:

```text
results/
├── manifest.json
├── pareto.json
├── statistical_tests.json
└── <model>/
    └── <dataset>/
        └── <method>/
            └── seed<seed>/
                ├── config/
                │   ├── global.yaml
                │   └── method.yaml
                ├── run.json
                ├── metrics.json
                ├── adapter_config.json
                ├── predictions.csv
                ├── misclassified.csv
                ├── train_log.csv
                └── checkpoint/
```

The repository also contains a Jupyter analysis notebook for aggregating results and generating visualizations.

The benchmark records the software environment, hardware information, random seed, Git revision, tokenizer information, and experiment configuration for each run.

---

<a id="14-conclusion"></a>
# 14. Conclusion

This benchmark demonstrates that PEFT is not simply a question of reducing the number of trainable parameters.

Across BERT-base and DistilBERT, the methods produced substantially different combinations of:

- predictive performance
- GPU memory consumption
- training time
- trainable parameter count
- checkpoint size

On SST-2, LoRA and Prefix Tuning retained much of Full Fine-Tuning's predictive performance while requiring substantially fewer resources. IA³ achieved extreme parameter and checkpoint efficiency, but with a larger accuracy penalty.

On MRPC and RTE, several PEFT methods degraded substantially under the fixed training configuration, with LoRA, AdaLoRA, and IA³ reaching the MRPC majority-class baseline.

The main practical lesson is therefore simple:

> **There is no universally best PEFT method. The appropriate choice depends on the balance between predictive performance and the computational constraints of the deployment environment.**

The value of this benchmark is not in proposing a new algorithm, but in making these trade-offs **measurable, reproducible, and easy to inspect**.

---

<a id="project-artifacts"></a>
## Project Artifacts

- **Source code:** GitHub repository
- **Benchmark analysis:** `notebooks/analysis.ipynb`
- **Raw experiment artifacts:** `results/`
- **Machine-readable metrics:** `metrics.json` files
- **Predictions and failure analysis:** `predictions.csv` and `misclassified.csv`
