# Radiology Report Summarisation Using FLAN-T5: Full Fine-Tuning vs LoRA

**Author:** Tran Minh Quang Nguyen - 50124769

## Table of Contents

1. [Introduction](#1-introduction)
   - [1.1 Background and Problem](#11-background-and-problem)
   - [1.2 Objectives and Research Question](#12-objectives-and-research-question)
   - [1.3 Proposed Approach](#13-proposed-approach)
2. [Feasibility Review](#2-feasibility-review)
3. [Dataset and Preprocessing](#3-dataset-and-preprocessing)
   - [3.1 Dataset and Preprocessing](#31-dataset-and-preprocessing)
   - [3.2 Data Leakage Analysis](#32-data-leakage-analysis)
4. [Methodology](#4-methodology)
   - [4.1 FLAN-T5 and Fine-Tuning Strategies](#41-flan-t5-and-fine-tuning-strategies)
   - [4.2 Training Procedure](#42-training-procedure)
   - [4.3 Experimental Setup](#43-experimental-setup)
5. [Results and Evaluation](#5-results-and-evaluation)
   - [5.1 Evaluation Metrics](#51-evaluation-metrics)
   - [5.2 Summarisation Performance](#52-summarisation-performance)
   - [5.3 Computational Efficiency](#53-computational-efficiency)
   - [5.4 Results Discussion](#54-results-discussion)
6. [Clinical Safety Analysis](#6-clinical-safety-analysis)
   - [6.1 Qualitative Error Analysis](#61-qualitative-error-analysis)
   - [6.2 Failure Patterns and Root Causes](#62-failure-patterns-and-root-causes)
   - [6.3 Recommendations and Limitations](#63-recommendations-and-limitations)
7. [Reproducibility](#7-reproducibility)
   - [7.1 Environment and Dependencies](#71-environment-and-dependencies)
   - [7.2 Training and Evaluation Instructions](#72-training-and-evaluation-instructions)
8. [Conclusion](#8-conclusion)
9. [References](#9-references)
10. [AI Usage Disclosure](#10-ai-usage-disclosure)

## 1. Introduction

### 1.1 Background and Problem

Radiology reports contain essential diagnostic information but often use specialised terminology that patients may struggle to understand. Generating patient-friendly summaries can improve accessibility, but clinical accuracy remains critical: hallucinated findings, omitted information, or altered diagnostic certainty may lead to misinterpretation [7].

This project investigates **FLAN-T5 Base** for radiology report summarisation using the **BioLaySumm2025 LaymanRRG** dataset [1]. Building on previous T5-based approaches to radiology summarisation [6], it compares **Full Fine-Tuning (Full FT)** and **Low-Rank Adaptation (LoRA)** in terms of summarisation quality, computational efficiency, and clinical reliability.

### 1.2 Objectives and Research Question

**Research question:** Can LoRA achieve summarisation quality comparable to Full FT with substantially fewer trainable parameters, while maintaining clinical reliability?

The project aims to:

1. Compare **zero-shot, LoRA, and Full FT** using ROUGE-1, ROUGE-2, ROUGE-L, and ROUGE-Lsum.
2. Measure computational efficiency through trainable parameters, GPU memory usage, training time, and inference performance.
3. Investigate training–validation data overlap and its impact on evaluation reliability.
4. Analyse clinical safety through hallucinations, omissions, and readability.

### 1.3 Proposed Approach

The project uses **FLAN-T5 Base**, an instruction-tuned encoder-decoder Transformer [2, 3], with three configurations:

- **Zero-shot:** Pretrained model without task-specific training.
- **Full FT:** Updates all pretrained model parameters.
- **LoRA:** Trains lightweight low-rank adapters while keeping pretrained weights frozen [4].

All configurations follow a consistent preprocessing and evaluation pipeline. Summarisation quality is assessed using ROUGE [5], while computational profiling and qualitative clinical safety analysis provide complementary evidence beyond lexical similarity.

The overall workflow is illustrated in Figure 1.

<p align="center">
  <img src="figures/project_pipeline.png" alt="FLAN-T5 Summarisation Pipeline" width="100%">
</p>

<p align="center">
  <em>Figure 1. Overview of the FLAN-T5 radiology report summarisation pipeline, comparing zero-shot inference, Full FT, and LoRA.</em>
</p>

## 2. Feasibility Review

### Problem, User Need and Scope

Radiology reports contain specialised medical terminology that can be difficult for non-expert readers to understand. This project aims to generate layperson-friendly summaries while preserving clinically important findings.

The scope is text-to-text summarisation using the official BioLaySumm2025 LaymanRRG dataset. The project focuses on fine-tuning, quantitative evaluation, and clinical safety analysis, rather than image interpretation or clinical deployment.

### Measurable Acceptance Criteria

1. Implement a reproducible FLAN-T5 fine-tuning pipeline with a custom PyTorch training loop (without the Hugging Face Trainer API).
2. Evaluate pretrained and LoRA-fine-tuned models using ROUGE-1, ROUGE-2, and ROUGE-L, targeting an improvement in ROUGE-L over the pretrained baseline.
3. Report results on both the full validation set (10,000 records) and the unseen-report subset (5,494 records).
4. Analyse 3–5 validation examples for factual hallucinations, critical omissions, and readability.
5. Complete training within the available GPU budget, documenting runtime, peak GPU memory, and configuration.

### Technical Approach and Related Work

The BioLaySumm 2025 shared task establishes the radiology-to-layperson summarisation setting [1]. This project adopts FLAN-T5 Base, building on the text-to-text Transformer framework [2] and instruction fine-tuning [3].

Low-Rank Adaptation (LoRA) [4] will be used for parameter-efficient fine-tuning. A custom PyTorch training loop will be implemented without the Hugging Face Trainer API.

Model performance will be assessed using ROUGE [5], alongside qualitative analysis of clinical accuracy and readability.

### Preliminary Feasibility Evidence

The official BioLaySumm2025 LaymanRRG dataset [1] was successfully loaded and audited on Google Colab. It contains 150,454 training, 10,000 validation, and 10,537 test records.

A case-insensitive, whitespace-normalised exact-match audit identified **2,150 unique reports** shared between training and validation, affecting **4,506 validation records (45.06%)**. Furthermore, 12,154 unique training reports have multiple distinct target summaries. All test records lack reference summaries.

These findings highlight a potential evaluation bias from report overlap. The official splits will be preserved, and ROUGE will be reported on both the full validation set and the **5,494-record unseen-report subset**. GPU feasibility and model performance remain to be verified.

### Risks, Resources and Fallback

| Risk | Mitigation Strategy |
|---|---|
| **Data overlap** | Compare ROUGE on the full validation set against the unseen-report subset. |
| **Clinical accuracy** | Examine unsupported statements, critical omissions, and readability in qualitative cases. |
| **GPU budget** | Target a single Colab GPU with LoRA, batch size 4, and up to 3 epochs. Record peak VRAM and training time. Reduce batch size to 1 with gradient accumulation if memory is insufficient. |
| **Time constraints** | Prioritise a working baseline, reproducible fine-tuning, and evaluation before optional hyperparameter experiments. |

## 3. Dataset and Preprocessing

### 3.1 Dataset and Preprocessing

This project uses the **BioLaySumm2025 LaymanRRG** dataset [1], containing radiology reports and their corresponding patient-friendly summaries. The dataset provides four fields: `source`, `images_path`, `radiology_report`, and `layman_report`. Only the text fields are used, with `radiology_report` as the model input and `layman_report` as the target summary.

**Table 1. Official BioLaySumm2025 dataset splits.**

| Split | Records | Purpose |
|---|---:|---|
| Training | 150,454 | Model fine-tuning |
| Validation | 10,000 | Checkpoint selection and evaluation |
| Test | 10,537 | Unlabelled test set |
| **Total** | **170,991** | |

The official splits are preserved. Since test references are withheld, quantitative evaluation uses the validation set.

**Preprocessing.** Reports and summaries are tokenised using the pretrained FLAN-T5 tokenizer, with truncation to configured sequence lengths and dynamic padding within each batch. Target padding tokens are replaced with `-100` to exclude them from cross-entropy loss. A custom PyTorch `DataLoader` produces `input_ids`, `attention_mask`, and `labels`, with shuffled training batches and deterministic validation ordering. Both fine-tuning strategies use the same preprocessing pipeline.

### 3.2 Data Leakage Analysis

A case-insensitive, whitespace-normalised exact-match audit was conducted to detect report overlap between training and validation.

**Table 2. Training–validation overlap analysis.**

| Metric | Count |
|---|---:|
| Unique training reports | 93,505 |
| Unique validation reports | 7,599 |
| Unique reports shared between splits | 2,150 |
| Validation records with overlapping reports | 4,506 |
| Validation records with unseen reports | 5,494 |
| **Validation overlap rate** | **45.06%** |
| Training reports with multiple reference summaries | 12,154 |

**45.06% of validation records contain reports also present in training**, potentially inflating evaluation scores. Furthermore, 12,154 unique training reports have multiple distinct reference summaries, introducing ambiguity in the training targets.

![Training–Validation Dataset Overlap](figures/dataset_overlap.png)
<p align="center">
  <em>Figure 2. Distribution of overlapping (45.06%) and unseen (54.94%) validation reports.</em>
</p>


To assess the impact of overlap, ROUGE is evaluated on both the **full validation set (10,000 records)** and the **unseen-report subset (5,494 records)**. This comparison helps distinguish performance on previously encountered report text from generalisation to unseen reports.


## 4. Methodology

### 4.1 Model and Fine-Tuning Strategies

This project uses **FLAN-T5 Base** (`google/flan-t5-base`), a 247.58M-parameter instruction-tuned encoder-decoder Transformer [2, 3]. Three configurations are considered: **zero-shot**, **Full Fine-Tuning (Full FT)**, and **Low-Rank Adaptation (LoRA)**.

- **Zero-shot:** Uses the pretrained model without task-specific training.
- **Full FT:** Updates all 247,577,856 model parameters during training.
- **LoRA:** Freezes the pretrained weights and introduces trainable low-rank adapters into the attention query (`q`) and value (`v`) projections [4].

LoRA represents the adapted weight matrix as:

$$
W' = W_0 + \frac{\alpha}{r}BA
$$

where $W_0$ is the frozen pretrained weight matrix, $A$ and $B$ are trainable low-rank matrices, $r$ is the adapter rank, and $\alpha$ controls the update scaling.

**Table 3. Fine-tuning configurations.**

| Configuration | Trainable Parameters | Trainable (%) |
|---|---:|---:|
| Zero-shot | 0 | 0% |
| Full FT | 247,577,856 | 100% |
| LoRA ($r=8$, $\alpha=16$) | 884,736 | 0.3561% |

LoRA uses a dropout rate of **0.05** and targets the `q` and `v` projections. Compared with Full FT, it reduces trainable parameters by approximately **99.64%**.

### 4.2 Training and Experimental Setup

Both Full FT and LoRA were trained using a **custom PyTorch training loop**, without the Hugging Face Trainer API. The models minimise token-level cross-entropy loss:

$$
\mathcal{L}_{CE} = -\frac{1}{T}\sum_{t=1}^{T}
\log P_{\theta}(y_t \mid y_{<t}, x)
$$

where $x$ is the input radiology report, $y_t$ is the reference summary token, and $T$ is the number of valid target tokens. Padding labels (`-100`) are excluded from the loss.

**Table 4. Training hyperparameters.**

| Hyperparameter | Full FT | LoRA |
|---|---|---|
| Epochs | 3 | 3 |
| Batch size | 4 | 4 |
| Gradient accumulation | 4 | 4 |
| Effective batch size | 16 | 16 |
| Learning rate | 5e-5 | 1e-4 |
| Random seed | 42 | 42 |
| Hardware | NVIDIA A100 | NVIDIA A100 |

Gradient accumulation provides an effective batch size of $4 \times 4 = 16$, reducing per-step GPU memory requirements.

Experiments were conducted on the **UQ Rangpur HPC cluster**. Validation loss was recorded after each epoch, and the checkpoint with the lowest validation loss was selected for subsequent evaluation.

Both configurations showed decreasing training and validation losses over three epochs (Figures 3 and 4). Full FT achieved a lower final validation loss (0.4668) than LoRA (0.7654), indicating better optimisation under the tested configurations. Neither model exhibited an increase in validation loss during training.

| Full Fine-Tuning | LoRA |
|:---:|:---:|
| ![Full FT Loss](figures/loss_curve_full.png) | ![LoRA Loss](figures/loss_curve_lora.png) |
| *Figure 3. Full FT training history.* | *Figure 4. LoRA training history.* |


## 5. Results and Evaluation
### 5.1 Evaluation Metrics
### 5.2 Summarisation Performance
### 5.3 Computational Efficiency
### 5.4 Results Discussion

## 6. Clinical Safety Analysis
### 6.1 Qualitative Error Analysis
### 6.2 Failure Patterns and Root Causes
### 6.3 Recommendations and Limitations

## 7. Reproducibility
### 7.1 Environment and Dependencies
### 7.2 Training and Evaluation Instructions

## 8. Conclusion

## 9. References

[1] Xiao et al. (2025). *Overview of the BioLaySumm 2025 Shared Task*. BioNLP, Association for Computational Linguistics (ACL).

[2] Raffel et al. (2020). *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer*. Journal of Machine Learning Research (JMLR), 21(140).

[3] Chung et al. (2024). *Scaling Instruction-Finetuned Language Models*. Journal of Machine Learning Research (JMLR), 25(70).

[4] Hu et al. (2022). *LoRA: Low-Rank Adaptation of Large Language Models*. International Conference on Learning Representations (ICLR).

[5] Lin (2004). *ROUGE: A Package for Automatic Evaluation of Summaries*. Text Summarization Branches Out, Association for Computational Linguistics (ACL).

[6] Zhang et al. (2025). *AEHRC at BioLaySumm 2025: Leveraging T5 for Lay Summarisation of Radiology Reports*. Proceedings of the 24th Workshop on Biomedical Language Processing (Shared Tasks), ACL, pp. 171–178.

[7] Jeblick et al. (2024). *ChatGPT Makes Medicine Easy to Swallow: An Exploratory Case Study on Simplified Radiology Reports*. European Radiology, 34, pp. 2817–2825.

[8] Zhao et al. (2026). *X-ray Made Simple: Lay Radiology Report Generation and Robust Evaluation*. Findings of ACL 2026, pp. 34583–34598.

## 10. AI Usage Disclosure