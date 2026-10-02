# TB-MER: A Controlled Study of Bilinear Fusion for Multimodal Emotion Recognition

Official implementation and reproducibility materials for **TB-MER**, a multimodal emotion recognition framework developed to investigate the contribution of second-order cross-modal interaction modeling under a controlled experimental setting.

TB-MER uses frozen CLIP ViT-L/14 representations and performs frame-level fusion between visual and textual features before temporal aggregation. The experimental design compares two fusion strategies while keeping the remaining components of the pipeline fixed:

- **Concatenation:** first-order feature fusion.
- **MLB:** low-rank bilinear fusion for explicit second-order cross-modal interaction modeling.

The repository contains the implementation, preprocessing utilities, experimental scripts, per-seed evaluation results, statistical analyses, and class-wise analyses supporting the associated manuscript.

## Overview

The primary objective of this work is to determine whether explicitly modeling second-order interactions between visual and textual representations improves multimodal emotion recognition when the underlying pretrained representations and downstream experimental pipeline are controlled.

The experimental pipeline consists of:

1. Frozen CLIP ViT-L/14 visual and textual feature extraction.
2. Frame-level multimodal fusion using either concatenation or low-rank bilinear fusion.
3. Temporal aggregation of the fused frame representations.
4. A shared classification head for emotion prediction.

The fusion mechanism is treated as the primary experimental variable. The CLIP backbone, temporal aggregation strategy, classifier, dataset splits, training protocol, and evaluation procedure are otherwise kept fixed within the controlled comparison.

Experiments are conducted on **MELD** and **IEMOCAP**. A preliminary temporal study on MELD compares temporal mean pooling with a lightweight Conformer-based encoder at multiple visual sampling frequencies. Based on this model-selection experiment, temporal mean pooling at 4 Hz is used for the subsequent controlled fusion experiments.

The repository includes the multi-seed outputs used to compute the aggregate results, paired statistical tests, per-class recall analysis, and confusion-based error-flow analysis reported in the manuscript.

## Repository Structure

```text
TB-MER/
├── README.md
├── requirements.txt
├── src/
│   ├── feature_extractor.py
│   ├── extract_frames.py
│   ├── cached_dataset.py
│   ├── classifier.py
│   ├── mlb.py
│   ├── conformer.py
│   ├── train.py
│   ├── evaluate.py
│   ├── run_experiments.py
│   ├── paired_ttest.py
│   ├── analyze_confusion.py
│   ├── meld_dataset.py
│   ├── iemocap_dataset.py
│   └── datasets/
│       ├── iemocap_parser.py
│       └── create_iemocap_splits.py
├── analysis/
│   └── summarize_temporal_selection.py
└── results/
    ├── meld/
    │   ├── mean_pooling_4hz/
    │   │   ├── concat/
    │   │   ├── mlb/
    │   │   ├── statistical_analysis/
    │   │   └── confusion_analysis/
    │   └── temporal_selection/
    │       ├── conformer_2hz/
    │       ├── conformer_4hz/
    │       ├── conformer_6hz/
    │       ├── conformer_8hz/
    │       └── temporal_selection_summary.json
    └── iemocap/
        └── mean_pooling_4hz/
            ├── concat/
            ├── mlb/
            └── statistical_analysis/

## Datasets

Experiments are conducted on the MELD and IEMOCAP benchmarks. The original datasets are **not redistributed in this repository**. Users should obtain them from their respective original sources and comply with the corresponding licenses and access conditions.

### MELD

[MELD](https://affective-meld.github.io/) is used with its official training, development, and test partitions. The experiments retain the seven emotion categories provided by the dataset:

`anger`, `disgust`, `fear`, `joy`, `neutral`, `sadness`, and `surprise`.

The preprocessing pipeline associates each utterance transcript with the corresponding utterance-level video clip. Visual frames are sampled from the complete utterance clip.

### IEMOCAP

[IEMOCAP](https://sail.usc.edu/iemocap/) requires access through its original provider and is not included in this repository.

Six emotion categories are retained:

`ang`, `exc`, `fru`, `hap`, `neu`, and `sad`.

The original `happy` (`hap`) and `excited` (`exc`) annotations are retained as separate classes.

The evaluation protocol uses **Session 5 exclusively as the test set**. Sessions 1–4 form the development data, from which 10% of the dialogues are selected for validation using a fixed random seed of 42. Splitting is performed at the dialogue level rather than the utterance level to avoid dialogue overlap between training and validation partitions.

For IEMOCAP, utterance boundaries are obtained from the emotion annotation files. The corresponding start and end timestamps are used to extract visual frames from the dialogue-level video.

## Environment Setup

The implementation is written in Python and uses PyTorch together with the OpenAI CLIP implementation.

Clone the repository and install the required dependencies:

```bash
git clone <REPOSITORY_URL>
cd TB-MER
pip install -r requirements.txt
```

The released `requirements.txt` lists the direct dependencies required by the experimental and analysis pipeline. Exact package versions from the original experimental environment may be added separately when available.

A CUDA-capable GPU is recommended for feature extraction and model training. The experiments reported in the manuscript were executed using GPU acceleration.

## Data Preparation

The preprocessing pipeline consists of three main stages:

1. dataset metadata and split preparation;
2. visual frame extraction;
3. frozen CLIP feature extraction.

Raw datasets, extracted video frames, and cached CLIP representations are intentionally excluded from the repository.

### IEMOCAP metadata and splits

After obtaining IEMOCAP, the annotation and transcription files can be parsed using:

```bash
python src/datasets/iemocap_parser.py
```

The six-class experimental partitions are then generated using:

```bash
python src/datasets/create_iemocap_splits.py
```

The split-generation script reserves Session 5 for testing and performs the dialogue-level training/validation split over Sessions 1–4 using random seed 42.

### Visual frame extraction

Frames are extracted using `src/extract_frames.py`.

For the main MELD experiments at 4 Hz:

```bash
python src/extract_frames.py --dataset meld --split train --sampling_mode fps --target_fps 4
python src/extract_frames.py --dataset meld --split dev --sampling_mode fps --target_fps 4
python src/extract_frames.py --dataset meld --split test --sampling_mode fps --target_fps 4
```

For IEMOCAP:

```bash
python src/extract_frames.py --dataset iemocap --split train --sampling_mode fps --target_fps 4
python src/extract_frames.py --dataset iemocap --split dev --sampling_mode fps --target_fps 4
python src/extract_frames.py --dataset iemocap --split test --sampling_mode fps --target_fps 4
```

For the MELD temporal encoder selection experiment, the Conformer-based configuration was evaluated at 2, 4, 6, and 8 Hz. The corresponding frame sets can be generated by changing `--target_fps`, for example:

```bash
python src/extract_frames.py --dataset meld --split train --sampling_mode fps --target_fps 2
python src/extract_frames.py --dataset meld --split train --sampling_mode fps --target_fps 6
python src/extract_frames.py --dataset meld --split train --sampling_mode fps --target_fps 8
```

The same sampling frequency must be used for the training, development, and test partitions of each experimental configuration.

### Frozen CLIP feature extraction

TB-MER uses **CLIP ViT-L/14** as the frozen vision-language backbone. Each utterance transcript is encoded once as a 768-dimensional textual representation, while the sampled visual frames are encoded individually to obtain a variable-length sequence of 768-dimensional visual representations.

The CLIP parameters remain frozen during feature extraction and subsequent TB-MER training. Extracted representations are cached to disk and subsequently loaded by `CachedFeatureDataset`, avoiding repeated CLIP inference across experimental seeds.

Feature extraction is performed using:

```bash
python src/feature_extractor.py --dataset meld --split train
python src/feature_extractor.py --dataset meld --split dev
python src/feature_extractor.py --dataset meld --split test
```

and equivalently for IEMOCAP:

```bash
python src/feature_extractor.py --dataset iemocap --split train
python src/feature_extractor.py --dataset iemocap --split dev
python src/feature_extractor.py --dataset iemocap --split test
```

Users should verify the input and output paths in the preprocessing scripts according to their local dataset organization before running feature extraction.

## Running the Experiments

### Controlled Fusion Comparison

The main experiments compare two multimodal fusion strategies under an otherwise fixed experimental pipeline:

- **Concatenation (`concat`)** — first-order fusion of the projected visual and textual representations.
- **Low-rank bilinear fusion (`mlb`)** — multiplicative second-order interaction modeling between the projected visual and textual representations.

Fusion is performed at the frame level before temporal aggregation. For the main controlled experiments, temporal mean pooling is used for both fusion strategies. The frozen CLIP backbone, temporal aggregation, prediction head, optimization protocol, dataset partitions, and evaluation procedure are kept fixed within each dataset comparison.

Experiments are repeated using the following ten random seeds:

```text
0, 7, 13, 21, 42, 56, 78, 101, 123, 999
```

Individual experiments are configured through `src/train.py`. The two fusion conditions are selected using:

```bash
--fusion concat
```

or:

```bash
--fusion mlb
```

with temporal mean pooling selected using:

```bash
--temporal_encoder none
```

The complete multi-seed experiment can also be automated using:

```bash
python src/run_experiments.py
```

The resulting trained model and experiment metadata are stored for subsequent evaluation with `src/evaluate.py`.

### Temporal Encoder Selection

Before the controlled fusion comparison, a preliminary experiment on MELD was conducted to select the temporal aggregation strategy used in the main experiments.

The evaluated configurations are:

- temporal mean pooling at 4 Hz;
- Conformer-based temporal encoding at 2 Hz;
- Conformer-based temporal encoding at 4 Hz;
- Conformer-based temporal encoding at 6 Hz;
- Conformer-based temporal encoding at 8 Hz.

For each temporal configuration, both concatenation and MLB are evaluated using the same ten random seeds. Temporal-encoder performance is summarized by first computing the multi-seed mean for each fusion strategy and subsequently averaging the two fusion-specific means. This provides a fusion-agnostic comparison for temporal model selection.

The released per-seed outputs for this experiment are available under:

```text
results/meld/temporal_selection/
```

The aggregated temporal-selection results can be reproduced using:

```bash
python analysis/summarize_temporal_selection.py
```

The generated summary is stored as:

```text
results/meld/temporal_selection/temporal_selection_summary.json
```

### Experimental Outputs

Each evaluated run produces a JSON file containing the experimental configuration and evaluation results, including:

- dataset and random seed;
- fusion and temporal configuration;
- aggregate evaluation metrics;
- per-class recall;
- confusion matrix;
- ground-truth and predicted labels;
- sample identifiers.

The released per-seed results for the main controlled experiments are organized as:

```text
results/
├── meld/
│   └── mean_pooling_4hz/
│       ├── concat/
│       └── mlb/
└── iemocap/
    └── mean_pooling_4hz/
        ├── concat/
        └── mlb/
```

These files allow the reported aggregate, class-wise, and statistical analyses to be reproduced without retraining the models.

## Evaluation and Statistical Analysis

### Evaluation Metrics

Model performance is evaluated using both aggregate and class-wise metrics. The primary aggregate metrics are:

- **Weighted F1-score**, which accounts for the empirical class distribution.
- **Macro F1-score**, which assigns equal importance to each emotion category and therefore provides a complementary class-balanced evaluation.

Per-class recall is additionally reported to examine how the fusion strategies affect individual emotion categories.

The evaluation pipeline also stores the confusion matrix, ground-truth labels, predicted labels, and sample identifiers for each run, enabling subsequent class-wise and error-flow analyses.

### Multi-Seed Evaluation

All controlled fusion experiments are evaluated over the same ten random seeds:

```text
0, 7, 13, 21, 42, 56, 78, 101, 123, 999
```

Aggregate results are reported as the mean and standard deviation across seeds.

Because concatenation and MLB are evaluated using matched random seeds under the same experimental protocol, statistical comparisons are performed on paired observations.

### Paired Statistical Tests

Differences between MLB and concatenation are evaluated using a two-sided paired t-test with a significance level of:

```text
alpha = 0.05
```

Effect size is quantified using paired-samples Cohen's d, computed from the mean and sample standard deviation of the within-seed differences:

```text
d = mean(MLB - Concat) / std(MLB - Concat)
```

The statistical analysis can be reproduced with `src/paired_ttest.py`.

For MELD:

```bash
python src/paired_ttest.py \
    --mlb results/meld/mean_pooling_4hz/mlb \
    --concat results/meld/mean_pooling_4hz/concat \
    --out results/meld/mean_pooling_4hz/statistical_analysis
```

For IEMOCAP:

```bash
python src/paired_ttest.py \
    --mlb results/iemocap/mean_pooling_4hz/mlb \
    --concat results/iemocap/mean_pooling_4hz/concat \
    --out results/iemocap/mean_pooling_4hz/statistical_analysis
```

The generated outputs are stored in:

```text
statistical_analysis/
├── ttest_results.json
└── ttest_report.txt
```

The IEMOCAP statistical outputs are included as supplementary reproducibility material. The inferential statistical discussion in the associated manuscript focuses on the MELD controlled experiment.

### Confusion and Error-Flow Analysis

To investigate whether second-order fusion changes the structure of classification errors beyond aggregate performance, confusion matrices from the matched MELD runs are aggregated and compared.

For each fusion strategy, the confusion matrix is row-normalized to represent the distribution of predictions for each ground-truth emotion class. The difference matrix is then computed as:

```text
Delta = MLB - Concatenation
```

For off-diagonal entries:

- a **positive** value indicates that the corresponding misclassification occurs more frequently under MLB;
- a **negative** value indicates that the corresponding misclassification occurs less frequently under MLB.

The direction of each error flow is:

```text
ground-truth class -> predicted class
```

The MELD confusion analysis can be reproduced using:

```bash
python src/analyze_confusion.py \
    --mlb results/meld/mean_pooling_4hz/mlb \
    --concat results/meld/mean_pooling_4hz/concat \
    --out results/meld/mean_pooling_4hz/confusion_analysis
```

The generated analysis is stored under:

```text
results/meld/mean_pooling_4hz/confusion_analysis/
```

including the numerical confusion-difference results and the corresponding visualizations.

This analysis is intended to characterize changes in class-wise error structure rather than to establish that one fusion strategy is uniformly superior to the other.

## Reproducing the Paper Results

The repository provides the per-seed evaluation outputs underlying the main experiments reported in the associated manuscript. These outputs can be used to reproduce the aggregate and statistical analyses without retraining the models.

### Temporal Encoder Selection

The complete MELD temporal-selection experiment is available under:

```text
results/meld/temporal_selection/
```

The directory contains the per-seed results for the Conformer-based temporal encoder at 2, 4, 6, and 8 Hz. The corresponding temporal mean-pooling results are available under:

```text
results/meld/mean_pooling_4hz/
```

The fusion-agnostic temporal-selection summary can be regenerated using:

```bash
python analysis/summarize_temporal_selection.py
```

The generated summary is stored in:

```text
results/meld/temporal_selection/temporal_selection_summary.json
```

### MELD Controlled Fusion Experiment

The ten-seed results for the main MELD comparison are available under:

```text
results/meld/mean_pooling_4hz/
├── concat/
└── mlb/
```

These files provide the results underlying the controlled comparison between first-order concatenation and second-order low-rank bilinear fusion.

The associated paired statistical analysis is available under:

```text
results/meld/mean_pooling_4hz/statistical_analysis/
```

and the confusion-difference and error-flow analyses are available under:

```text
results/meld/mean_pooling_4hz/confusion_analysis/
```

### IEMOCAP Controlled Fusion Experiment

The corresponding ten-seed IEMOCAP results are available under:

```text
results/iemocap/mean_pooling_4hz/
├── concat/
└── mlb/
```

Supplementary statistical outputs computed from these matched runs are provided under:

```text
results/iemocap/mean_pooling_4hz/statistical_analysis/
```

These statistical files are provided for reproducibility and completeness; inferential statistical claims in the associated manuscript are based on the MELD controlled experiment.

### Precomputed Results

The released result files make it possible to inspect and reproduce the principal quantitative analyses without access to the original datasets or GPU-based model training.

In particular, the precomputed outputs support reproduction of:

- multi-seed mean and standard deviation;
- Macro F1 and Weighted F1 comparisons;
- per-class recall comparisons;
- paired two-sided t-tests;
- paired-samples Cohen's d effect sizes;
- MELD confusion-difference analysis;
- MELD error-flow analysis; and
- the fusion-agnostic temporal encoder selection summary.

Access to the original MELD and IEMOCAP datasets is required only when reproducing the complete pipeline from raw data, including preprocessing, CLIP feature extraction, model training, and evaluation.

## Reproducibility Scope

This repository is designed to support two levels of reproducibility.

**Analysis-level reproduction** can be performed directly from the released per-seed result files. This does not require access to MELD or IEMOCAP and allows the aggregate, statistical, class-wise, and confusion-based analyses to be regenerated.

**End-to-end reproduction** requires users to obtain MELD and/or IEMOCAP from their original providers and execute the preprocessing, frozen CLIP feature extraction, training, and evaluation pipeline described above.

Raw dataset files, extracted video frames, cached CLIP representations, model checkpoints, and temporary training artifacts are not redistributed.

## Data Availability

The experiments in this study use the MELD and IEMOCAP datasets. The original datasets are not redistributed as part of this repository.

MELD should be obtained from its original distribution source. IEMOCAP is distributed by its original provider subject to its applicable access conditions. Users wishing to reproduce the complete experimental pipeline should obtain the datasets directly from their respective providers.

This repository provides the source code, preprocessing utilities, experimental pipeline, analysis scripts, and per-seed results supporting the findings reported in the associated manuscript.

The released results include the outputs required to reproduce the reported multi-seed aggregate analyses, paired statistical comparisons, class-wise recall analysis, confusion-difference analysis, and temporal encoder selection summary without redistributing the original benchmark data.

A persistent archival DOI for this repository will be added following the public software release.

## Citation

If you use this code or the associated experimental results in your research, please cite the corresponding paper.

```bibtex
@article{AlvaFelixDiazTBMER,
  title   = {TB-MER: A Controlled Study of Bilinear Fusion for Multimodal Emotion Recognition},
  author  = {Alva Felix Diaz, Jorge and Xie, Yong Cheng},
  journal = {Pattern Recognition},
  note    = {Manuscript submitted for publication}
}
```

The citation information will be updated after publication.

A separate citation for the archived software release will also be provided after assignment of the repository DOI.

## License

The source code in this repository is released under the terms specified in the `LICENSE` file.

The MELD and IEMOCAP datasets are not covered by this repository's software license. Users are responsible for complying with the licenses, terms of use, and access conditions established by the original dataset providers.

## Contact

For questions regarding the implementation or reproducibility of the experiments, please contact:

**Jorge Alva Felix Diaz**  
Zhejiang University  
Email: 22421372@zju.edu.cn
