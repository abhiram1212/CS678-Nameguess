## NameGuess: Column Name Expansion for Tabular Data

NameGuess is a tool designed to expand abbreviated column names into their full, descriptive forms, enhancing the interpretability of tabular datasets. For example, it can expand "D_ID" to "Department ID" and "E_NAME" to "Employee Name." This is achieved through a generative model fine-tuned for abbreviation expansion tasks.

### Key Features
- Utilizes GPT-4 for abbreviation expansion with contextual understanding.
- Supports real-world tabular datasets from domains such as finance, public administration, and health.
- Benchmarked on a high-quality, human-annotated dataset of 9,218 examples.

### Getting Started

#### Environment Setup
To set up the environment, use the following commands:
```bash
conda create -n nameguess python=3.11 --file requirements.txt
conda activate nameguess
python -m spacy download en_core_web_sm
```

#### Repository Overview
- **Training Data Creation**: Scripts for generating training data, including logical name identification and abbreviation generation.
- **Evaluation Scripts**: Comprehensive evaluation pipeline for testing model performance on metrics like Exact Match (EM), F1 Score, and Precision Metric (PM).
- **Robustness Evaluation**: Includes techniques for assessing the model's performance under real-world conditions such as typographical errors and synonym replacement.

### Training Data Creation
The following scripts help prepare data for training:

#### Logical Name Identification
Identify logical names from text input.
```bash
python src/cryptic_identifier.py --text "nycpolicedepartment"
```

#### Abbreviation Generation
Generate abbreviated forms from descriptive names.
```bash
python src/cryptic_generator.py --text "customer name"
```

### Benchmarks
We provide a human-annotated benchmark consisting of:
- 9,218 column names across 895 tables.
- Examples sourced from diverse domains to ensure generalizability.

[Benchmark Dataset](./data/)

### Results
The reproduced results closely align with the original findings:
- **Exact Match (EM)**: 54.98%
- **F1 Score**: 70.29%
- **Precision Metric (PM)**: 83.08%

These metrics slightly outperform the original results due to improved dataset diversity and fine-tuning techniques. The model demonstrates high accuracy under ideal conditions but faces challenges with ambiguous abbreviations and noisy inputs.

### Evaluation
Run the evaluation scripts to test the model:
```bash
python run_eval.py --model_name gpt-4
```

The evaluation includes:
- Comparison against human-annotated benchmarks.
- Analysis of robustness to typos, column order changes, and synonym replacement.

### Key Insights from Robustness Study
- **Sensitivity to Noise**: Model performance drops significantly under typographical errors and synonym replacement.
- **Context Dependence**: The model relies heavily on column context for disambiguation.
- **Improvements**: Suggestions include adversarial training and incorporating noise in training datasets.

### Contributions
This study contributes to reproducibility in AI by validating the generative model's effectiveness in real-world scenarios. Practical applications include:
- **Finance**: Expanding cryptic financial datasets for analysis.
- **Public Administration**: Enhancing usability of open government datasets.
- **Healthcare**: Standardizing column names for clinical data.
