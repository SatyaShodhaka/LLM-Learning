# Beyond One-Shot Learning: Sustaining Growth in LLMs

This project focuses on enhancing large language models (LLMs) by identifying and integrating new knowledge using innovative strategies like Question-based Retrieval Augmented Generation (RAG) and curriculum learning.

## Directory Structure

### Root Directory:

This is the main directory containing all components of the project. It is further divided into several subdirectories and scripts.

### Subdirectory: `data`

Contains various datasets and related resources used for training and evaluation.

- `chat_gpt_questions`: Questions generated for evaluating ChatGPT.
- `cpt_results`: Results from our experiments.
- `freshqa`, `judgements_v2`, `openmatch`, `questions_v1`, `wikimia`: Various datasets used in our experiments.
- `newwiki_passages_v2.csv` and `newwiki_passages.csv`: CSV files containing passages from Wikipedia, used as the corpus for identifying new knowledge.

### Subdirectory: `evaluation`

Includes scripts and outputs related to evaluating the performance of our models.

- `eval_out`: Contains output from evaluation runs.
- `compute_metrics.py`: Script for computing evaluation metrics.
- `evaluate_cl.py`: Evaluates curriculum learning strategies.
- `evaluate_dpr.py`: Evaluates dense passage retrieval performance.
- `evaluation_metrics.py`: Additional metrics computations.

### Subdirectory: `notebooks/cpt`

Contains Jupyter notebooks used for prototyping and experimentation.

- `PEFT_CPT.ipynb`: Notebook for parameter-efficient fine-tuning experiments.

### Subdirectory: `pipeline`

Contains scripts for managing and running the entire pipeline.

- `batcher.py`: Script for batching data for processing.
- `question_generation_chatgpt`: Contains scripts and outputs related to generating questions using ChatGPT.

  - `generate_questions.py`: Script for generating questions.

- `question2passage-dpr`: Contains scripts and results related to converting questions to passages using dense passage retrieval.
  - `results`: Contains results from different datasets like combinedwiki, FreshQA, oldwiki, and WikiMIA.

### Subdirectory: `scripts`

Contains various helper scripts used throughout the project.

## Key Scripts

- `compute_metrics.py`: Calculates various metrics to evaluate the performance of our models, such as accuracy, precision, recall, and F1-score.
- `evaluate_cl.py`: Evaluates the effectiveness of different curriculum learning strategies by running experiments and comparing results.
- `evaluate_dpr.py`: Assesses the performance of the dense passage retrieval mechanism by testing it on different datasets and configurations.
- `generate_questions.py`: Leverages the ChatGPT model to generate a set of questions from the provided corpus. These questions are then used in subsequent retrieval and evaluation tasks.
- `batcher.py`: Batches data for efficient processing during training and evaluation.

## Workflow Overview

1. **Data Preparation**: The `data` directory contains all necessary datasets, which are processed and batched using `batcher.py`.
2. **Question Generation**: Questions are generated from the corpus using `generate_questions.py` and stored in `output_generate_questions`.
3. **Dense Passage Retrieval**: The `question2passage-dpr` directory handles the conversion of questions to relevant passages using dense retrieval methods.
4. **Model Evaluation**: Evaluation scripts in the `evaluation` directory compute metrics and assess the performance of various strategies.
5. **Prototyping and Experiments**: Jupyter notebooks in `notebooks/cpt` are used for prototyping and testing new ideas.
