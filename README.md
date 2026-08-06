# Multi-Hop Question Answering using Retrieval-Augmented Generation (RAG)

## Overview

This project implements a baseline Retrieval-Augmented Generation (RAG) system for multi-hop question answering using the HotpotQA dataset.

The objective is to retrieve relevant supporting documents using dense vector search and generate answers using an instruction-tuned language model. The project also evaluates the system using standard question-answering metrics and analyzes common failure cases.

This project was developed as part of an AI internship take-home assignment.


# Project Pipeline

```
HotpotQA Dataset
        │
        ▼
Load & Explore Dataset
        │
        ▼
Document Preprocessing
        │
        ▼
Sentence Transformer Embeddings
        │
        ▼
FAISS Vector Index
        │
        ▼
Top-5 Document Retrieval
        │
        ▼
FLAN-T5 Base
        │
        ▼
Answer Generation
        │
        ▼
Evaluation
        │
        ▼
Failure Analysis
```


# Dataset

**Dataset:** HotpotQA Question Answering Dataset

**File Used**

- `hotpot_dev_distractor_v1.json`

### Why HotpotQA?

HotpotQA is a multi-hop question answering dataset where answering a question often requires combining information from multiple Wikipedia articles. This makes it well suited for evaluating Retrieval-Augmented Generation (RAG) systems.


# Models Used

## Retriever

- Sentence Transformers
- Model: `all-MiniLM-L6-v2`

## Vector Search

- FAISS

## Generator

- Google FLAN-T5 Base


# Technologies

- Python
- PyTorch
- Transformers
- Sentence Transformers
- FAISS
- Pandas
- NumPy
- Matplotlib
- tqdm


# Methodology

The project follows these steps:

1. Load the HotpotQA dataset.
2. Build a document corpus from the provided contexts.
3. Generate dense document embeddings using Sentence Transformers.
4. Create a FAISS vector index for efficient retrieval.
5. Retrieve the Top-5 most relevant documents for each question.
6. Generate answers using FLAN-T5 Base.
7. Evaluate predictions using Exact Match (EM) and Token-level F1.
8. Analyze incorrect predictions to understand common failure cases.


# Evaluation Metrics

The following metrics were used:

- Exact Match (EM)
- Token-level F1 Score
- Retrieval Accuracy


# Results

| Metric | Score |
|---------|------:|
| Exact Match | **0.36** |
| F1 Score | **0.48** |
| Retrieval Accuracy | **0.99** |


# Observations

The retrieval component successfully identified relevant supporting documents for most questions, resulting in a high retrieval accuracy of approximately **99%**.

However, the Exact Match and F1 scores were considerably lower, indicating that the language model often struggled with multi-hop reasoning, combining evidence from multiple retrieved documents, or producing complete answers.

These observations suggest that the primary limitation of the baseline system lies in the answer generation stage rather than document retrieval.


# Repository Structure

```
├── assignment.ipynb
├── README.md
├── report.pdf
├── requirements.txt
├── baseline_prediction.csv
├── comparsion.csv
├── errors.csv
├── metrics.csv
├── predictions.csv
```

# Running the Project

This project was developed and executed using **Kaggle Notebooks** with GPU acceleration.

## Steps

1. Open the notebook in Kaggle.
2. Enable **GPU (T4)** from **Settings → Accelerator**.
3. Add the **HotpotQA Question Answering Dataset** as a Kaggle input.
4. Run all notebook cells sequentially.
5. The notebook will:
   - Load and preprocess the dataset.
   - Build document embeddings.
   - Create a FAISS index.
   - Retrieve relevant documents.
   - Generate answers using FLAN-T5.
   - Evaluate the predictions.
   - Save the output CSV files.

Generated files include:

- `baseline_predictions.csv`
- `comparsion.csv`
- `errors.csv`
- `metrics.csv`
- `predictions.csv`


# Future Work

Possible improvements include:

- Using stronger embedding models such as BAAI/BGE.
- Applying a cross-encoder reranker after retrieval.
- Improving prompt engineering.
- Using larger instruction-tuned language models.
- Fine-tuning the retriever for HotpotQA.
- Evaluating different retrieval depths (Top-5 vs Top-10).


# Author

**Sara Dhar**

M.Sc. Computer Science (Data Analytics)
