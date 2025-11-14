# Full-Stack ML/DL/NLP Portfolio Project

This project demonstrates proficiency across core machine learning, deep learning, and natural language processing domains using industry-standard Python frameworks, including **Scikit-learn**, **TensorFlow/Keras**, and **spaCy**, with all applications deployed via **Streamlit**.

The project is structured into three distinct tasks, each focusing on a different domain and framework.

## 🚀 Live Demos

You can interact with the deployed applications using the links below.

| Task | Framework | Model | Live Streamlit App Link |
| :--- | :--- | :--- | :--- |
| **Task 1** | Scikit-learn | Decision Tree Classifier (Iris) | **[https://wesley534-wk3-ai-plp-assignment-app-streamlit-4v4a1a.streamlit.app/]**|
| **Task 2** | TensorFlow/Keras | Convolutional Neural Network (MNIST) | **[https://mnistdigitclassifier-plp.streamlit.app/]**|
| **Task 3** | spaCy | Named Entity Recognition & Sentiment | **[https://amazon-customer-review.streamlit.app/]**This is is failing due to spacy not being installed in streamlit|

***

## 📁 Project Deliverables

| File/Link | Description |
| :--- | :--- |
| `README.md` | This document. |
| **`answers.md`** | **Theory Document:** Comparative analysis of ML frameworks (Q1, Q2, Q3). |
| `requirements.txt` | Consolidated list of all Python dependencies for the entire project. |
| `model.py` | Python script to train and save the **Iris Decision Tree** model (`model.pkl`). |
| `app_streamlit.py` | Streamlit app for the **Iris Species Prediction** (Task 1). |
| `model_cnn.py` | Python script to train and save the **MNIST CNN** model (`mnist_cnn_model.h5`). |
| `app_streamlit_dl.py` | Streamlit app for the **MNIST Digit Classifier** (Task 2). |
| `nlp_streamlit_app.py` | Streamlit app for **NER and Sentiment Analysis** (Task 3). |
| `Iris.csv` | The dataset used for Task 1. |
| `.pkl`/`.h5` Files | The pre-trained models and encoders (committed for cloud deployment). |

***

## 🎯 Task Breakdown

### Task 1: Classical ML with Scikit-learn

| Detail | Description |
| :--- | :--- |
| **Goal** | Predict Iris species using classical ML. |
| **Dataset** | Iris Species Dataset (`Iris.csv`). |
| **Framework** | Scikit-learn (Decision Tree Classifier). |
| **Metrics** | Accuracy, Precision, Recall (achieved high scores due to dataset simplicity). |
| **Deployment** | Model trained and saved locally (`model.pkl`, `label_encoder.pkl`) and deployed via Streamlit (`app_streamlit.py`). |

### Task 2: Deep Learning with TensorFlow/Keras

| Detail | Description |
| :--- | :--- |
| **Goal** | Classify handwritten digits with a CNN model, achieving **>95% accuracy**. |
| **Dataset** | MNIST Handwritten Digits (loaded via Keras API). |
| **Framework** | TensorFlow/Keras (Sequential CNN Model). |
| **Result** | Achieved **99.09%** test accuracy. |
| **Deployment** | Model trained and saved (`mnist_cnn_model.h5`), and deployed via a Streamlit app (`app_streamlit_dl.py`) featuring an **interactive drawing canvas** for live prediction. |

### Task 3: NLP with spaCy

| Detail | Description |
| :--- | :--- |
| **Goal** | Perform Named Entity Recognition (NER) and rule-based sentiment analysis on text reviews. |
| **Dataset** | Sample Amazon Product Reviews (hardcoded for demonstration). |
| **Framework** | spaCy. |
| **Methods** | **NER:** Extracted entities labeled as `ORG`, `PRODUCT`, and custom `BRAND_NAME`. **Sentiment:** Rule-based keyword matching (positive/negative scores). |
| **Deployment** | Deployed via a Streamlit app (`nlp_streamlit_app.py`) allowing users to input text and see instant, visualized NLP results. |

***

## 🛠 Setup and Local Run Instructions

To run this project locally:

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/Wesley534/wk3-ai-plp-assignment.git
    cd wk3-ai-plp-assignment 
    ```

2.  **Create and Activate Virtual Environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Train/Save All Models:**
    *   *Note: Ensure `Iris.csv` is in the directory.*
    ```bash
    python model.py       # Trains Iris Model, creates .pkl files
    python model_cnn.py   # Trains MNIST CNN, creates .h5 file
    ```

5.  **Run Streamlit Apps:**
    ```bash
    streamlit run app_streamlit.py      # Run Iris App
    streamlit run app_streamlit_dl.py   # Run MNIST CNN App
    streamlit run nlp_streamlit_app.py  # Run spaCy NLP App
    ```
