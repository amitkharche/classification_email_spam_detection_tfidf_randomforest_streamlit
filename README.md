
---

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Streamlit](https://img.shields.io/badge/built%20with-Streamlit-orange)

# 📧 Email Spam Detection using TF-IDF & Random Forest

A machine learning solution that classifies emails as **spam** or **not spam** using text features derived from email content. This project helps organizations **filter harmful or irrelevant emails**, improve user productivity, and prevent phishing attacks.

---

## Business Objective

Spam emails waste time, clog inboxes, and pose serious cybersecurity threats. Automatically classifying emails enables:

* Cleaner inboxes
* Enhanced protection from malicious content
* Intelligent filters that learn from historical data

This project builds a **binary classification pipeline** to detect whether an email is spam (`1`) or not (`0`), based on its textual content.

---

## Dataset Overview

* **Source:** Email content dataset
* **Target:** `Spam` → Converted to binary (`1` for spam, `0` for not spam)
* **Text Features:** Transformed using **TF-IDF Vectorization**

---

## Features Used

| Category       | Feature Description                                |
| -------------- | -------------------------------------------------- |
| **Textual**    | Raw email content (preprocessed)                   |
| **Engineered** | TF-IDF vectors for keyword frequency and relevance |

---

## Model Used

* **Random Forest Classifier** (wrapped in a complete training pipeline)

---

##  Project Structure

```
├── email_spam_data.csv             # Sample input dataset
├── spam_classifier_model.pkl       # Trained model file
├── model_training.py               # Script to train, evaluate, and save model
├── app.py                          # Streamlit UI for spam prediction
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
```

---

##  Pipeline Steps

### 1. Data Cleaning

* Remove null or empty records
* Normalize and clean text

### 2. Feature Engineering

* Transform text to numerical features using **TF-IDF**

### 3. Model Training & Evaluation

* Split into training and test sets
* Train **Random Forest Classifier**
* Evaluate using classification report (precision, recall, F1)

### 4. Model Saving

* Serialize the trained model as `spam_classifier_model.pkl`

---

## How to Run This Project

### Step 1: Clone the Repository

```bash
git clone https://github.com/amitkharche/classification_email_spam_detection_tfidf_randomforest_streamlit.git
cd classification_email_spam_detection_tfidf_randomforest_streamlit
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Train the Model

```bash
python model_training.py
```

This will generate:

* `spam_classifier_model.pkl`

### Step 4: Run Streamlit App

```bash
streamlit run app.py
```

> Upload a CSV with an email text column to receive predictions on spam classification.

---

## Requirements

```txt
pandas
scikit-learn
joblib
streamlit
```

*(See full list in `requirements.txt`)*

---

## Future Enhancements

* Add NLP-based preprocessing (stopwords removal, stemming)
* Expand to multiple classifiers (Naïve Bayes, SVM, etc.)
* Add explainability using SHAP or LIME
* Real-time API integration with FastAPI

---

## Acknowledgements

* Dataset from public repositories (custom email content or open sources)
* Built using: Python, scikit-learn, Streamlit

---

## Let's Connect

Have questions or want to collaborate? Reach out here:

* [LinkedIn](https://www.linkedin.com/in/amit-kharche)
* [Medium](https://medium.com/@amitkharche14)
* [GitHub](https://github.com/amitkharche)

---
