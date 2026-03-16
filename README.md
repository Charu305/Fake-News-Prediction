# 📰 Fake News Detection — LSTM & NLP

> **A Natural Language Processing (NLP) project** that classifies news articles as Real or Fake using a deep learning LSTM model — covering the full NLP pipeline from raw text preprocessing to sequence modelling with word embeddings.

---

## 📌 Project Overview

Misinformation spreads faster than corrections. Automated fake news detection is a critical tool for social media platforms, news aggregators, and fact-checking organisations. This project builds a **binary text classification model** that reads a news article and predicts whether it is **Real ✅** or **Fake ❌**.

Unlike structured/tabular ML projects, this one works entirely with **unstructured text data** — requiring a dedicated NLP preprocessing pipeline before any model can be applied. The classifier is built using an **LSTM (Long Short-Term Memory)** network, which is specifically designed to capture sequential dependencies in language.

---

## 🎯 Problem Statement

> *Given the text of a news article, classify it as Real or Fake.*

**Why this matters:**
- **Social media platforms** use such models to flag potentially misleading content at scale
- **News aggregators** use it to filter low-credibility sources
- **Search engines** use credibility signals to rank results
- **Researchers** study linguistic patterns that distinguish misinformation from factual reporting

---

## 🏗️ System Architecture

```
Raw Text Data
(Fake_compressed.csv + True_compressed.csv)
            │
            ▼
┌──────────────────────────────┐
│   Text Preprocessing         │
│   Lowercasing, punctuation   │
│   removal, stopword removal  │
│   Stemming / Lemmatization   │
└────────────┬─────────────────┘
             │  Clean text
             ▼
┌──────────────────────────────┐
│   Tokenization &             │
│   Sequence Encoding          │
│   Keras Tokenizer            │
│   Padding to fixed length    │
└────────────┬─────────────────┘
             │  Integer sequences
             ▼
┌──────────────────────────────┐
│   LSTM Model                 │
│   Embedding Layer            │
│   LSTM Layer(s)              │
│   Dense + Sigmoid output     │
└────────────┬─────────────────┘
             │
             ▼
     Real ✅  /  Fake ❌
```

---

## 🗂️ Project Structure

```
Fake-News-Prediction/
│
├── Fake_compressed.csv         # Fake news articles dataset
├── True_compressed.csv         # Real news articles dataset
└── Fake News - LSTM.ipynb      # Full NLP pipeline & LSTM model notebook
```

---

## 🔬 Technical Deep Dive

### 1. Data Loading & Labelling

- Loaded two separate datasets: `Fake_compressed.csv` (labelled `0`) and `True_compressed.csv` (labelled `1`).
- Combined into a single DataFrame with a binary `label` column.
- Shuffled the combined dataset to prevent order bias during training.
- Checked class balance — important for a binary classification task where imbalanced classes can mislead accuracy metrics.

### 2. Text Preprocessing Pipeline

Raw news text requires extensive cleaning before it can be fed to a model:

| Step | What it does | Why it matters |
|---|---|---|
| **Lowercasing** | Converts all text to lowercase | "News" and "news" are the same word |
| **Punctuation removal** | Strips special characters and symbols | Punctuation adds noise without semantic value |
| **Stopword removal** | Removes common words (the, is, at, etc.) | Focuses the model on meaningful content words |
| **Stemming / Lemmatization** | Reduces words to root form | "running", "ran", "runs" → "run" |
| **Whitespace normalisation** | Removes extra spaces and newlines | Clean input for tokeniser |

### 3. Tokenization & Sequence Encoding

- Used **Keras Tokenizer** to build a vocabulary from the training corpus — each unique word gets an integer index.
- Converted every cleaned article into a **sequence of integers** (one per word).
- Applied **padding** (`pad_sequences`) to ensure all sequences are the same fixed length — required by the LSTM for batch processing.
  - Sequences shorter than the max length are zero-padded at the start.
  - Sequences longer than the max length are truncated.

### 4. LSTM Model Architecture

```
Input (padded integer sequences)
        │
        ▼
Embedding Layer
  └── Maps each word index to a dense vector of fixed size (e.g., 128 dims)
  └── Learns word representations during training
        │
        ▼
LSTM Layer(s)
  └── Processes the sequence left-to-right
  └── Maintains a hidden state that carries context across words
  └── Captures long-range dependencies (e.g., negation, context)
        │
        ▼
Dropout Layer
  └── Regularisation to prevent overfitting on training text
        │
        ▼
Dense Layer (Sigmoid activation)
  └── Outputs probability: 0 = Fake, 1 = Real
```

**Why LSTM over simpler models?**
- A simple Bag-of-Words model loses word order entirely — "not good" and "good not" look identical.
- LSTM processes words sequentially and maintains memory of prior context, making it far better at understanding meaning from sentence structure.

### 5. Training & Evaluation

- Split dataset into **train / validation / test** sets.
- Trained with **binary cross-entropy loss** and **Adam optimiser**.
- Monitored **training vs. validation accuracy/loss curves** to detect overfitting.
- Final evaluation on held-out test set using:

| Metric | Description |
|---|---|
| **Accuracy** | Overall correct classifications |
| **Precision** | Of articles flagged as Fake, how many truly are |
| **Recall** | Of all Fake articles, how many were caught |
| **F1 Score** | Harmonic mean of Precision and Recall |
| **Confusion Matrix** | Full breakdown of TP, TN, FP, FN |

> *Precision and Recall are both critical here — a false positive (real news flagged as fake) is as harmful as a false negative (fake news missed).*

---

## 📊 Model Performance

| Model | Accuracy | Notes |
|---|---|---|
| Baseline (majority class) | ~50% | Random guess on balanced dataset |
| TF-IDF + Logistic Regression | ~90–92% | Strong classical NLP baseline |
| **LSTM with Embeddings** | **~95%+** | Best — captures sequential language patterns |

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3 |
| NLP Preprocessing | NLTK, RegEx |
| Tokenization & Sequences | Keras / TensorFlow (`Tokenizer`, `pad_sequences`) |
| Deep Learning | TensorFlow / Keras (LSTM, Embedding, Dense) |
| Data Manipulation | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| Environment | Jupyter Notebook |

---

## 🚀 How to Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/Charu305/Fake-News-Prediction.git
cd Fake-News-Prediction

# 2. Install dependencies
pip install pandas numpy matplotlib seaborn tensorflow nltk jupyter

# 3. Download NLTK resources (first run only)
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"

# 4. Launch the notebook
jupyter notebook "Fake News - LSTM.ipynb"
```

---

## 📁 Dataset Overview

| File | Description |
|---|---|
| `Fake_compressed.csv` | Fake / misinformation news articles (label = 0) |
| `True_compressed.csv` | Real / credible news articles (label = 1) |

**Key columns:**

| Column | Description |
|---|---|
| `title` | Headline of the news article |
| `text` | Full body text of the article |
| `subject` | News category (politics, world news, etc.) |
| `date` | Publication date |

---

## 💡 Key Learnings & Takeaways

- **NLP requires its own preprocessing pipeline** — unlike tabular data, text needs a sequence of domain-specific steps (lowercasing, stopword removal, stemming, tokenisation, padding) before any model can see it. Getting this pipeline right is often more impactful than model architecture choices.
- **Word order matters — use sequence models** — a Bag-of-Words model treats "not true" and "true not" identically. LSTM preserves word order and captures how meaning builds across a sentence, making it the right tool for language tasks.
- **Embedding layers learn representations** — rather than using fixed one-hot vectors, the Embedding layer learns a dense, meaningful representation for each word during training. Words with similar meanings end up with similar vectors.
- **Precision vs. Recall trade-off is real here** — in fake news detection, both types of error have real-world consequences. Flagging real news as fake (false positive) damages credibility; missing fake news (false negative) allows misinformation to spread. F1 Score, not just accuracy, is the right primary metric.
- **Pre-trained embeddings (GloVe, Word2Vec) can improve results further** — using embeddings trained on billions of words transfers general language knowledge to the task, especially valuable when the labelled dataset is small.

---

## 👩‍💻 Author

**Charunya**
🔗 [GitHub Profile](https://github.com/Charu305)

---

## 📄 License

This project is developed for educational and research purposes.
