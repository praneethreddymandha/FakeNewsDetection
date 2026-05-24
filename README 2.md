# Fake News Detection System

![Fake News Detection](https://img.shields.io/badge/ML-Fake%20News%20Detection-blue)
![NLP](https://img.shields.io/badge/AI-Natural%20Language%20Processing-green)
![Deep Learning](https://img.shields.io/badge/AI-Deep%20Learning-red)

## Project Overview

This initiative establishes a comprehensive system for identifying false news across diverse datasets, employing a range of machine learning and deep learning methods. The system assesses traditional ML classifiers and transformer-based models on text-oriented fake news data, offering detailed performance metrics and visual representations.

## Table of Contents

- [Datasets](#datasets)
- [Features](#features)
- [Models](#models)
- [Performance Metrics](#performance-metrics)
- [Project Structure](#project-structure)
- [Key Code Components](#key-code-components)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the System](#running-the-system)
- [Results](#results)

## Datasets

The system uses four different fake news datasets:

| Dataset | Description | Features |
|---------|-------------|----------|
| COVID-19 | News articles and tweets related to COVID-19 | Text labeled as real or fake |
| FakeNewsNet | Collection of news articles from various sources | Title (used as text) with fake/real labels |
| ISOT | News articles from real and fake news websites | Multiple text features with binary labels |
| LIAR | Short statements rated on a truthfulness scale | Text with true/false labels |

## Features

- **Data preprocessing**: Cleans and tokenizes text, including emoji handling, stopword removal, and stemming
- **Balanced training**: Ensures balanced dataset through downsampling
- **Multiple model evaluation**: Compares performance of various classifiers
- **Cross-validation**: Multiple iterations for robust performance assessment
- **Detailed visualization**: Generates plots for accuracy, precision, recall, and F1 scores
- **N-gram analysis**: Visualizes important n-grams in the text data

## Models

The system implements and evaluates several models:

| Model Type | Implementation |
|------------|----------------|
| Naive Bayes | GaussianNB from scikit-learn |
| Logistic Regression | LogisticRegression from scikit-learn |
| Support Vector Machine | LinearSVC and SVC from scikit-learn |
| Random Forest | RandomForestClassifier from scikit-learn |
| Gaussian Mixture Model | GaussianMixture from scikit-learn |
| Local Outlier Factor | LocalOutlierFactor from scikit-learn |
| BERT | BertForSequenceClassification from Hugging Face transformers |
| Ensemble | Voting-based combination of all models |

## Performance Metrics

The system evaluates models using:

- Accuracy
- Precision
- Recall
- F1-Score
- Runtime performance

## Project Structure

```
project/
├── grp_proj/
│   └── data/
│       ├── covid19/      # COVID-19 dataset
│       ├── FakeNewsNet/  # FakeNewsNet dataset
│       ├── ISOT/         # ISOT dataset
│       └── LIAR/         # LIAR dataset
├── FinalProjCodes/
│   ├── main.py           # Main execution script
│   ├── LoadDataSets.py   # Dataset loading and preprocessing
│   ├── FakeNewsDetector.py # ML/DL model implementation
│   └── PlotMetric.py     # Visualization utilities
├── Test4AI/              # Jupyter notebook and figures
│   ├── Group7ProjectRev3_050425_Rev4.ipynb  # Complete analysis notebook
│   ├── Group7ProjectRev3_050425_Rev4.html   # HTML version of notebook
│   └── fig*.png          # Generated figures
└── requirements.txt      # Dependencies list
```

## Key Code Components

### 1. Data Loading and Preprocessing

```python
# From LoadDataSets.py
def preprocess_data(self, df):
    '''
    This function cleans and tokenizes the text column.
    Returns: a list of (cleaned_text, label) pairs
    '''
    cleaned_pairs = []

    for _, row in df.iterrows():
        text = str(row['text'])
        label = row['label']

        tokens = re.split(r'\W+', re.sub(r'(?<!^)(?=[A-Z])', ' ', emoji.demojize(text)))
        words = [
            w_.lower() for w in tokens
            for w_ in (nltk.word_tokenize(w) if not w.isalpha() else [w])
            if w_.isalpha() and w_.lower() not in stopwords
        ]
        words = [stemmer.stem(w) for w in words]

        cleaned_text = ' '.join(words)
        cleaned_pairs.append((cleaned_text, label))
    # Balance the dataset
    cleaned_data = pd.DataFrame(cleaned_pairs, columns=['text', 'label'])
    # ... additional balancing code ...
```

### 2. Model Training and Evaluation

```python
# From FakeNewsDetector.py
def evaluate_classifiers(self, df, test_size=0.3, random_state=42, loop=3, save_model=True):
    """
    This function evaluates multiple traditional ML classifiers and Hugging Face 
    transformer models on text-based fake news data.
    """
    # TF-IDF features
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    X = vectorizer.fit_transform(df['text'])
    y = df['label'].values
    modelnames = ['NB', 'LogReg', 'SVM', 'RF', 'GMM', 'LOF']
    results = {name: {'acc': [], 'precval': [], 'recall': [], 'f1': [], 
                      'ytrue': [], 'ypred': [], 'times': []} for name in modelnames}
    
    # Model training and evaluation loops
    # ...
```

### 3. BERT Implementation

```python
# From FakeNewsDetector.py
def train_test_bert(self, df_train, df_val, epochs=5, batch_size=16):
    '''
    This function trains a BERT model on the given training and validation data.
    '''
    train_encodings = self.tokenizer(list(df_train['text']), truncation=True, 
                                     padding=True, return_tensors='pt')
    train_labels = torch.tensor(df_train['label'].values)
    train_dataset = TensorDataset(train_encodings['input_ids'], 
                                 train_encodings['attention_mask'], train_labels)
    
    # Training and evaluation code
    # ...
```

## Getting Started

### Prerequisites

- Python 3.7 or higher
- GPU support recommended for BERT model training (CUDA compatible)

### Installation


1. Install dependencies using pip:
   ```bash
   pip install -r requirements.txt
   ```

2. Download NLTK resources:
   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('punkt')
   ```

3. Ensure datasets are available in the `grp_proj/data/` directory with the following structure:
   - `covid19/` - COVID-19 dataset files
   - `FakeNewsNet/` - FakeNewsNet dataset files
   - `ISOT/` - ISOT dataset files
   - `LIAR/` - LIAR dataset files

### Running the System

1. Execute the main script:
   ```bash
   cd FinalProjCodes
   python main.py
   ```

2. The script will:
   - Load and preprocess all datasets
   - Train and evaluate all models
   - Generate performance visualizations in the current directory

3. To adjust parameters, modify values in `main.py`:
   ```python
   # Sample parameters to adjust
   test_size=0.3     # Test set proportion
   random_state=42   # Random seed for reproducibility
   loop=1            # Number of evaluation iterations
   datalen=400       # Number of samples to use from each dataset
   ```

## Model Saving & Loading
- After training, the BERT model is saved using:
   ```python
   model.save_pretrained('bert_model/')
   tokenizer.save_pretrained('bert_model/')
   ```
- To reload later:
  ```python
  model = BertForSequenceClassification.from_pretrained('bert_model/')
  tokenizer = BertTokenizer.from_pretrained('bert_model/')
  ```
## Results

The project evaluates multiple models across four datasets, with detailed analysis of:

- Performance across different datasets
- Relative strengths of traditional ML vs. deep learning approaches
- Feature importance and n-gram analysis
- Runtime efficiency comparison

Detailed results with visualizations are available in the Jupyter notebook in the Test4AI directory.

---

## Contributors

- Zelalem Tesfaye
- Praneeth Reddy Mandha
- Alexander Keitt-Saenz

## Acknowledgments

This project incorporates multiple open-source datasets and leverages several machine learning libraries and frameworks. 