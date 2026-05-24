# Standard Libraries
import os
import numpy as np
import pandas as pd
# NLP Libraries
from nltk.corpus import stopwords
from nltk.stem import snowball
# Visualization Libraries
import matplotlib.pyplot as plt
import seaborn as sns
# Sklearn Machine Learning Libraries
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score, confusion_matrix)
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC,LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor
# BERT Library
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from transformers import BertForSequenceClassification,BertTokenizer
from torch.optim import AdamW
from transformers import logging
logging.set_verbosity_error()
# Other Utilities
import time
from tqdm import tqdm
stopwords = set(stopwords.words('english'))
stemmer = snowball.SnowballStemmer('english')
import warnings
warnings.filterwarnings('ignore')
tqdm.pandas()
import gc
gc.collect()
torch.cuda.empty_cache()
torch.cuda.ipc_collect()
print(__doc__)

class FakeNewsDetector:
    def __init__(self):
        # Define device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Load model and tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
        self.model.to(self.device)
        self.test_size = 0.3
        self.random_state = 42
        self.figure_counter = 1

    def train_test_bert(self, df_train, df_val, epochs=5, batch_size=16):
        '''
        This function trains a BERT model on the given training and validation data.
        :param df_train, df_val, epochs, batch_size:
        :return: acc, precval, recall, f1, true_labels, preds
        '''

        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
        warnings.filterwarnings('ignore')

        logging.set_verbosity_error()
        train_encodings = self.tokenizer(list(df_train['text']), truncation=True, padding=True, return_tensors='pt')
        train_labels = torch.tensor(df_train['label'].values)
        train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], train_labels)

        # Prepare validation dataset
        val_encodings = self.tokenizer(list(df_val['text']), truncation=True, padding=True, return_tensors='pt')
        val_labels = torch.tensor(df_val['label'].values)
        val_dataset = TensorDataset(val_encodings['input_ids'], val_encodings['attention_mask'], val_labels)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        optimizer = AdamW(self.model.parameters(), lr=3e-5)

        self.model.train()

        for epoch in range(epochs):
            total_loss = 0
            for batch in train_loader:
                input_ids, attention_mask, labels = [b.to(self.device) for b in batch]
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                torch.cuda.empty_cache()
            print(f"Epoch {epoch + 1}: Loss = {total_loss / len(train_loader):.4f}")
        # Evaluation
        preds, true_labels = [], []
        self.model.eval()

        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_mask, labels = [b.to(self.device) for b in batch]
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                batch_preds = torch.argmax(logits, axis=1).cpu().numpy()
                preds.extend(batch_preds)
                true_labels.extend(labels.cpu().numpy())
                torch.cuda.empty_cache()

        acc = accuracy_score(true_labels, preds)
        precval = precision_score(true_labels, preds)
        recall = recall_score(true_labels, preds)
        f1 = f1_score(true_labels, preds)
        return acc, precval, recall, f1, true_labels, preds

    def evaluate_classifiers(self, df, test_size=0.3, random_state=42, loop=3,save_model=True):
        """
        This function evaluates multiple traditionl ML classifiers and Hugging Face transformer models on text-based fake news data.
        Input: df must have 'text' and 'label' columns.
        Note that the transformer models are not functional
        """
        # TF-IDF features
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        X = vectorizer.fit_transform(df['text'])
        y = df['label'].values
        modelnames = ['NB', 'LogReg', 'SVM', 'RF', 'GMM', 'LOF']
        results = {name: {'acc': [], 'precval': [], 'recall': [], 'f1': [], 'ytrue': [], 'ypred': [], 'times': []} for
                   name in modelnames}
        results['BERT'] = {'acc': [], 'precval': [], 'recall': [], 'f1': [], 'ytrue': [], 'ypred': [], 'times': []}
        results['Ensemble'] = {'acc': [], 'precval': [], 'recall': [], 'f1': [], 'ytrue': [], 'ypred': [], 'times': []}

        for i in range(loop):
            print(f" Loop {i + 1}/{loop}")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                                random_state=random_state * (i + 1))
            for mname in modelnames:
                starttime = time.time()
                if mname == 'NB':
                    model = GaussianNB()
                    X_train_dense, X_test_dense = X_train.toarray(), X_test.toarray()
                    model.fit(X_train_dense, y_train)
                    y_pred = model.predict(X_test_dense)

                elif mname == 'LogReg':
                    model = LogisticRegression(max_iter=1000)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                elif mname == 'SVM':
                    model = LinearSVC(max_iter=10000)
                    # model = SVC(probability=True)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                elif mname == 'RF':
                    model = RandomForestClassifier()
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                elif mname == 'GMM':
                    gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
                    gmm.fit(X_train.toarray())
                    y_pred = gmm.predict(X_test.toarray())
                    pred_map = {i: int(np.mean(y_train[gmm.predict(X_train.toarray()) == i]) > 0.5) for i in range(2)}
                    y_pred = np.vectorize(pred_map.get)(y_pred)

                elif mname == 'LOF':
                    model = LocalOutlierFactor(n_neighbors=5, contamination=0.05, novelty=True)
                    model.fit(X_train)
                    y_pred = model.predict(X_test)
                    y_pred = [1 if y == -1 else 0 for y in y_pred]

                else:
                    continue

                # Metrics
                runtime = time.time() - starttime
                precval = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                acc = accuracy_score(y_test, y_pred)
                results[mname]['acc'].append(acc)
                results[mname]['precval'].append(precval)
                results[mname]['recall'].append(recall)
                results[mname]['f1'].append(f1)
                results[mname]['times'].append(runtime)
                results[mname]['ytrue'].append(y_test)
                results[mname]['ypred'].append(y_pred)
                print(f" {mname}: precval={precval:.2f}, recall={recall:.2f}, acc={acc:.2f}, , f1={f1:.2f}")
                summary = pd.DataFrame({'Model': modelnames,
                                        'Accuracy': [np.mean(results[m]['acc']) for m in modelnames],
                                        'Precision': [np.mean(results[m]['precval']) for m in modelnames],
                                        'Recall': [np.mean(results[m]['recall']) for m in modelnames],
                                        'F1-Score': [np.mean(results[m]['f1']) for m in modelnames]})

            print(f"\n Evaluating BERT model")
            starttime = time.time()

            df_split = df.sample(frac=1, random_state=random_state * (i + 1)).reset_index(drop=True)
            split_idx = int(len(df_split) * (1 - test_size))
            train_data = df_split.iloc[:split_idx]
            test_data = df_split.iloc[split_idx:]
            acc, precval, recall, f1, ytrue, ypred = self.train_test_bert(train_data, test_data, epochs=5, batch_size=16)
            runtime = time.time() - starttime

            results['BERT']['acc'].append(acc)
            results['BERT']['precval'].append(precval)
            results['BERT']['recall'].append(recall)
            results['BERT']['f1'].append(f1)
            results['BERT']['times'].append(runtime)
            results['BERT']['ytrue'].append(ytrue)
            results['BERT']['ypred'].append(ypred)
            print(f" BERT:  prec={precval:.2f}, recall={recall:.2f}, acc={acc:.2f}, f1={f1:.2f}")

        # saving the model
        if save_model:
            self.model.save_pretrained('saved_bert_model/')
            self.tokenizer.save_pretrained('saved_bert_model/')
            print(" Model and tokenizer have been saved successfully")

        ytrue, ypred_ensemble, acc, precval, recall, f1 = self.ensemble_voting(results)

        results['Ensemble']['acc'].append(acc)
        results['Ensemble']['precval'].append(precval)
        results['Ensemble']['recall'].append(recall)
        results['Ensemble']['f1'].append(f1)
        results['Ensemble']['ytrue'].append(ytrue)
        results['Ensemble']['ypred'].append(ypred_ensemble)
        results['Ensemble']['times'].append(0)

        summary = pd.DataFrame({
            'Model': list(results.keys()),
            'Accuracy': [np.mean(results[m]['acc']) for m in results],
            'Precision': [np.mean(results[m]['precval']) for m in results],
            'Recall': [np.mean(results[m]['recall']) for m in results],
            'F1-Score': [np.mean(results[m]['f1']) for m in results]})

        print("\n Final Summary:")
        print(summary.sort_values(by='Accuracy', ascending=False))


        print("\nPlotting Confusion Matrices...")
        for modelname, res in results.items():
            ytrue = np.concatenate(res['ytrue'])
            ypred = np.concatenate(res['ypred'])
            cm = confusion_matrix(ytrue, ypred, normalize='true')  # normalized to percentage
            plt.figure(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', cbar=False)
            plt.title(f"Confusion Matrix - {modelname}")
            plt.xlabel('Predicted')
            plt.ylabel('True')
            self.save_figure()

            # plt.show()

        return summary, results

    def ensemble_voting(self, results):
        '''
        This function takes a dictionary of results from evaluate_classifiers() and returns the ensemble acc, prec, recall, f1.
        :return: ytrue, ypred, acc, prec, recall, f1
        '''

        modelnames = ['NB', 'LogReg', 'SVM', 'RF', 'BERT']
        prelist = []
        for m in modelnames:
            preds = results.get(m, {}).get('ypred', [])
            if preds:
                prelist.append(np.array(preds[-1]))
        minlen = min(p.shape[0] for p in prelist)
        if any(p.shape[0] != minlen for p in prelist):
            prelist = [p[:minlen] for p in prelist]
        ytrue = np.array(results[list(results.keys())[0]]['ytrue'][-1])[:minlen]

        pred_ = np.vstack(prelist)
        ypred_ensemble = np.round(np.mean(pred_, axis=0)).astype(int)
        acc = accuracy_score(ytrue, ypred_ensemble)
        prec = precision_score(ytrue, ypred_ensemble)
        recall = recall_score(ytrue, ypred_ensemble)
        f1 = f1_score(ytrue, ypred_ensemble)

        return ytrue, ypred_ensemble, acc, prec, recall, f1

    def save_figure(self):
        filename = f"{'.'}/{'fig'}{self.figure_counter}.png"
        plt.savefig(filename, bbox_inches='tight')
        print(f"saved:{filename}")
        self.figure_counter += 1


