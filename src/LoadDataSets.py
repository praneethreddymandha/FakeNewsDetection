import pandas as pd
from pathlib import Path
import re
import emoji
from nltk.stem import snowball
from sklearn.utils import shuffle, resample
# NLP Libraries
import nltk
from nltk.corpus import stopwords

stopwords = set(stopwords.words('english'))
stemmer = snowball.SnowballStemmer('english')


class LoadDataSets:

    def __init__(self,data_path = '../datasets', data_names=None,datalen = None):
        if data_names is None:
            data_names = ['covid19', 'FakeNewsNet', 'ISOT', 'LIAR']
        self.data_path = data_path
        self.data_names = data_names
        self.datalen = datalen
    def get_data_path(self):
        fdir = Path(self.data_path)
        files = []
        for data in self.data_names:
            fpath = fdir / data
            for fpath in fpath.glob('*'):
                files.append(fpath)
        return files

    def get_all_datasets(self):
        files = self.get_data_path()
        print(files)
        # 1) covid19 dataset
        dcovid = pd.read_excel(files[0])
        dcovid['label'] = dcovid['label'].str.lower().map({'real': 1, 'fake': 0})
        dcovid = dcovid.rename(columns={'tweet': 'text'})
        dcovid = shuffle(dcovid, random_state=42).reset_index(drop=True)
        if self.datalen is not None and len(dcovid) > self.datalen:
            dcovid = dcovid.iloc[:self.datalen]
        # 2) FakeNewsNet
        dfake = pd.read_csv(files[2])
        dfake['label'] = 0
        dreal = pd.read_csv(files[1])
        dreal['label'] = 1
        dreal = shuffle(dreal, random_state=42).reset_index(drop=True)
        dfake = shuffle(dfake, random_state=42).reset_index(drop=True)
        dfakenews = pd.concat([dfake, dreal]). \
            sort_index(kind="merge").reset_index(drop=True)
        # To keep it uniform with other datasets, we are chaning the feature name 'title' to 'text'
        dfakenews = dfakenews.rename(columns={'title': 'text'})
        if self.datalen is not None and len(dfakenews) > self.datalen:
            dfakenews = dfakenews.iloc[:self.datalen]
        # 3) ISOT dataset
        dtrue = pd.read_csv(files[3])
        dtrue['label'] = 1
        dfalse = pd.read_csv(files[4])
        dfalse['label'] = 0
        disot = pd.concat([dtrue, dfalse]). \
            sort_index(kind="merge").reset_index(drop=True)
        disot = shuffle(disot, random_state=42).reset_index(drop=True)
        if self.datalen is not None and len(disot) > self.datalen:
            disot = disot.iloc[:self.datalen]
        # 4) LAIR dataset
        dtrain = pd.read_csv(files[5], sep='\t', header=None, dtype=str, on_bad_lines='skip')
        dtest = pd.read_csv(files[6], sep='\t', header=None, dtype=str, on_bad_lines='skip')
        dlair_ = pd.concat([dtrain, dtest]).sort_index(kind="merge").reset_index(drop=True)
        dlair = dlair_[dlair_[1].str.lower().isin(['true', 'false'])].copy()
        dlair['label'] = dlair[1].str.lower().map({'true': 1, 'false': 0})
        dlair = dlair.rename(columns={2: 'text'})
        shuffle(dlair, random_state=42).reset_index(drop=True)
        if self.datalen is not None and len(dlair) > self.datalen:
            dlair = dlair.iloc[:self.datalen]

        return dcovid,dfakenews,disot,dlair

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
        cleaned_data = pd.DataFrame(cleaned_pairs, columns=['text', 'label'])

        cleaned_df_0 = cleaned_data[cleaned_data['label'] == 0]
        cleaned_df_1 = cleaned_data[cleaned_data['label'] == 1]
        if len(cleaned_df_0) != len(cleaned_df_1):
            minsize = min(len(cleaned_df_0), len(cleaned_df_1))
            downsample_df_0 = resample(cleaned_df_0, replace=False, n_samples=minsize, random_state=42)
            downsample_df_1 = resample(cleaned_df_1, replace=False, n_samples=minsize, random_state=42)
            cleaned_df_0_1 = pd.concat([downsample_df_0, downsample_df_1])

            return cleaned_df_0_1.sample(frac=1, random_state=42).reset_index(drop=True)
        else:
            return cleaned_data