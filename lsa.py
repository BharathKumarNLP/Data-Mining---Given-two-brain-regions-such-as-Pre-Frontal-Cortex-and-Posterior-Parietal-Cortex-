import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer

import nltk
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

from tqdm import tqdm
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from io import StringIO
import re

files_start = 1
files_end = 14958

data = []

# this is a function to use pdfminer script in program. i have got it from stackoverflow




print("Extracted pdfs length ", len(data))

# loading text data into dataframe/matrix
df = pd.read_csv("text.csv")
df.dropna(subset=['text'], how='all', inplace = True)

vectorizer = TfidfVectorizer(max_df=0.5, max_features=10000,
                             min_df=2, stop_words='english',
                             use_idf=True)

X_train_tfidf = vectorizer.fit_transform(df["text"])

feat_names = vectorizer.get_feature_names()

svd = TruncatedSVD(100)
lsa = make_pipeline(svd, Normalizer(copy=False))

X_train_lsa = lsa.fit_transform(X_train_tfidf)

lsa_df = pd.DataFrame(X_train_lsa)
lsa_df.to_csv("lsa.csv", encoding='utf-8', index=True)