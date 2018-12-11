import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer

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
keywords = ["pre frontal cortex", "posterior parietal cortex", "pfc", "prefrontal cortex", "posteriorparietal cortex", "ppc"]

# this is a function to use pdfminer script in program. i have got it from stackoverflow
def convert_pdf_to_txt(path):
    rsrcmgr = PDFResourceManager()
    retstr = StringIO()
    codec = 'utf-8'
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)
    fp = open(path, 'rb')
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    password = ""
    maxpages = 0
    caching = True
    pagenos=set()

    for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages, password=password,caching=caching, check_extractable=True):
        interpreter.process_page(page)

    text = retstr.getvalue()

    fp.close()
    device.close()
    retstr.close()
    return text

# this is a function to pre_process text
def pre_process(text):
    
    # lowercase
    text=text.lower()
    
    #remove tags
    text=re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",text)
    
    # remove special characters and digits
    text=re.sub("(\\d|\\W)+"," ",text)
    
    return text

# this is a function to get stopwords
def get_stop_words(stop_file_path):
    """load stop words """
    
    with open(stop_file_path, 'r', encoding="utf-8") as f:
        stopwords = f.readlines()
        stop_set = set(m.strip() for m in stopwords)
        return frozenset(stop_set)

# function to count occurences of word in a text doc x
def count(x, word):
    return x.count(word)

# extracting text from all pdf files and storing them in array: data
for i in tqdm(range(files_start,files_end+1)):
    try:
        text  = convert_pdf_to_txt("extracted/"+str(i))
        data.append(text)
    except Exception as e:
        print("error processing ", i)


print("Extracted pdfs length ", len(data))

# loading text data into dataframe/matrix
df = pd.DataFrame({"text" : data})

# creating dataframe for dtm
dt_matrix = pd.DataFrame()

# applying preprocessing on text
df["text"] = df["text"].apply(lambda x:pre_process(x))

# counting occurences of keywords in text dataframe(df) and storing values in dtm
for word in tqdm(keywords):
    dt_matrix[word] = df["text"].apply(lambda x:count(x, word))

print(list(dt_matrix.columns.values))
dt_matrix.to_csv("counts.csv", encoding='utf-8', index=False)

# applying tf idf transformer
tfidf = TfidfTransformer()
tfidfMatrix = tfidf.fit_transform(dt_matrix).toarray()
tf_idf = pd.DataFrame(tfidfMatrix)
tf_idf.to_csv("tf_idf.csv", encoding='utf-8', index=True)
print(tfidfMatrix.shape)