import pandas as pd
import numpy as np
import demoji
import nltk

from nltk.stem import WordNetLemmatizer

def cleaning(data, col):
    cleaned = data.copy()
    cleaned[col] = cleaned[col].str.replace(r'&\w+;','',regex=True)
    cleaned[col] = cleaned[col].apply(lambda row_text:demoji.replace(row_text,''))
    cleaned[col] = cleaned[col].str.replace(r'[@#]\w+','',regex=True)
    cleaned[col] = cleaned[col].str.replace(r'(https?|www):\/{1,}\w+\W+\w+\/{1,}\w+','',regex=True)
    cleaned[col] = cleaned[col].str.replace(r'[0-9]+','',regex=True)
    cleaned[col] = cleaned[col].str.replace(r'[^a-zA-Z0-9]+',' ',regex=True)
    cleaned[col] = cleaned[col].str.replace(r'^\s+|\s+$','',regex=True)
    
    return cleaned

def casefolding(data,col):
    caseFolding_df = data.copy()
    caseFolding_df[col] = caseFolding_df[col].str.lower()

    return caseFolding_df

def tokenisasi(data, col):
    
    token_df = data.copy()
    token_df[col] = token_df[col].apply(lambda sentence:sentence.split(' '))
    
    return token_df

def stopword(data,col):
    stopwords = nltk.corpus.stopwords.words('english')
    stp_df = data.copy()
    for row in range(0,len(data[col])):
        tmp = data[col][row]
        cleaned = [word for word in tmp if word not in stopwords]
        data[col][row]=cleaned
    
    return stp_df

def stemming(data,col):
    stemmer = nltk.stem.PorterStemmer()
    stemmer_df = data.copy()
    for row in range(0,len(data[col])):
        tmp = data[col][row]
        for word in range(0,len(tmp)):
            stemmed = stemmer.stem(tmp[word])
            tmp[word]=stemmed
        stemmer_df[col][row]=tmp
    return stemmer_df

train_df = pd.read_csv("dataset/train.csv",index_col=False)
clean_df = cleaning(train_df,"text")

casefolding_df = casefolding(clean_df,"text")
token_df = tokenisasi(casefolding_df,"text")
stopword_df = stopword(token_df,"text")
stemming_df = stemming(token_df,"text")
print(stemming_df.head(5))