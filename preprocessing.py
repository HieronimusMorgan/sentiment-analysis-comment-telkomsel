import re

import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.stem import WordNetLemmatizer

df = pd.read_csv('sentimentanalysisproject/data/cleaned_reduction_new.csv', encoding='latin')

# create stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()
lemmatizer = WordNetLemmatizer()

stopword = pd.read_csv('sentimentanalysisproject/data/Reduction Stopword.csv', encoding='latin')
#
listStopwords = set(list(stopword.stopword))

character = ['.', ',', ';', ':', '-,', '...', '?', '!', '(', ')', '[', ']', '{', '}', '<', '>', '"', '/', '\'', '#',
             '-', '@']


def clean_doc(doc):
    # Case Folding
    # case_folding = doc.lower()
    # print('case folding: ', case_folding)

    # create stemmer
    # factory = StemmerFactory()
    # stemmer = factory.create_stemmer()
    # Stemming
    # stemmed_words = stemmer.stem(case_folding)
    # print('Stemming: ', stemmed_words)

    # Lemmatized
    # lemmatized_words = lemmatizer.lemmatize(word=stemmed_words, pos='v')
    # print('Lemmatizer: ', lemmatized_words)

    # Tokenizing
    words = doc.split(" ")
    # print('Tokenizing: ', words)

    # stopword = pd.read_csv('data/stopwords.csv', encoding='latin')
    # listStopwords = set(list(stopword.Stopwords))
    shortlisted_words = []
    # remove stop words
    for w in words:
        if w not in listStopwords:
            shortlisted_words.append(w)
    # print('Stopword: ', shortlisted_words)

    # Punctuation Removal
    # cleaned_words = [punctuation_removal(w) for w in shortlisted_words]
    # print('Punctuation: ', cleaned_words)

    # word = ' '.join(cleaned_words)
    # stemmed_words = stemmer.stem(word)
    # words = stemmed_words.split(" ")
    return shortlisted_words


def punctuation_removal(text):
    text = re.sub(r'([' + ''.
                  join(map(re.escape, character))
                  + r'])(?=\S)', r'\1 ', text)
    text = re.sub(r'(\S)([' + ''.
                  join(map(re.escape, character))
                  + r'])', r'\1 \2', text)
    # remove html markup
    text = re.sub(r"\s—\s", "", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub("(<.*?>)", "", text)
    # remove non-ascii and digits
    text = re.sub("(\\W|\\d)", " ", text)
    # remove whitespace
    text = text.strip()
    return text


def deEmojify(text):
    regrex_pattern = re.compile(pattern="["
                                        u"\U0001F600-\U0001F64F"  # emoticons
                                        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                        u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                        "]+", flags=re.UNICODE)
    return regrex_pattern.sub(r'', text)


temp = []
save = df.Tweet
for line in save:
    line = deEmojify(line)
    line = line.strip()
    cleaned = clean_doc(line)
    cleaned = ' '.join(cleaned)
    temp.append(cleaned)

df.Tweet = temp
