# import pandas as pd
import pandas as pd
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, StopWordRemover, ArrayDictionary

df = pd.read_csv('data/data.csv', encoding='latin')
df.narasi = df.judul + ' ' + df.narasi

stopword = pd.read_csv('data/stopwords.csv', encoding='latin')
more_stopword = stopword.Stopwords.to_list()

stop_factory = StopWordRemoverFactory().get_stop_words()
data = stop_factory + more_stopword

dictionary = ArrayDictionary(data)
str = StopWordRemover(dictionary)

sss = []
temp = df.narasi

for line in temp:
    stop = str.remove(line)
    sss.append(stop)

df.narasi = sss
