import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv('data.csv', encoding='latin')

# DataFlair - Split the dataset
x_train, x_test, y_train, y_test = train_test_split(df.Tweet, df.Label, test_size=0.2, random_state=100)

# DataFlair - Initialize a TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(max_df=0.7)
# DataFlair - Fit and transform train set, transform test set
tfidf_train = tfidf_vectorizer.fit_transform(x_train)
tfidf_test = tfidf_vectorizer.transform(x_test)

cosine_train = cosine_similarity(tfidf_train)
cosine_test = cosine_similarity(tfidf_test, tfidf_train)

knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(cosine_train, y_train)

scores = cross_val_score(knn, cosine_train, y_train, cv=20)
print("Mean accuracy KNN %0.2f" % (scores.mean()))

pred = knn.predict(cosine_test)
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))
print(accuracy_score(y_test, pred))
