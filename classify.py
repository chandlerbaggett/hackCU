import pandas as pd

df = pd.read_csv("data/wine-reviews/winemag-data-130k-v2.csv")

#print(len(df))
#print(type(df))
desc = df['description']
print(desc)

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(desc)
print(X_train_counts)
print(X_train_counts.shape)
