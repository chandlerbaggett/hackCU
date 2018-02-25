import pandas as pd

df = pd.read_csv("data/wine-reviews/winemag-data-130k-v2.csv")
#print(df)

from io import StringIO

col = ['description', 'variety']
df = df[col]
#print(df)

df.columns = ['description', 'variety']
df['category_id'] = df['variety'].factorize()[0]
#print(df)
category_id_df = df[['variety', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'variety']].values)

#print(df)

#import matplotlib.pyplot as plt
#fig = plt.figure(figsize=(8,6))
#df.groupby('variety').description.count().plot.bar(ylim=0)
#plt.show()

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1,2), stop_words='english')
features = tfidf.fit_transform(df.description).toarray()
labels = df.category_id
#print(features.shape)

from sklearn.feature_selection import chi2
import numpy as np
N = 2
for variety, category_id in sorted(category_to_id.items()):
	features_chi2 = chi2(features, labels == category_id)
	indices = np.argsort(features_chi2[0])
	feature_names = np.array(tfidf.get_feature_names())[indices]
	unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
	bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
	print("# '{}':".format(variety))
	print(" . Most correlated unigrams:\n {}".format('\n.'.join(unigrams[-N:])))
	print(" . Most correlated bigrams:\n. {}".format('\n.'.join(bigrams[-N:])))

