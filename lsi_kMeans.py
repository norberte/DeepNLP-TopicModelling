from __future__ import print_function
import gensim
import os
import matplotlib.pyplot as plt
import nltk
import numpy as np
import logging
from sklearn.cluster import KMeans

# setting seed for recreatibility
np.random.seed(2016)
# logger
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

def iter_docs(topdir, stoplist):
    for fn in os.listdir(topdir):
        fin = open(os.path.join(topdir, fn), 'rb')
        text = fin.read()
        fin.close()
        yield (x for x in
            gensim.utils.tokenize(text, lowercase=True, deacc=True,
                                  errors="ignore")
            if x not in stoplist)

class MyCorpus(object):
    def __init__(self, topdir, stoplist):
        self.topdir = topdir
        self.stoplist = stoplist
        self.dictionary = gensim.corpora.Dictionary(iter_docs(topdir, stoplist))

    def __iter__(self):
        for tokens in iter_docs(self.topdir, self.stoplist):
            yield self.dictionary.doc2bow(tokens)

TEXTS_DIR = "C:\Users\Norbert\Desktop\work\Research 2016\Airline_reviews\Air_canada.csv"
MODELS_DIR = "C:\Users\Norbert\Desktop\work\Research 2016\LDA_Models"

stoplist = set(nltk.corpus.stopwords.words("english"))
corpus = MyCorpus(TEXTS_DIR, stoplist)

corpus.dictionary.save(os.path.join(MODELS_DIR, "air_canada.dict"))
gensim.corpora.MmCorpus.serialize(os.path.join(MODELS_DIR, "air_canada.mm"),
                                              corpus)

dictionary = gensim.corpora.Dictionary.load(os.path.join(MODELS_DIR,
                                                         "air_canada.dict"))
tfidf = gensim.models.TfidfModel(corpus, normalize=True)
corpus_tfidf = tfidf[corpus]

# project to 2 dimensions for visualization
lsi = gensim.models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=2)

# write out coordinates to file
fcoords = open(os.path.join(MODELS_DIR, "coords.csv"), 'wb')
for vector in lsi[corpus]:
    if len(vector) != 2:
        continue
    fcoords.write("%6.4f\t%6.4f\n" % (vector[0][1], vector[1][1]))
fcoords.close()

MAX_K = 10

X = np.loadtxt(os.path.join(MODELS_DIR, "coords.csv"), delimiter="\t")
ks = range(1, MAX_K + 1)

inertias = np.zeros(MAX_K)
diff = np.zeros(MAX_K)
diff2 = np.zeros(MAX_K)
diff3 = np.zeros(MAX_K)
for k in ks:
    kmeans = KMeans(k).fit(X)
    inertias[k - 1] = kmeans.inertia_
    # first difference
    if k > 1:
        diff[k - 1] = inertias[k - 1] - inertias[k - 2]
    # second difference
    if k > 2:
        diff2[k - 1] = diff[k - 1] - diff[k - 2]
    # third difference
    if k > 3:
        diff3[k - 1] = diff2[k - 1] - diff2[k - 2]

elbow = np.argmin(diff3[3:]) + 3

plt.plot(ks, inertias, "b*-")
plt.plot(ks[elbow], inertias[elbow], marker='o', markersize=12,
         markeredgewidth=2, markeredgecolor='r', markerfacecolor=None)
plt.ylabel("Inertia")
plt.xlabel("K")


kmeans = KMeans(elbow).fit(X)
y = kmeans.labels_

colors = ["b", "g", "r", "m", "c"]
for i in range(X.shape[0]):
    plt.scatter(X[i][0], X[i][1], c=colors[y[i]], s=10)
plt.show()

lda = gensim.models.LdaModel(corpus, id2word=dictionary, num_topics=elbow)
lda.print_topics(elbow)
