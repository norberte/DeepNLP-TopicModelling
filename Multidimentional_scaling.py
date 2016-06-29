from __future__ import print_function
from Clustering import getWordSimilarityMatrix, import_words
import numpy as np
import os
import gensim
from numpy import array
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from sklearn import manifold
from sklearn.metrics import euclidean_distances
from sklearn.decomposition import PCA

## http://scikit-learn.org/stable/auto_examples/manifold/plot_mds.html

data_file = "C:\\Users\\Norbert\\Desktop\\newest data\\Doc2Vec_Words.csv"
MODELS_DIR = 'C:\\Users\\Norbert\\Desktop\\work\\Research 2016\\LDA_Models\\'

doc2vec =  gensim.models.Doc2Vec.load(os.path.join(MODELS_DIR, "docsAirCanada.doc2vec"))
#doc2vec =  gensim.models.Doc2Vec.load(os.path.join(MODELS_DIR, "CUSCS_chat.doc2vec"))
word2vec =  gensim.models.Word2Vec.load(os.path.join(MODELS_DIR, "docsAirCanada.word2vec"))


def multiDimScaling():
    seed = np.random.RandomState(seed=3)

    wordsArray = import_words(data_file)  # unique words only; duplicate words from the file only show up once
    wordSimilarityMatrix = getWordSimilarityMatrix(doc2vec, wordsArray)

    n_samples = len(wordSimilarityMatrix)
    X_true = array(wordSimilarityMatrix).astype(np.float)
    #X_true = X_true.reshape(n_samples, 2)

    # Center the data
    X_true -= X_true.mean()

    similarities = euclidean_distances(X_true)

    # Add noise to the similarities
    noise = np.random.rand(n_samples, n_samples)
    noise = noise + noise.T
    noise[np.arange(noise.shape[0]), np.arange(noise.shape[0])] = 0
    similarities += noise

    mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9, random_state=seed,
                       dissimilarity="precomputed", n_jobs=1)
    pos = mds.fit(similarities).embedding_

    nmds = manifold.MDS(n_components=2, metric=False, max_iter=3000, eps=1e-12,
                        dissimilarity="precomputed", random_state=seed, n_jobs=1,
                        n_init=1)
    npos = nmds.fit_transform(similarities, init=pos)

    # Rescale the data
    pos *= np.sqrt((X_true ** 2).sum()) / np.sqrt((pos ** 2).sum())
    npos *= np.sqrt((X_true ** 2).sum()) / np.sqrt((npos ** 2).sum())

    # Rotate the data
    clf = PCA(n_components=2)
    X_true = clf.fit_transform(X_true)

    pos = clf.fit_transform(pos)

    npos = clf.fit_transform(npos)

    fig = plt.figure(1)
    ax = plt.axes([0., 0., 1., 1.])

    plt.scatter(X_true[:, 0], X_true[:, 1], c='r', s=20)

    for i, txt in enumerate(wordsArray):
        ax.annotate(txt, (X_true[i, 0], X_true[i, 1]))

    #plt.scatter(pos[:, 0], pos[:, 1], s=20, c='g')
    #plt.scatter(npos[:, 0], npos[:, 1], s=20, c='b')
    plt.legend(('True position', 'MDS', 'NMDS'), loc='best')

    similarities = similarities.max() / similarities * 100
    similarities[np.isinf(similarities)] = 0

    # Plot the edges
    start_idx, end_idx = np.where(pos)
    # a sequence of (*line0*, *line1*, *line2*), where::
    #            linen = (x0, y0), (x1, y1), ... (xm, ym)
    segments = [[X_true[i, :], X_true[j, :]]
                for i in range(len(pos)) for j in range(len(pos))]
    values = np.abs(similarities)
    #lc = LineCollection(segments,
    #                    zorder=0, cmap=plt.cm.hot_r,
    #                    norm=plt.Normalize(0, values.max()))
    #lc.set_array(similarities.flatten())
   # lc.set_linewidths(0.5 * np.ones(len(segments)))
    #ax.add_collection(lc)

    plt.show()

def main():
    multiDimScaling()

if __name__ == "__main__": main()