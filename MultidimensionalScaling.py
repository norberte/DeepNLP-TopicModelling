from __future__ import print_function
from Clustering import getWordSimilarityMatrix, import_words
import numpy as np
import os, csv
import gensim
from numpy import array
from matplotlib import font_manager
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from sklearn import manifold
from sklearn.metrics import euclidean_distances
from sklearn.decomposition import PCA

# MDS 1 http://scikit-learn.org/stable/auto_examples/manifold/plot_mds.html
# MDS 2 http://baoilleach.blogspot.ca/2014/01/convert-distance-matrix-to-2d.html
# cmdscale from http://www.nervouscomputer.com/hfs/cmdscale-in-python/
# screePlot from https://gist.github.com/johntyree/8785541

data_file = "C:\\Users\\Norbert\\Desktop\\newest data\\Doc2Vec_Words.csv"
MODELS_DIR = 'C:\\Users\\Norbert\\Desktop\\work\\Research 2016\\LDA_Models\\'

doc2vec =  gensim.models.Doc2Vec.load(os.path.join(MODELS_DIR, "docsAirCanada.doc2vec"))
word2vec =  gensim.models.Word2Vec.load(os.path.join(MODELS_DIR, "docsAirCanada.word2vec"))

def screePlot(eigvals):
    fig = plt.figure(figsize=(8, 5))
    sing_vals = np.arange(len(eigvals)) + 1
    plt.plot(sing_vals, eigvals, 'ro-', linewidth=2)
    plt.title('Scree Plot')
    plt.xlabel('Dimensions')
    plt.ylabel('Eigenvalue')

    leg = plt.legend(['Eigenvalues from MDS'], loc='best', borderpad=0.3,
                     shadow=False, prop=font_manager.FontProperties(size='small'),
                     markerscale=0.4)
    leg.get_frame().set_alpha(0.4)
    leg.draggable(state=True)
    plt.show()

def multidimensionalScaling(wordDissimilarityMatrix, wordsList):
    # wordSimilarityMatrix contains only unique words only; duplicate words from the file only show up once
    seed = 2016
    n_samples = len(wordDissimilarityMatrix)
    X_true = array(wordDissimilarityMatrix).astype(np.float)
    # X_true = X_true.reshape(n_samples, 2)

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

    for i, txt in enumerate(wordsList):
        ax.annotate(txt, (X_true[i, 0], X_true[i, 1]))

    # plt.scatter(pos[:, 0], pos[:, 1], s=20, c='g')
    # plt.scatter(npos[:, 0], npos[:, 1], s=20, c='b')
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

    # lc = LineCollection(segments,
    #                    zorder=0, cmap=plt.cm.hot_r,
    #                    norm=plt.Normalize(0, values.max()))
    # lc.set_array(similarities.flatten())
    # lc.set_linewidths(0.5 * np.ones(len(segments)))
    # ax.add_collection(lc)

    plt.show()

def multiDimensionalScaling2(wordDissimilarityMatrix, wordsList):
    mds = manifold.MDS(n_components=2, dissimilarity="precomputed", random_state=6)
    results = mds.fit(wordDissimilarityMatrix)

    coords = results.embedding_

    plt.subplots_adjust(bottom=0.1)
    plt.scatter(
        coords[:, 0], coords[:, 1], marker='o'
    )

    #for i, txt in enumerate(wordsList):
    #    plt.annotate(txt, (coords[i, 0], coords[i, 1]))

    for label, x, y in zip(wordsList, coords[:, 0], coords[:, 1]):
        plt.annotate(
            label,
            xy=(x, y), xytext=(-20, 20),
            textcoords='offset points', ha='right', va='bottom'
        )

    plt.show()

def MDS(dimension, wordDissimilarityMatrix):
    seed = np.random.RandomState(seed=3)
    mds = manifold.MDS(n_components=100, max_iter=3000, eps=1e-9,
                       random_state=seed, dissimilarity="precomputed", n_jobs=1)

    results = mds.fit(wordDissimilarityMatrix)

    coords = results.embedding_


def main():
    wordsArray = import_words(data_file)  # unique words only; duplicate words from the file only show up once
    wordSimilarityMatrix = getWordSimilarityMatrix(doc2vec, wordsArray)

    multiDimensionalScaling2(wordSimilarityMatrix, wordsArray)

if __name__ == "__main__": main()