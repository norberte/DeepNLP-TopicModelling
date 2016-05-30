from __future__ import print_function
from gensim import corpora, models
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet, stopwords
from stop_words import get_stop_words
import re, nltk
import numpy as np
import matplotlib.pyplot as plt
import logging
from sklearn.cluster import KMeans


def LSI_topic():
    # seed
    np.random.seed(2016)

    # logger
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.INFO)
    # import data-set
    import csv
    csvFile = open('C:\Users\Norbert\Desktop\work\Research 2016\Data sets\Air_Canada_data-set.csv', 'r')
    reader = csv.reader(csvFile, delimiter=',', quotechar='"')
    my_list = list(reader)
    doc_set = []
    for item in my_list:
        doc_set.append(item[1])

    # pre-processing
    lemmatizer = WordNetLemmatizer()  # lemmatizer
    # stops = stopwords.words('english') # weaker stopwords
    stops = get_stop_words('en')  # stronger stopwords

    tag_to_type = {'J': wordnet.ADJ, 'V': wordnet.VERB, 'R': wordnet.ADV}  # POS tagger
    noNum = re.compile(r'[^a-zA-Z ]')  # number and punctuation remover
    shortword = re.compile(r'\W*\b\w{1,2}\b')  # short word remover (1-2 letters)

    # function that returns only the nouns
    def get_wordnet_pos(treebank_tag):
        return tag_to_type.get(treebank_tag[:1], wordnet.NOUN)

    # text cleaning function
    def clean(text):
        text.replace("air canada", "aircanada")
        text.replace("Air Canada", "aircanada")
        clean_text = noNum.sub('', text)            # remove numbers & punctuation
        words = nltk.word_tokenize(shortword.sub('', clean_text.lower()))  #tokenize
        filtered_words = [w for w in words if not w in stops]    # remove stopwords
        tags = nltk.pos_tag(filtered_words)         # get parts of speech tags
        return ' '.join(
            lemmatizer.lemmatize(word, get_wordnet_pos(tag[1]))         # lemmatization and return only nouns
            for word, tag in zip(filtered_words, tags)
        )

    texts = []
    # loop through document list & create corpus text file
    with open('C:\Users\Norbert\Desktop\corpus.txt', 'w') as f:
        f.truncate()
        for line in doc_set:
            text = clean(line)
            f.write(text + '\n')
            texts.append(text)

    # create dictionary
    dictionary = corpora.Dictionary(line.lower().split() for line in open('C:\Users\Norbert\Desktop\corpus.txt', 'rb'))
    once_ids = [tokenid for tokenid, docfreq in dictionary.dfs.iteritems() if docfreq == 1]
    dictionary.filter_tokens(once_ids)  # filter words that only appear once
    dictionary.filter_extremes(no_above=10, keep_n=100000)
    dictionary.compactify()

    # convert tokenized documents into a document-term matrix (bag of words)
    corpus = [dictionary.doc2bow(text.lower().split()) for text in texts]

    # generate LDA model
    #lsi = models.lsimodel.LsiModel(corpus=corpus, id2word=dictionary, num_topics=20)

    tfidf = models.TfidfModel(corpus, normalize=True)
    corpus_tfidf = tfidf[corpus]

    # project to 2 dimensions for visualization
    lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=20)

    MODELS_DIR = "C:\Users\Norbert\Desktop\work\Research 2016\LDA_Models"
    import os
    # write out coordinates to file
    fcoords = open(os.path.join(MODELS_DIR, "LSI_coords.csv"), 'wb')
    for vector in lsi[corpus]:
        fcoords.write("%6.4f\t%6.4f\n" % (vector[0][1], vector[1][1]))
    fcoords.close()

    MAX_K = 20

    X = np.loadtxt(os.path.join(MODELS_DIR, "LSI_coords.csv"), delimiter="\t")
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

    lsi_elbow = np.argmin(diff3[3:]) + 3

    print("Minimum")
    print(np.argmin(diff3[3:]))
    print()

    plt.plot(ks, inertias, "b*-")
    plt.plot(ks[lsi_elbow], inertias[lsi_elbow], marker='o', markersize=12,
             markeredgewidth=2, markeredgecolor='r', markerfacecolor=None)
    plt.ylabel("Inertia")
    plt.xlabel("K")

    kmeans = KMeans(lsi_elbow).fit(X)
    y = kmeans.labels_

    colors = ["b", "g", "r", "m", "c"]
    for i in range(X.shape[0]):
        plt.scatter(X[i][0], X[i][1], c=colors[y[i]], s=10)
    #plt.show()
    plt.savefig('C:\Users\Norbert\Desktop\LSI_kMeansElbow.png', bbox_inches='tight')


    lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=lsi_elbow)
    print("LSI")
    print(lsi.print_topics(lsi_elbow))

    hdp = models.HdpModel(corpus, id2word=dictionary)
    print("HDP")
    print(hdp.print_topics(lsi_elbow))

    lda= models.LdaModel(corpus, id2word=dictionary, num_topics=lsi_elbow)
    print("LDA")
    print(lda.print_topics(lsi_elbow))

def main():
    LSI_topic()

if __name__ == "__main__": main()