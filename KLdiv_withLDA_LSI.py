from __future__ import print_function
from gensim import corpora, models, matutils
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from stop_words import get_stop_words
import re, nltk
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import logging
import csv
from itertools import chain
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

def preprocessing(doc_set):
    lemmatizer = WordNetLemmatizer()  #lemmatizer
    stops = get_stop_words('en')        # stronger stopwords

    tag_to_type = {'J': wordnet.ADJ, 'V': wordnet.VERB, 'R': wordnet.ADV}  # POS tagger
    noNum = re.compile(r'[^a-zA-Z ]')     # number and punctuation remover
    shortword = re.compile(r'\W*\b\w{1,2}\b')       # short word remover (1-2 letters)

    # function that returns only the nouns
    def get_wordnet_pos(treebank_tag):
        return tag_to_type.get(treebank_tag[:1], wordnet.NOUN)

    # text cleaning function
    def clean(text):
        text.replace("air canada", "aircanada")
        text.replace("Air Canada", "aircanada")
        clean_text = noNum.sub('', text)
        words = nltk.word_tokenize(shortword.sub('', clean_text.lower()))
        filtered_words = [w for w in words if not w in stops]
        tags = nltk.pos_tag(filtered_words)
        return ' '.join(
            lemmatizer.lemmatize(word, get_wordnet_pos(tag[1]))
            for word, tag in zip(filtered_words, tags)
        )

    texts = []
    # loop through document list
    for line in doc_set:
        texts.append(clean(line))

    return texts

def lda():
    # seed
    np.random.seed(2016)

    # logger
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    # import data-set
    training = open('C:\\Users\\Norbert\\Desktop\\work\\Research 2016\\Data sets\\Discovery_data-set.csv', 'r')
    #testing = open('C:\Users\Norbert\Desktop\work\Research 2016\Data sets\Air_Canada_data-set.csv', 'r')

    trainingReader = csv.reader(training, delimiter=',', quotechar='"')
    #testingReader = csv.reader(testing, delimiter=',', quotechar='"')

    trainingList = list(trainingReader)
    #testingList = list(testingReader)

    trainingDocs = []
    #testingDocs = []

    for item in trainingList:
        trainingDocs.append(item[1])

    #for item in testingList:
    #    testingDocs.append(item[1])

    # pre-processing
    trainingData = preprocessing(trainingDocs)
    print()
    print(trainingData)
    #testingData = preprocessing(testingDocs)

    MODELS_DIR = 'C:\\Users\\Norbert\\Desktop\\work\\Research 2016\\LDA_Models\\'
    import os

    # create dictionary
    dictionary = corpora.Dictionary(line.lower().split() for line in trainingData)
    once_ids = [tokenid for tokenid, docfreq in dictionary.dfs.iteritems() if docfreq == 1]
    dictionary.filter_tokens(once_ids) # filter words that only appear once
    dictionary.filter_extremes(no_below=3, keep_n=None)
    dictionary.compactify()
    dictionary.save(os.path.join(MODELS_DIR, "trainingExperiment.dict")) # store the dictionary, for future reference

    # convert tokenized documents into a document-term matrix
    corpus = [dictionary.doc2bow(text.lower().split()) for text in trainingData]
    corpora.MmCorpus.serialize(os.path.join(MODELS_DIR, "trainingExperiment.mm"), corpus)

    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]

    corpora.MmCorpus.serialize(os.path.join(MODELS_DIR, "tfidfCorpus.mm"), corpus_tfidf)
    tfidf.save(os.path.join(MODELS_DIR, "trainingExperiment.tdidf"))

    lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=50)  # initialize an LSI transformation
    corpus_lsi = lsi[corpus_tfidf]
    corpora.MmCorpus.serialize(os.path.join(MODELS_DIR, "lsiCorpus.mm"), corpus_lsi)
    lsi.save(os.path.join(MODELS_DIR, "trainingExploration.lsi"))

    hdp = models.HdpModel(corpus_tfidf, id2word=dictionary)
    lda = models.LdaModel(corpus_tfidf, id2word=dictionary, num_topics=50)

    hdp.save(os.path.join(MODELS_DIR, "trainingExploration.hdp"))
    lda.save(os.path.join(MODELS_DIR, "trainingExploration.lda"))

    randProjection = models.rpmodel.RpModel(corpus_tfidf, num_topics=50)
    randProjection.save(os.path.join(MODELS_DIR, "trainingExperiment.rp"))

    KLvalues = []
    l = np.array([sum(cnt for _, cnt in doc) for doc in corpus])

    def sym_kl(p, q):
        return np.sum([stats.entropy(p, q), stats.entropy(q, p)])

    def arun(corpus, dictionary, min_topics, max_topics, step, passes):
        kl = []
        flag = 0
        for i in range(min_topics, max_topics, step):
            lda = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=i, passes=passes)
            m1 = lda.expElogbeta
            U, cm1, V = np.linalg.svd(m1)
            # Document-topic matrix
            lda_topics = lda[corpus]
            m2 = matutils.corpus2dense(lda_topics, lda.num_topics).transpose()
            cm2 = l.dot(m2)
            cm2 = cm2 + 0.0001
            cm2norm = np.linalg.norm(l)
            cm2 = cm2 / cm2norm
            if flag > 0:
                KLvalues.append(sym_kl(cm1, cm2))
            flag += 1
            kl.append(sym_kl(cm1, cm2))
        return kl

    # estimate topic number according to Arun Rajkumar's research(2010)
    kl = arun(corpus=corpus, dictionary=dictionary,min_topics= 1,max_topics=20, step=1, passes= 20)

    # Plot kl divergence against number of topics
    print("KL Divergence Values")
    print(KLvalues)
    plt.plot(kl)
    plt.ylabel('Symmetric KL Divergence')
    plt.xlabel('Number of Topics')
    plt.savefig('C:\\Users\\Norbert\\Desktop\\exploration.png', bbox_inches='tight')

    # find the proper parameter (minimize KL divergence value)
    def getMinimum(KLvalues):
        import operator
        min_index, min_value = min(enumerate(KLvalues), key=operator.itemgetter(1))
        topicNum = min_index + 1
        return topicNum

    topicNum = getMinimum(KLvalues)

    lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=topicNum)
    hdp = models.HdpModel(corpus, id2word=dictionary)
    lda = models.LdaModel(corpus, id2word=dictionary, num_topics=topicNum)

    hdp.save(os.path.join(MODELS_DIR, "exploration.hdp"))
    lda.save(os.path.join(MODELS_DIR, "exploration.lda"))
    lsi.save(os.path.join(MODELS_DIR, "exploration.lsi"))
    #lsi = models.LsiModel.load('/tmp/model.lsi')

    testingCorpus = corpora.MmCorpus(os.path.join(MODELS_DIR, "airCanada.mm"))
    testingDictionary = corpora.Dictionary.load(os.path.join(MODELS_DIR, "airCanada.dict"))

    merged_dictionary = dictionary.merge_with(testingDictionary)
    merged_corpus = chain(corpus, merged_dictionary[testingCorpus])

    dictionary.save(os.path.join(MODELS_DIR, "merged_Dictionary.dict"))
    corpora.MmCorpus.serialize(os.path.join(MODELS_DIR, "merged_corpus.mm"), merged_corpus)

    KLvalues = []
    l = np.array([sum(cnt for _, cnt in doc) for doc in testingCorpus])

    KLdiv = arun(testingCorpus, dictionary,1,10,1,20)

    plt.plot(KLdiv)
    plt.ylabel('Symmetric KL Divergence')
    plt.xlabel('Number of Topics')
    plt.savefig('C:\\Users\\Norbert\\Desktop\\testingCorpus+MergedDictionary.png', bbox_inches='tight')

    topicNum = getMinimum(KLvalues)

    testing_merged_LSI = models.LsiModel(testingCorpus, id2word = dictionary, num_topics=topicNum)
    hdp = models.HdpModel(testingCorpus, id2word = dictionary)
    lda = models.LdaModel(testingCorpus, id2word = dictionary, num_topics=topicNum)

    testing_merged_LSI.save(os.path.join(MODELS_DIR, "testingCorpus_mergedDictionary.lsi"))
    hdp.save(os.path.join(MODELS_DIR, "testingCorpus_mergedDictionary.hdp"))
    lda.save(os.path.join(MODELS_DIR, "testingCorpus_mergedDictionary.lda"))

    KLvalues = []
    l = np.array([sum(cnt for _, cnt in doc) for doc in merged_corpus])

    KLDIV = arun(merged_corpus, dictionary,1,10,1,20)

    plt.plot(KLDIV)
    plt.ylabel('Symmetric KL Divergence')
    plt.xlabel('Number of Topics')
    plt.savefig('C:\\Users\\Norbert\\Desktop\\Merged_Corpus+Dictionary.png', bbox_inches='tight')

    topicNum = getMinimum(KLvalues)

    merged_merged_LSI = models.LsiModel(corpus = merged_corpus, id2word = dictionary, num_topics = topicNum)
    merged_merged_LSI.save(os.path.join(MODELS_DIR, "mergedCorpus_mergedDictionary.lsi"))

    hdp = models.HdpModel(corpus = merged_corpus, id2word = dictionary)
    lda = models.LdaModel(corpus = merged_corpus, id2word = dictionary, num_topics=topicNum)

    hdp.save(os.path.join(MODELS_DIR, "mergedCorpus_mergedDictionary.hdp"))
    lda.save(os.path.join(MODELS_DIR, "mergedCorpus_mergedDictionary.lda"))

    lsi.add_documents(testingCorpus,decay = 0.1)
    lsi.save(os.path.join(MODELS_DIR, "mergedLSI.lsi"))

def main():
    lda()

if __name__ == "__main__": main()