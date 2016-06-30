from __future__ import print_function
from __future__ import division
from gensim import corpora, models, matutils, summarization
import re, nltk
import numpy as np
from nltk.corpus import wordnet
import scipy.stats as stats
from TextProcessing import text_pre_processing
import matplotlib.pyplot as plt
import os
import logging

tag_to_type = {'J': wordnet.ADJ, 'V': wordnet.VERB, 'R': wordnet.ADV}
MODELS_DIR = 'C:\\Users\\Norbert\\Desktop\\work\\Research 2016\\LDA_Models\\'
DATA_DIR = 'C:\\Users\\Norbert\\Desktop\\newer data\\'

def klDivergence(corpus, dictionary):
    KLvalues = []  # Kullback - Leibler divergence values
    l = np.array([sum(cnt for _, cnt in doc) for doc in corpus])

    # entropy calculation for the optimizer function
    def sym_kl(p, q):
        return np.sum([stats.entropy(p, q), stats.entropy(q, p)])

    # function that optimizes the topicNum parameter
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
    kl = arun(corpus=corpus, dictionary=dictionary, min_topics=1, max_topics=20, step=1, passes=3)
    print("KL values")
    print(KLvalues)
    # Plot kl divergence against number of topics
    plt.plot(kl)
    plt.ylabel('Symmetric KL Divergence')
    plt.xlabel('Number of Topics')
    plt.savefig('C:\Users\Norbert\Desktop\ldaKlDivergence.png', bbox_inches='tight')
    return KLvalues

def createDictionary(data):
    dictionary = corpora.Dictionary(line.split() for line in data)
    once_ids = [tokenid for tokenid, docfreq in dictionary.dfs.iteritems() if docfreq == 1]
    dictionary.filter_tokens(once_ids)  # filter words that only appear once
    dictionary.filter_extremes(keep_n=100000)
    dictionary.compactify()
    return dictionary

def get_wordnet_pos(treebank_tag, pos):
    return tag_to_type.get(treebank_tag[:1], pos)

def word_analysis(text, words):
    tokens = nltk.word_tokenize(text)
    docs = nltk.Text(tokens)
    #### do something with the so-far-unidentified-collocations
    #docs.collocations()

    w = words.split()
    for word in w:
        similar_words = docs.similar(word)
        tags = nltk.pos_tag(list(similar_words))
        print(similar_words)

        adjectives = get_wordnet_pos(tags[1], wordnet.ADJ)
        nouns = get_wordnet_pos(tags[1], wordnet.NOUN)
        adverbs = get_wordnet_pos(tags[1], wordnet.ADV)
        verbs = get_wordnet_pos(tags[1], wordnet.VERB)

        print("WORD")
        print(word)
        print("Adjectives")
        print(adjectives)
        print("Nouns")
        print(nouns)
        print("Adverbs")
        print(adverbs)
        print("Verbs")
        print(verbs)
        print()

def keywordFiltration(text, topicWords):
    text = text.replace("  ", " ")
    textWordCount = len(re.findall(r'\w+', text))
    topicWordCount = len(re.findall(r'\w+', topicWords))
    ratio = round((topicWordCount / textWordCount), 5) * 100
    keywords = summarization.keywords(text = text,ratio = 0.1, pos_filter='NN',lemmatize=True)

    topicList = topicWords.split()
    keywordList = keywords.split()

    crossRefList = []
    [crossRefList.append(x) for x in topicList if x in keywordList]
    return crossRefList

def unique_list(l):
    ulist = []
    [ulist.append(x) for x in l if x not in ulist]
    return ulist

def getUniqueTopicWords(lsi, topicNum, wordNum):
    topics = lsi.show_topics(num_topics=topicNum, num_words=wordNum, formatted=True)
    coefficients, topicText = zip(*topics)

    topicTerms = []
    biGramFilter = re.compile(r'[0-9\+\-\*\.\"\']')
    for line in topicText:
        topicTerms.append(biGramFilter.sub('', line))

    topicWords = ' '.join(topicTerms)
    uniqueTopicWords = ' '.join(unique_list(topicWords.split()))
    return uniqueTopicWords

# find the proper parameter (minimize KL divergence value)
def getMinimum(KLvalues):
    import operator
    min_index, min_value = min(enumerate(KLvalues), key=operator.itemgetter(1))
    topicNum = min_index + 1
    return topicNum

def getLSIModel(filePath):
    lsi = models.LsiModel.load(filePath)
    return lsi

def sentenceBasedTopicAnalysis(dataSetPath, corpusPath):
    np.random.seed(2016)  # seed
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)  # logger

    ############## text pre-processing #############################################
    #corpusPath = 'C:\\Users\\Norbert\\Desktop\\newer data\\corpus.txt'
    #dataSetPath = 'C:\Users\Norbert\Desktop\work\Research 2016\Data sets\Air_Canada_sentences.csv'
    sentences = text_pre_processing(dataSetPath, corpusPath)

    ############# create dictionary ################################################
    sentDictionary = createDictionary(sentences)
    sentDictionary.save(os.path.join(MODELS_DIR, "sentencesAirCanada.dict"))

    ############# corpus creation ##################################################
    # convert tokenized documents into a document-term matrix
    sentCorpus = [sentDictionary.doc2bow(text.lower().split()) for text in sentences]
    corpora.MmCorpus.serialize(os.path.join(MODELS_DIR, "sentencesAirCanada.mm"), sentCorpus)

    ########### Number of Topics Estimation ########################################
    # estimate topic number according to Arun Rajkumar's research(2010)
    # KLvalues = klDivergence(sentCorpus, sentDictionary)
    # topicNum = getMinimum(KLvalues)
    topicNum = 6

    ############# LSI Topic Model ####################################################
    lsi = models.LsiModel(sentCorpus, id2word=sentDictionary, num_topics=topicNum)
    print("SENTENCES LSI w/ 6 topics")
    print(lsi.print_topics(topicNum))

    ########## unique keywords & filtration of keywords ##############################
    sentencesTopicWords = getUniqueTopicWords(lsi, topicNum, 10)
    sentencesAllWords = ' '.join(sentences)
    finalSentenceTopicTerms = keywordFiltration(sentencesAllWords,sentencesTopicWords)
    crossRefSentenceTopicWords = str(' '.join(finalSentenceTopicTerms))

    print("DOC Topic Words", sentencesTopicWords)
    print()
    print("Topic Words after filtration", crossRefSentenceTopicWords)

    lsi.save(os.path.join(MODELS_DIR, "sentencesAirCanada.lsi"))  # save model
    return topicNum

def documentBasedTopicAnalysis(dataSetPath, corpusPath):
    np.random.seed(2016)  # seed
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)  # logger

    ############## pre-processing ##################################################
    docs = text_pre_processing(dataSetPath, corpusPath)

    ############# create dictionary ################################################
    docsDictionary = createDictionary(docs)
    docsDictionary.save(os.path.join(MODELS_DIR, "docsAirCanada.dict"))

    ############# corpus creation ##################################################
    # convert tokenized documents into a document-term matrix
    docsCorpus = [docsDictionary.doc2bow(text.lower().split()) for text in docs]
    corpora.MmCorpus.serialize(os.path.join(MODELS_DIR, "docsAirCanada.mm"), docsCorpus)

    ########### Number of Topics Estimation ########################################
    # estimate topic number according to Arun Rajkumar's research(2010)
    # KLvalues = klDivergence(docsCorpus, docsDictionary)
    # topicNum = getMinimum(KLvalues)
    topicNum = 6

    ############# LSI Topic Model ####################################################
    lsi = models.LsiModel(docsCorpus, id2word=docsDictionary, num_topics=topicNum)
    print("DOCS LSI w/ 6 topics")
    print(lsi.print_topics(topicNum))

    ########## unique keywords & filtration of keywords ##############################
    docsTopicWords = getUniqueTopicWords(lsi, topicNum, 10)
    docsAllWords = ' '.join(docs)
    finalDocsTopicTerms = keywordFiltration(docsAllWords, docsTopicWords)
    crossRefDocsTopicWords = str(' '.join(finalDocsTopicTerms))

    print("DOC Topic Words", docsTopicWords)
    print()
    print("Topic Words after filtration", crossRefDocsTopicWords)

    # word_analysis(textWords, topicWords) does not work yet

    lsi.save(os.path.join(MODELS_DIR, "docsAirCanada.lsi"))  # save model
    return topicNum

def main():
    documentBasedTopicAnalysis()

if __name__ == "__main__": main()