from __future__ import print_function
from __future__ import division
from gensim import corpora, models, matutils, summarization
import re, nltk
import numpy as np
from nltk.corpus import wordnet
import scipy.stats as stats
from text_processing import text_pre_processing
from text_processing import remove_non_ascii
import matplotlib.pyplot as plt
import logging
from word2vec import getSimilar_for_Topics, getSimilar_for_Words, getDoc2VecModel, getWord2VecModel

tag_to_type = {'J': wordnet.ADJ, 'V': wordnet.VERB, 'R': wordnet.ADV}

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
    kl = arun(corpus=corpus, dictionary=dictionary, min_topics=1, max_topics=10, step=1, passes=20)
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

def topic_analysis():
    MODELS_DIR = 'C:\\Users\\Norbert\\Desktop\\work\\Research 2016\\LDA_Models\\'
    DATA_DIR = 'C:\\Users\\Norbert\\Desktop\\newer data\\'

    import os
    np.random.seed(2016)    # seed
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)    # logger

    ############## pre-processing ##################################################
    #sentences = text_pre_processing('C:\Users\Norbert\Desktop\work\Research 2016\Data sets\Air_Canada_sentences.csv')
    docs = text_pre_processing('C:\Users\Norbert\Desktop\work\Research 2016\Data sets\Air_Canada_data-set.csv')

    ############# create dictionary ################################################
    #sentDictionary = createDictionary(sentences)
    #sentDictionary.save(os.path.join(MODELS_DIR, "sentencesAirCanada.dict"))

    docsDictionary = createDictionary(docs)
    docsDictionary.save(os.path.join(MODELS_DIR, "docsAirCanada.dict"))

    ############# corpus creation ##################################################
    # convert tokenized documents into a document-term matrix
    #sentCorpus = [sentDictionary.doc2bow(text.lower().split()) for text in sentences]
    #corpora.MmCorpus.serialize(os.path.join(MODELS_DIR, "sentencesAirCanada.mm"), sentCorpus)

    docsCorpus = [docsDictionary.doc2bow(text.lower().split()) for text in docs]
    corpora.MmCorpus.serialize(os.path.join(MODELS_DIR, "docsAirCanada.mm"), docsCorpus)

    ########### Number of Topics Estimation ########################################
    # estimate topic number according to Arun Rajkumar's research(2010)
    #KLvalues = klDivergence(docsCorpus, docsDictionary)
    #topicNum = getMinimum(KLvalues)
    topicNum = 6

    ############# LSI Topic Model ####################################################
    #lsi = models.LsiModel(sentCorpus, id2word=sentDictionary, num_topics=topicNum)
    #print("SENTENCES LSI w/ 6 topics")
    #print(lsi.print_topics(topicNum))

    docsLSI = models.LsiModel(docsCorpus, id2word=docsDictionary, num_topics=topicNum)
    print("DOCS LSI w/ 6 topics")
    print(docsLSI.print_topics(topicNum))

    ########## unique keywords & filtration of keywords ##############################
    #sentencesTopicWords = getUniqueTopicWords(lsi, topicNum, 10)
    #sentencesAllWords = ' '.join(sentences)
    #finalSentenceTopicTerms = keywordFiltration(sentencesAllWords,sentencesTopicWords)
    #crossRefSentenceTopicWords = str(' '.join(finalSentenceTopicTerms))

    docsTopicWords = getUniqueTopicWords(docsLSI, topicNum, 10)
    docsAllWords = ' '.join(docs)
    finalDocsTopicTerms = keywordFiltration(docsAllWords, docsTopicWords)
    crossRefDocsTopicWords = str(' '.join(finalDocsTopicTerms))

    print("DOC Topic Words", docsTopicWords)
    print()
    print("Topic Words after filtration", crossRefDocsTopicWords)

    ############## model creations #####################################################
    #corpusPath = os.path.join(DATA_DIR, "corpus.txt")
    #doc2vec = getDoc2VecModel(filePath = corpusPath, dim = 100, min = 5)
    #doc2vec.save(os.path.join(MODELS_DIR, "docsAirCanada.doc2vec")) #save model
    #word2vec = getWord2VecModel(filePath = corpusPath, dim=100, min=5)
    #word2vec.save(os.path.join(MODELS_DIR, "docsAirCanada.word2vec"))  # save model


    DATA_DIR = 'C:\\Users\\Norbert\\Desktop\\super new data\\'
    corpusPath = 'C:\\Users\\Norbert\\Desktop\\newer data\\corpus.txt'
    ############### doc2vec ###########################################################
    doc2vec = models.Doc2Vec.load(os.path.join(MODELS_DIR, "airCanada_allPreProcessedWords.doc2vec"))
    word2vec = models.Word2Vec.load(os.path.join(MODELS_DIR, "airCanada_allPreProcessedWords.word2vec"))

    saveFileTo = os.path.join(DATA_DIR, "airCanada_allPreProcessed_Doc2Vec_Words.csv")
    logFile = os.path.join(DATA_DIR, "airCanada_allPreProcessed_Doc2Vec_Words.txt")
    getSimilar_for_Words(model = doc2vec,docsTopicWords = docsTopicWords, filePath = saveFileTo, logFile = logFile)

    saveFileTo = os.path.join(DATA_DIR, "airCanada_allPreProcessed_Doc2Vec_Topics.csv")
    logFile = os.path.join(DATA_DIR, "airCanada_allPreProcessed_Doc2Vec_Topics.txt")
    getSimilar_for_Topics(model = doc2vec, topicNumber = topicNum, topicModel = docsLSI, filePath = saveFileTo, logFile = logFile)

    ############### word2vec #########################################################
    saveFileTo = os.path.join(DATA_DIR, "airCanada_allPreProcessed_Word2Vec_Words.csv")
    logFile = os.path.join(DATA_DIR, "airCanada_allPreProcessed_Word2Vec_Words.txt")
    getSimilar_for_Words(model = word2vec,docsTopicWords = docsTopicWords, filePath  = saveFileTo, logFile=logFile)

    saveFileTo = os.path.join(DATA_DIR, "airCanada_allPreProcessed_Word2Vec_Topics.csv")
    logFile = os.path.join(DATA_DIR, "airCanada_allPreProcessed_Word2Vec_Topics.txt")
    getSimilar_for_Topics(model = word2vec, topicNumber = topicNum, topicModel = docsLSI, filePath = saveFileTo, logFile = logFile)


    #word_analysis(textWords, topicWords)

    #lsi.save(os.path.join(MODELS_DIR, "airCanada.lsi"))  # save model
    #lsi = models.LsiModel.load('/tmp/model.lsi')

def main():
    topic_analysis()


if __name__ == "__main__": main()
