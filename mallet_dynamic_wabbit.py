from __future__ import print_function
from gensim import corpora, models, matutils
import re, nltk
import numpy as np
import scipy.stats as stats
from text_processing import text_pre_processing
import matplotlib.pyplot as plt
import logging
import os

MODELS_DIR = 'C:\\Users\\Norbert\\Desktop\\work\\Research 2016\\LDA_Models\\'

def getDictionary():
    dictionary = corpora.Dictionary.load(os.path.join(MODELS_DIR, "airCanada.dict"))
    return dictionary

def getCorpus():
    corpus = corpora.MmCorpus(os.path.join(MODELS_DIR, "airCanada.mm"))
    return corpus

def set_up():
    np.random.seed(2016)  # seed
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)  # logger

    # pre-processing
    data = text_pre_processing('C:\Users\Norbert\Desktop\work\Research 2016\Data sets\Air_Canada_sentences.csv')
    return data

def mallet(my_corpus, dictionary, dir, topicNum):
    mallet = models.wrappers.LdaMallet(mallet_path=dir, corpus=my_corpus, num_topics=topicNum, id2word=dictionary)
    print(mallet.print_topics(num_topics=topicNum))

def dynamic(my_corpus, dictionary, dir, topicNum):
    model = models.wrappers.DtmModel(dtm_path=dir, corpus=my_corpus,model = 'fixed', num_topics=topicNum,id2word=dictionary)
    model.print_topics(topics = topicNum)

def wabbit(corpus, dictionary, dir, topicNum):
    lda = models.wrappers.LdaVowpalWabbit(dir,corpus=corpus,num_topics=topicNum,id2word=dictionary)
    print(lda.print_topics(topicNum))

    #lda.save('vw_lda.model')
    #lda = models.wrappers.LdaVowpalWabbit.load('vw_lda.model')
    # get bound on log perplexity for given test set
    #print(lda.log_perpexity(test_corpus))

def main():
     #data = set_up()
     dictionary = getDictionary()
     corpus = getCorpus()

     # malletDir = 'C:\\Users\\Norbert\\Desktop\\mallet-2.0.8RC3\\bin\\mallet'
     # dynamicDir = 'C:\\Users\\Norbert\\Desktop\\dtm-master\\bin\\dtm-win64.exe'
     # mallet(corpus, dictionary, malletDir, 5)
     # dynamic(corpus, dictionary, dynamicDir, 5)

     wabbitDir = 'C:\\Users\\Norbert\\Desktop\\vowpal_wabbit-8.1.1\\vowpalwabbit\\win32\\make_config_h.exe'

     wabbit(corpus, dictionary, wabbitDir, 5)


if __name__ == "__main__": main()