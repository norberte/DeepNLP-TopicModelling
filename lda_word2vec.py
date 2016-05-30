from __future__ import print_function
from gensim import corpora, models, matutils
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet, stopwords
from stop_words import get_stop_words
import re, nltk
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import logging
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

def lda():
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
    lemmatizer = WordNetLemmatizer()  #lemmatizer
    #stops = stopwords.words('english') # weaker stopwords
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
    with open('C:\Users\Norbert\Desktop\corpus.txt', 'w') as f:
        f.truncate()
        for line in doc_set:
            text = clean(line)
            f.write(text + '\n')
            texts.append(text)

    # create dictionary
    dictionary = corpora.Dictionary(line.lower().split() for line in open('C:\Users\Norbert\Desktop\corpus.txt', 'rb'))
    once_ids = [tokenid for tokenid, docfreq in dictionary.dfs.iteritems() if docfreq == 1]
    dictionary.filter_tokens(once_ids) # filter words that only appear once
    dictionary.filter_extremes(no_above=10, keep_n=100000)
    dictionary.compactify()

    # convert tokenized documents into a document-term matrix
    corpus = [dictionary.doc2bow(text.lower().split()) for text in texts]

    ### new code
    #count = CountVectorizer()
    #tfidf = TfidfTransformer()
    #np.set_printoptions(precision=2)

   # print(tfidf.fit_transform(count.fit_transform(texts)).toArray())



    KLvalues = []
    l = np.array([sum(cnt for _, cnt in doc) for doc in corpus])

    def sym_kl(p, q):
        return np.sum([stats.entropy(p, q), stats.entropy(q, p)])

    def arun(corpus, dictionary, min_topics, max_topics, step):
        kl = []
        flag = 0
        for i in range(min_topics, max_topics, step):
            lda = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=i, passes = 5)
            m1 = lda.expElogbeta
            print(m1)
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
            flag+=1
            kl.append(sym_kl(cm1, cm2))
        return kl

    # estimate topic number according to Arun Rajkumar's research(2010)
    kl = arun(corpus, dictionary, 1, 7, 1)

    # Plot kl divergence against number of topics
    print()
    print(KLvalues)
    plt.plot(kl)
    plt.ylabel('Symmetric KL Divergence')
    plt.xlabel('Number of Topics')
    plt.savefig('C:\Users\Norbert\Desktop\ldaKlDivergence.png', bbox_inches='tight')

    # find the proper parameter (minimize KL divergence value)
    import operator
    min_index, min_value = min(enumerate(KLvalues), key=operator.itemgetter(1))
    topicNum = min_index + 1

    print()
    print(topicNum)

    # generate LDA model
    ldaModel = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=topicNum)
    print(ldaModel.print_topics())
    print()
    #print(ldaModel.get_document_topics(corpus, minimum_probability=None))
    #print()
    #print(ldaModel.top_topics(corpus))

    lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=topicNum)
    print("LSI")
    print(lsi.print_topics(topicNum))

    hdp = models.HdpModel(corpus, id2word=dictionary)
    print("HDP")
    print(hdp.print_topics(topicNum))


def main():
    lda()

if __name__ == "__main__": main()
