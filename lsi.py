from __future__ import print_function
from gensim import corpora, models
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet, stopwords
from stop_words import get_stop_words
import re, nltk
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import logging


def lsi():
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
    lsi = models.lsimodel.LsiModel(corpus=corpus, id2word=dictionary, num_topics=20)
    #print("lsi corpus")
    #print(lsi[corpus])  # project some document into LSI space
    #print("lsi topics")
    #print(lsi.print_topics())
    print()
    print(lsi.show_topics(num_topics=-1, num_words=10, log=False, formatted=True))

    #print(lsi.print_topics(4))


def main():
    lsi()

if __name__ == "__main__": main()