from __future__ import print_function
from gensim import corpora, models, matutils
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from stop_words import get_stop_words
import re, nltk
import numpy as np
import logging


def hdp():
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
    dictionary.filter_tokens(once_ids)  # filter words that only appear once
    dictionary.filter_extremes(no_above=10, keep_n=100000)
    dictionary.compactify()

    # convert tokenized documents into a document-term matrix
    corpus = [dictionary.doc2bow(text.lower().split()) for text in texts]

    # generate LDA model
    hdp = models.HdpModel(corpus, dictionary)


    print("hdp")
    print(hdp.print_topics())
    print("hdp to lda")
    print(hdp.hdp_to_lda())
    print("optimal ordering")
    print(hdp.optimal_ordering())

def main():
    hdp()

if __name__ == "__main__": main()