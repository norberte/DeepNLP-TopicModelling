from __future__ import print_function
from gensim import models
import re, nltk


def import_data(fileName):
    import csv
    csvFile = open(fileName, 'r')
    reader = csv.reader(csvFile, delimiter=',', quotechar='"')
    my_list = list(reader)
    doc_set = []
    for item in my_list:
        doc_set.append(item[1])
    return doc_set

def pre_process(document):
    sentences = nltk.sent_tokenize(document)
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    sentences = [nltk.pos_tag(sent) for sent in sentences]

    namedEntity = nltk.ne_chunk(sentences, binary=True)
    print(namedEntity)


def entity_extraction(fileName):
    # import data-set
    documents = import_data(fileName)
    for doc in documents:
        pre_process(doc)





def main():
    # NLTK Text.collocation()
    # tokens = nltk.word_tokenize(text)
    # text = nltk.Text(tokens)
    # text.collocations()
    # text.concordance('airline')

    entity_extraction('C:\Users\Norbert\Desktop\work\Research 2016\Data sets\Air_Canada_data-set.csv')


if __name__ == "__main__": main()