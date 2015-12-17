__author__ = 'hxu'
import numpy as np
from collections import Counter
from gensim import corpora
from scipy.sparse import lil_matrix

import pandas as pd
DOCUMENT = ["Human machine interface for lab abc computer applications",
              "A survey of user opinion of computer system response time",
              "The EPS user interface management system",
              "System and human system engineering testing of EPS",
              "Relation of user perceived response time to error measurement",
              "The generation of random binary unordered trees",
              "The intersection graph of paths in trees",
              "Graph minors IV Widths of trees and well quasi ordering",
              "Graph minors A survey"]

LABELS = ['interface', 'system', 'system', 'system', 'interface', 'interface', 'graph', 'graph']



class Labelled_LDA():
    def __init__(self):
        """
        self.nDocs
        self.nTopics
        self.nVocab
        self.alpha #parameter of dirichlet prior for doc-topic matrix
        self.beta #parameter of dirichlet prior for topic-word matrix
        self.theta #doc-topic matrix, of size(nDocs, nTopics)
        self.phi #topic-word matrix, of size(nTopics, nVocab)
        self.z #topic for word in a doc, of size(nDocs, nVocab)
        self.w #matrix, indicator of word in a doc, of size(nDocs, nVocab)
        self.dictionary #index to word mapping
        """

    def read_text(self, docs, labels):
        """
        :param docs: a list of texts
        :param labels: a list of labels
        :return: None
        """
        #read the texts and transform to matrix
        texts = [[i for i in doc.lower().split()] for doc in docs]
        word_counter = Counter()
        for i in texts:
            word_counter.update(i)
        texts = [[i for i in doc if word_counter[i] >=1] for doc in texts]

        self.dictionary = corpora.Dictionary(texts)
        w = [self.dictionary.doc2bow(text) for text in texts]
        corpora.MmCorpus.serialize('corpus.mm', w)
        self.w = corpora.MmCorpus('corpus.mm')

        #read the labels and trnasform to matrix
        label_Counter = Counter(labels)
        label_dict = dict(zip(label_Counter.keys(), range(1,1+len(label_Counter))))

        #initialize the parameter
        self.nDocs = self.w.num_docs
        self.nVocab = self.w.num_terms


    def unsup_train(self, num_topic, num_Iteration):
        #Traditional LDA, use unsupervised way of training
        self.nTopics = num_topic

        #TODO: currently just use uniform parameter for alpha and beta
        self.alpha = np.ones(self.nTopics)/self.nTopics
        self.beta = np.ones(self.nVocab)/self.nVocab
        self.z = lil_matrix((self.nDocs, self.nVocab))

        doc_topic_count = np.zeros((self.nDocs, self.nTopics))
        topic_word_count = np.zeros((self.nTopics, self.nVocab))

        #Initialization
        for doc_index in range(len(self.w)):
            doc_vec = self.w[doc_index]
            for word_index, word_count in doc_vec:
                self.z[doc_index, word_index] = np.argmax(np.random.multinomial(1, self.alpha))+1

        #Burn in period
        print self.z.todense()
        doc_topic_count = np.matrix([[j[j==i].shape[1] for i in range(1, 1+self.nTopics)] for j in self.z.todense()])
        word_topic_count = np.matrix([[j[j==i].shape[1] for i in range(1, 1+self.nTopics)] for j in self.z.todense().T])
        topic_word_count = word_topic_count.T

        print doc_topic_count
        print topic_word_count

        current_alpha = np.ones(self.nTopics)/self.nTopics
        current_beta = np.ones(self.nVocab)/self.nVocab


        n_iter = 0
        while (True):
            n_iter+=1
            for doc_index in range(len(self.w)):
                doc_vec = self.w[doc_index]
                for word_index, word_count in doc_vec:
                    topic_index = self.z[doc_index, word_index]
                    doc_topic_count[doc_index, topic_index] -= word_count
                    topic_word_count[topic_index, word_index] -= word_count

                    #compute the posterior dist and resample
                    current_alpha = self.alpha + doc_topic_count[doc_index]
                    current_alpha = current_alpha/np.sum(current_alpha)
                    current_beta = self.beta[word_index] + word_topic_count[word_index]
                    current_beta = current_beta/np.sum(current_beta)
                    current_p = np.multiply(current_alpha, current_beta)
                    current_p = np.squeeze(np.asarray(current_p/np.sum(current_p)))
                    new_topic_index = np.argmax(np.random.multinomial(1, current_p))+1

                    #reassign new value
                    self.z[doc_index, word_index] = new_topic_index
                    doc_topic_count[doc_index, topic_index] += word_count
                    topic_word_count[topic_index, word_index] += word_count

                # print "Iter %d, Doc %d " %(n_iter, doc_index)
                # print doc_topic_count
                # print topic_word_count

            if n_iter>num_Iteration:
                break

        #read out the phi and theta parameters
        self.theta = doc_topic_count + np.repeat([self.alpha], self.nDocs, axis=0)
        self.theta = self.theta/np.sum(self.theta, axis=1)
        self.phi = word_topic_count + np.repeat([self.beta], self.nTopics, axis=0).T
        self.phi = self.phi/np.sum(self.phi, axis=1)
        print self.theta
        print self.phi



def main():
    s = Labelled_LDA()
    s.read_text(DOCUMENT,LABELS)
    s.unsup_train(4, 20)

if __name__ == "__main__":
    main()














