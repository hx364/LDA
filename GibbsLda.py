import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error
from scipy.special import psi
import matplotlib.pyplot as plt

class LDA():
    def __init__(self):
        self.nTopics = 1
        self.nVocab = 0
        self.alpha = np.zeros(self.nTopics)
        self.beta = np.zeros((self.nVocab, self.nTopics))
        self.vocabDic = {}
        self.true_theta = np.zeros(self.nTopics)
        self.true_theta = joblib.load("ps4_data-3/theta_expectation.dump")
        self.error = []
        self.burn_period = 50
        
    def read_par(self, file_name):
        f = open(file_name)
        self.nTopics = int(f.readline().strip())
        self.alpha = np.array([float(i) for i in f.readline().strip().split()])
        self.nTopics = self.alpha.shape[0]
        f.close()
        
        f = pd.read_csv(file_name, skiprows=2, delimiter=' ', names = range(202))
        dict0 = f[0].to_dict()
        self.vocabDic = dict((key, value) for (value, key) in dict0.iteritems())
        self.nVocab = len(self.vocabDic)
        self.beta = np.array(f.icol(range(1,201)))
    
    def SetTrueTheta(self, l):
        self.true_theta = l
    
    def __get_error__(self):
        return self.error
    
    def GibbsLDA(self, nIter):
        # Gibbs sampling for LDA
        """
        #initialize z
        #z: the topic for each word in the doc
        #z: an array of length nVocab
        """
        self.z = np.random.randint(0, self.nTopics, self.nVocab)
        """
        #initialize n corresponded
        # n_k: number of words with topic k in doc
        # n: an array of length nTopics
        """
        self.n = [len(self.z[self.z == i]) for i in range(self.nTopics)]
        self.theta_expectation = np.zeros(self.nTopics)
        self.error=[]
        
        for i in range(nIter):
            for word_index in range(self.nVocab):
                """update theta using dirichlet sample
                   theta: array of size nTopics"""
                self.theta = np.random.dirichlet((self.alpha + self.n))
                
                """update z at word_index"""
                """first update n"""
                self.n[self.z[word_index]] -=1
                
                z_dist = np.zeros(self.nTopics)
                
                z_dist = self.theta*self.beta[word_index,:]
                z_dist = z_dist/sum(z_dist)
                
                """sample z, update n"""
                new_z = np.random.multinomial(1, z_dist)
                self.n = self.n + new_z
                self.z[word_index] = new_z.argmax()
            if i >= self.burn_period:
                self.theta_expectation = self.theta_expectation + self.alpha + self.n
                if i%10 == 0:
                    current_theta = self.theta_expectation/sum(self.theta_expectation)
                    print i, np.max(current_theta), np.min(current_theta), np.mean(current_theta)
                self.error.append(mean_squared_error(self.theta_expectation/sum(self.theta_expectation), self.true_theta)*self.nTopics)
                
                
    def CollapsedGibbsLDA(self, nIter):
        # Collapsed Gibbs sampling for LDA
        """
        #initialize z
        #z: the topic for each word in the doc
        #z: an array of length nVocab
        """
        self.z = np.random.randint(0, self.nTopics, self.nVocab)
        
        """
        #initialize n corresponded
        # n_k: number of words with topic k in doc
        # n: an array of length nTopics
        """
        self.n = [len(self.z[self.z == i]) for i in range(self.nTopics)]
        self.theta_expectation = np.zeros(self.nTopics)
        self.error = []
           
        
        for i in range(nIter):
            for word_index in range(self.nVocab):
                """update z at word_index"""
                """first update n"""
                self.n[self.z[word_index]] -=1
                
                #update z's distribution
                z_dist = np.zeros(self.nTopics)
                z_dist = (self.n+self.alpha)*self.beta[word_index,:]
                #normalize z_dist
                z_dist = z_dist/sum(z_dist)
                
                new_z = np.random.multinomial(1, z_dist)
                self.n = self.n + new_z
                self.z[word_index] = new_z.argmax()
            
            if i >= self.burn_period:
                self.theta_expectation = self.theta_expectation + self.alpha + self.n
                if i%10 == 0:
                    current_theta = self.theta_expectation/sum(self.theta_expectation)
                    print i, np.max(current_theta), np.min(current_theta), np.mean(current_theta)
                self.error.append(mean_squared_error(self.theta_expectation/sum(self.theta_expectation), self.true_theta)*self.nTopics)
             
        self.theta_expectation = self.theta_expectation/sum(self.theta_expectation)
        return self.theta_expectation
    
    
    def mean_field(self, nIter):
        #Mean field algorithm for LDA
        self.error = []
        """
        initialize phi and gamma
        self.phi: parameter of q(z|phi), multinoial
                  matrix of size(nVocab, nTopics)
        self.gamma: parameter of q(theta|gamma), dirichlet
                 array of size(nTopics)
        """
        self.phi = np.ones((self.nVocab, self.nTopics))/self.nTopics
        self.gamma = self.alpha + float(self.nVocab)/self.nTopics
        
        for i in range(nIter):
            for word_index in range(self.nVocab):
                #update phi
                current_phi = self.beta[word_index, :]*np.exp(psi(self.gamma) - psi(self.gamma.sum()))
                self.phi[word_index,:] = current_phi/sum(current_phi)
            
            self.gamma = self.alpha + np.sum(self.phi, axis=0) #update gamma
            norm_gamma = self.gamma/sum(self.gamma)
            self.error.append(mean_squared_error(norm_gamma, self.true_theta)*self.nTopics)
            if i%10 == 0:
                print i, np.max(norm_gamma), np.min(norm_gamma), np.mean(norm_gamma)
