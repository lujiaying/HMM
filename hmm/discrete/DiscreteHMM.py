'''
Created on Nov 12, 2012

@author: GuyZ
'''

from hmm._BaseHMM import _BaseHMM
import numpy

class DiscreteHMM(_BaseHMM):
    '''
    A Discrete HMM - The most basic implementation of a Hidden Markov Model,
    where each hidden state uses a discrete probability distribution for
    the physical observations.
    
    Model attributes:
    - n            number of hidden states
    - m            number of observable symbols
    - A            hidden states transition probability matrix ([NxN] numpy array)
    - B            PMFs denoting each state's distribution ([NxM] numpy array)
    - pi           initial state's PMF ([N] numpy array).
    
    Additional attributes:
    - precision    a numpy element size denoting the precision
    - verbose      a flag for printing progress information, mainly when learning
    '''

    def __init__(self,n,m,A=None,B=None,pi=None,init_type='uniform',precision=numpy.double,verbose=False):
        '''
        Construct a new Discrete HMM.
        In order to initialize the model with custom parameters,
        pass values for (A,B,pi), and set the init_type to 'user'.
        
        Normal initialization uses a uniform distribution for all probablities,
        and is not recommended.
        '''
        _BaseHMM.__init__(self,n,m,precision,verbose) #@UndefinedVariable
        
        self.A = A
        self.pi = pi
        self.B = B
        
        self.reset(init_type=init_type)

    def reset(self,init_type='uniform'):
        '''
        If required, initalize the model parameters according the selected policy
        '''
        if init_type == 'uniform':
            self.pi = numpy.ones( (self.n), dtype=self.precision) *(1.0/self.n)
            self.A = numpy.ones( (self.n,self.n), dtype=self.precision)*(1.0/self.n)
            self.B = numpy.ones( (self.n,self.m), dtype=self.precision)*(1.0/self.m)
    
    def _mapB(self,observations):
        '''
        Required implementation for _mapB. Refer to _BaseHMM for more details.
        '''
        self.B_map = numpy.zeros( (self.n,len(observations)), dtype=self.precision)
        
        for j in xrange(self.n):
            for t in xrange(len(observations)):
                self.B_map[j][t] = self.B[j][observations[t]]
                
    def _updatemodel(self,new_model):
        '''
        Required extension of _updatemodel. Adds 'B', which holds
        the in-state information. Specfically, the different PMFs.
        '''
        _BaseHMM._updatemodel(self,new_model) #@UndefinedVariable
        
        self.B = new_model['B']
    
    def _reestimate(self,stats,observations):
        '''
        Required extension of _reestimate. 
        Adds a re-estimation of the model parameter 'B'.
        '''
        # re-estimate A, pi
        new_model = _BaseHMM._reestimate(self,stats,observations) #@UndefinedVariable
        
        # re-estimate the discrete probability of the observable symbols
        B_new = self._reestimateB(observations,stats['gamma'])
        
        new_model['B'] = B_new
        
        return new_model
    
    def _reestimateB(self,observations,gamma):
        '''
        Helper method that performs the Baum-Welch 'M' step
        for the matrix 'B'.
        '''        
        # TBD: determine how to include eta() weighing
        B_new = numpy.zeros( (self.n,self.m) ,dtype=self.precision)
        
        for j in xrange(self.n):
            for k in xrange(self.m):
                numer = 0.0
                denom = 0.0
                for t in xrange(len(observations)):
                    if observations[t] == k:
                        numer += gamma[t][j]
                    denom += gamma[t][j]
                B_new[j][k] = numer/denom
        
        return B_new

    def score(self, observations):
        '''
        Compute the log probability under the model.

        Parameters
        ----------
        observations: array_like, shape(n, n_features=1)
          Sequence of n_feature-dimensional data points.
          Each row corresponds to a single data point.
          Current support n_features = 1

        Returns
        -------
        logprob: float
          Log likehood of the `observations`.
        '''
        observations = numpy.asarray(observations)

        # init stage - alpha_1(j) = pi(j)b_j_k1
        alpha_t = self.pi * self.B[:, observations[0]]

        # recursion
        for index in range(1, len(observations)):
            alpha_tn = numpy.dot(alpha_t, self.A) * self.B[:, observations[index]]
            alpha_t = alpha_tn

        # TODO: current not use log
        logprob = alpha_t.sum()
        return logprob


if __name__ == '__main__':
    n = 3
    m = 2
    A = numpy.array([[0.4, 0.6, 0], [0, 0.8, 0.2], [0, 0, 1]])
    B = numpy.array([[0.7, 0.3], [0.4, 0.6], [0.8, 0.2]])
    pi = numpy.array([1, 0, 0])

    hmm = DiscreteHMM(n, m, A, B, pi, init_type='user', verbose=True)
    print(hmm)

    result = hmm.score([0, 1, 0, 1])
    print(result)
