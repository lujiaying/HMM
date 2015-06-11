HMM
===

A numpy/python-only Hidden Markov Models framework. No other dependencies are required.

This implementation (like many others) is based on the paper:
"A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition, LR RABINER 1989"

Major supported features:

* Discrete HMMs
* Continuous HMMs - Gaussian Mixtures
* Supports a variable number of features
* Easily extendable with other types of probablistic models (simply override the PDF. Refer to 'GMHMM.py' for more information)
* Non-linear weighing functions - can be useful when working with a time-series

Open concerns:
* Examples are somewhat out-dated
* Convergence isn't guaranteed when using certain weighing functions 


-------------
Update by jiaying.lu

New features:

* Support DiscreteHMM score method to compute the log probability under the model.
    * Usage
    
    ```Python
    n = 3
    m = 2
    A = numpy.array([[0.4, 0.6, 0], [0, 0.8, 0.2], [0, 0, 1]])
    B = numpy.array([[0.7, 0.3], [0.4, 0.6], [0.8, 0.2]])
    pi = numpy.array([1, 0, 0])

    hmm = DiscreteHMM(n, m, A, B, pi, init_type='user', verbose=True)

    result = hmm.score([0, 1, 0, 1])
    ```
