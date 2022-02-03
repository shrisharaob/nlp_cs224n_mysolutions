#!/usr/bin/env python
"""
## Word vectors
A mapping from a high dimensional discret vocabulary space $\mathbb{Y}$ to a lower dimensional continuous vector space. Each element $y_i$ of $\mathbb{Y}$ can be represented as a one hot vector of a every word in the vocabulary.
Each word $w_k$ in the vocabulary is assigned two word-vectors $(u_k, v_k) \in \mathbb{R}^{d \times 1}$, outside and center word-vector respectively where $d$ is the embedding dimension. 

$$f: \mathbb{Y} \rightarrow \mathbb{R}^{d \times 1}  $$


## Naive softmax
$W$: size of the vocabulary

$m$: dims of low-dim embedding

$ U, V \in \mathbb{R}^{m \times W}$, outside and center word vectors $ (u_k, v_k) \in  \mathbb{R}^{ m \times 1}$, one hot vector $y \in \mathbb{R}^{W \times 1}$, softmax prediction $\hat{y} \in \mathbb{R}^{W \times 1}$
     

Loss function for a given center word $v_c$: 
$$J = - \sum_{i=1}^W y_i log(\frac{exp(u_i^Tv_c)}{\sum_{w=1}^Wexp(u_w^Tv_c)})$$ 
Simplifying, 

$$ J = - \sum_{i=1}^Wy_i[u_i^Tv_c - log(\sum_{w=1}^Wexp(u_w^Tv_c))]; \qquad y_o = 1 $$ 
$y_k = 1$ if $k=o$ else $y_k = 0 \; \forall k \neq o$ (one hot encoding)

<a id="eq1"> $$\implies J(v_c, o, U) = - u_o^Tv_c + log(\sum_{w=1}^Wexp(u_w^Tv_c)) \quad (1)$$  </a>

#### Gradients of the softmax w.r.t the parameters of the model:

$$\frac{\partial J}{\partial v_c}  = U[\hat{y} -y]$$

$$\frac{\partial J}{\partial U}  = v_c [\hat{y} - y]^T$$

**The issue with naive softmax loss is the summation over all the vocabulary  [eq1](#eq1)**

## Negative sampling

The idea is to maximize the probability of the observed context word $u_o$ and minimize the likelihood of $K$ randomly chosen outside words from the vocabulary (negative samples $u_k$, acts as noise) i.e train binary logistic regression to give high prob to an observed outside word  versus noise words.


$$J = - log\left[\sigma(u_o^T v_c)\right] - \sum_{k=1}^{K} log\left[\sigma(-u_k^T v_c)\right] $$
Where, $\sigma(x) = \frac{1}{1 + e^{-x}}$

#### Gradients of Negative Sampling loss w.r.t model parameters (U, V)

$$\frac{\partial J}{\partial v_c} = \sigma(- u_o^T v_c) u_o + \sum_{k=1}^{K} \sigma(- u_k^T v_c) u_k$$

$$\frac{\partial J}{\partial u_o} = \sigma(- u_o^T v_c) v_c$$

$$\frac{\partial J}{\partial u_k} = \sigma(u_k^T v_c) v_c \quad \forall k \in {1...K}$$ 

**some useful identities:**

$\sigma(x) = (\sigma(-x) - 1)$, $\sigma^{\prime}(x) = \sigma(x) \, (1 - \sigma(x))$

### Skip-gram model:

Sliding window defines the contex for a given center word $c = w_t$. 
A context window of size $m$ given by $[w_{t-m}, ...,w_{t-1}, w_{t}, w{t+1}, ..., w_{t+m}]$.

#### Gradients of Skip-gram loss ($J_{sg}(v_c, w_{t-m}, ..., w{t+m}, U)$) with Neg. Sam.

$$\frac{\partial J_{sg}}{\partial U} = \sum_{j=-m, j \neq 0}^{j=m} \frac{\partial J(v_c, w_{t+j}, U)}{\partial U}$$
$$\frac{\partial J_{sg}}{\partial v_c} = \sum_{j=-m, j \neq 0}^{j=m} \frac{\partial J(v_c, w_{t+j}, U)}{\partial v_c} $$
$$\frac{\partial J_{sg}}{\partial v_w} = 0; \quad \text{when} \; w \neq c$$




"""

import numpy as np
import random

from utils.gradcheck import gradcheck_naive
from utils.utils import normalizeRows, softmax


def sigmoid(x):
    """
    Compute the sigmoid function for the input here.
    Arguments:
    x -- A scalar or numpy array.
    Return:
    s -- sigmoid(x)
    """

    ### YOUR CODE HERE
    s = 1 / (1 + np.exp(-x))
    ### END YOUR CODE

    return s


def naiveSoftmaxLossAndGradient(centerWordVec, outsideWordIdx, outsideVectors,
                                dataset):
    """ Naive Softmax loss & gradient function for word2vec models

    Implement the naive softmax loss and gradients between a center word's 
    embedding and an outside word's embedding. This will be the building block
    for our word2vec models.

    Arguments:
    centerWordVec -- numpy ndarray, center word's embedding
                    (v_c in the pdf handout)
    outsideWordIdx -- integer, the index of the outside word
                    (o of u_o in the pdf handout)
    outsideVectors -- outside vectors (rows of matrix) for all words in vocab
                      (U in the pdf handout)
    dataset -- needed for negative sampling, unused here.

    Return:
    loss -- naive softmax loss
    gradCenterVec -- the gradient with respect to the center word vector
                     (dJ / dv_c in the pdf handout)
    gradOutsideVecs -- the gradient with respect to all the outside word vectors
                    (dJ / dU)
    """

    ### YOUR CODE HERE

    ### Please use the provided softmax function (imported earlier in this file)
    ### This numerically stable implementation helps you avoid issues pertaining
    ### to integer overflow.
    '''
     W: size of the vocabulary
     m: dims of low-dim embedding
     U: W x m
     u_k, v_k: column vectors m x 1
     y: one hot vector W x 1
     y_hat: softmax prediction W x 1
    
     d_J / dv_c = U transose([y_hat - y])     shape: m x 1
     
    
    '''
    # u_o = outsideVectors[outsideWordIdx]
    arg = outsideVectors @ centerWordVec  # W x 1 = (W x m) @ (m x 1)
    y_hat = softmax(arg)  # (W x 1)
    loss = -1 * np.log(y_hat[outsideWordIdx])  # (W x 1)

    print('arg shape = ', arg.shape)

    y_hat_copy = y_hat.copy()
    # (y_hat - y) speed up by updating only for contex words
    # y is the true one-hot vec for observed outside word, so minus one below
    y_diff[outsideWordIdx] = y_hat_copy[outsideWordIdx] - 1 
    gradCenterVec = outsideVectors.T @ y_diff  # dJ_dv_c (m x 1) = (m x W) @ (W x 1)
    gradOutsideVecs = np.outer(
        y_hat, centerWordVec)  # y @ v^T # (W x m) = (W x m) x (m x 1)

    ### END YOUR CODE

    return loss, gradCenterVec, gradOutsideVecs


def getNegativeSamples(outsideWordIdx, dataset, K):
    """ Samples K indexes which are not the outsideWordIdx """

    negSampleWordIndices = [None] * K
    for k in range(K):
        newidx = dataset.sampleTokenIdx()
        while newidx == outsideWordIdx:
            newidx = dataset.sampleTokenIdx()
        negSampleWordIndices[k] = newidx
    return negSampleWordIndices


def negSamplingLossAndGradient(centerWordVec,
                               outsideWordIdx,
                               outsideVectors,
                               dataset,
                               K=10):
    """ Negative sampling loss function for word2vec models

    Implement the negative sampling loss and gradients for a centerWordVec
    and a outsideWordIdx word vector as a building block for word2vec
    models. K is the number of negative samples to take.

    Note: The same word may be negatively sampled multiple times. For
    example if an outside word is sampled twice, you shall have to
    double count the gradient with respect to this word. Thrice if
    it was sampled three times, and so forth.

    Arguments/Return Specifications: same as naiveSoftmaxLossAndGradient
    """

    # Negative sampling of words is done for you. Do not modify this if you
    # wish to match the autograder and receive points!
    negSampleWordIndices = getNegativeSamples(outsideWordIdx, dataset, K)
    indices = [outsideWordIdx] + negSampleWordIndices

    ### YOUR CODE HERE
    ### Please use your implementation of sigmoid in here.
    """
    $$\frac{\partial J}{\partial v_c} = \sigma(- u_o^T v_c) u_o + \sum_{k=1}^{K} \sigma(- u_k^T v_c) u_k$$
    $$\frac{\partial J}{\partial u_o} = \sigma(- u_o^T v_c) v_c$$
    $$\frac{\partial J}{\partial u_k} = \sigma(u_k^T v_c) v_c \quad \forall k \in {1...K}$$ 
    """
    #
    gradCenterVec = np.zeros(centerWordVec.shape)  # m x 1
    gradOutsideVecs = np.zeros(outsideVectors.shape)  # W x m 

    #
    u_o = outsideVectors[outsideWordIdx]
    z = u_o.T @ centerWordVec # (1, 1)
    sigma_z = sigmoid(z)
    dJ_vc_term0 = 
    #
    loss = -1 * np.log(sigma_z)
    gradCenterVec = sigma_z @ u_o

    #
    for k in range(K):
        z_out = outsideVectors[]
        loss -= np.log(sigma_z)
        
    
    

    ### END YOUR CODE

    return loss, gradCenterVec, gradOutsideVecs


def skipgram(currentCenterWord,
             windowSize,
             outsideWords,
             word2Ind,
             centerWordVectors,
             outsideVectors,
             dataset,
             word2vecLossAndGradient=naiveSoftmaxLossAndGradient):
    """ Skip-gram model in word2vec

    Implement the skip-gram model in this function.

    Arguments:
    currentCenterWord -- a string of the current center word
    windowSize -- integer, context window size
    outsideWords -- list of no more than 2*windowSize strings, the outside words
    word2Ind -- a dictionary that maps words to their indices in
              the word vector list
    centerWordVectors -- center word vectors (as rows) for all words in vocab
                        (V in pdf handout)
    outsideVectors -- outside word vectors (as rows) for all words in vocab
                    (U in pdf handout)
    word2vecLossAndGradient -- the loss and gradient function for
                               a prediction vector given the outsideWordIdx
                               word vectors, could be one of the two
                               loss functions you implemented above.

    Return:
    loss -- the loss function value for the skip-gram model
            (J in the pdf handout)
    gradCenterVecs -- the gradient with respect to the center word vectors
            (dJ / dV in the pdf handout)
    gradOutsideVectors -- the gradient with respect to the outside word vectors
                        (dJ / dU in the pdf handout)
    """

    loss = 0
    gradCenterVecs = np.zeros(centerWordVectors.shape)
    gradOutsideVectors = np.zeros(outsideVectors.shape)

    ### YOUR CODE HERE

    ### END YOUR CODE

    return loss, gradCenterVecs, gradOutsideVectors


#############################################
# Testing functions below. DO NOT MODIFY!   #
# ############################################


def word2vec_sgd_wrapper(word2vecModel,
                         word2Ind,
                         wordVectors,
                         dataset,
                         windowSize,
                         word2vecLossAndGradient=naiveSoftmaxLossAndGradient):
    batchsize = 50
    loss = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0]
    centerWordVectors = wordVectors[:int(N / 2), :]
    outsideVectors = wordVectors[int(N / 2):, :]
    for i in range(batchsize):
        windowSize1 = random.randint(1, windowSize)
        centerWord, context = dataset.getRandomContext(windowSize1)

        c, gin, gout = word2vecModel(centerWord, windowSize1, context,
                                     word2Ind, centerWordVectors,
                                     outsideVectors, dataset,
                                     word2vecLossAndGradient)
        loss += c / batchsize
        grad[:int(N / 2), :] += gin / batchsize
        grad[int(N / 2):, :] += gout / batchsize

    return loss, grad


def test_word2vec():
    """ Test the two word2vec implementations, before running on Stanford Sentiment Treebank """
    dataset = type('dummy', (), {})()

    def dummySampleTokenIdx():
        return random.randint(0, 4)

    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0,4)], \
            [tokens[random.randint(0,4)] for i in range(2*C)]

    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalizeRows(np.random.randn(10, 3))
    dummy_tokens = dict([("a", 0), ("b", 1), ("c", 2), ("d", 3), ("e", 4)])



    print(
        "==== Gradient check for skip-gram with naiveSoftmaxLossAndGradient ===="
    )
    gradcheck_naive(
        lambda vec: word2vec_sgd_wrapper(skipgram, dummy_tokens, vec, dataset,
                                         5, naiveSoftmaxLossAndGradient),
        dummy_vectors, "naiveSoftmaxLossAndGradient Gradient")

    print(
        "==== Gradient check for skip-gram with negSamplingLossAndGradient ===="
    )
    gradcheck_naive(
        lambda vec: word2vec_sgd_wrapper(skipgram, dummy_tokens, vec, dataset,
                                         5, negSamplingLossAndGradient),
        dummy_vectors, "negSamplingLossAndGradient Gradient")

    print("\n=== Results ===")
    print("Skip-Gram with naiveSoftmaxLossAndGradient")

    print("Your Result:")
    print(
        "Loss: {}\nGradient wrt Center Vectors (dJ/dV):\n {}\nGradient wrt Outside Vectors (dJ/dU):\n {}\n"
        .format(
            *skipgram("c", 3, ["a", "b", "e", "d", "b", "c"], dummy_tokens,
                      dummy_vectors[:5, :], dummy_vectors[5:, :], dataset)))

    print("Expected Result: Value should approximate these:")
    print("""Loss: 11.16610900153398
Gradient wrt Center Vectors (dJ/dV):
 [[ 0.          0.          0.        ]
 [ 0.          0.          0.        ]
 [-1.26947339 -1.36873189  2.45158957]
 [ 0.          0.          0.        ]
 [ 0.          0.          0.        ]]
Gradient wrt Outside Vectors (dJ/dU):
 [[-0.41045956  0.18834851  1.43272264]
 [ 0.38202831 -0.17530219 -1.33348241]
 [ 0.07009355 -0.03216399 -0.24466386]
 [ 0.09472154 -0.04346509 -0.33062865]
 [-0.13638384  0.06258276  0.47605228]]
    """)

    print("Skip-Gram with negSamplingLossAndGradient")
    print("Your Result:")
    print(
        "Loss: {}\nGradient wrt Center Vectors (dJ/dV):\n {}\n Gradient wrt Outside Vectors (dJ/dU):\n {}\n"
        .format(*skipgram("c", 1, ["a", "b"], dummy_tokens,
                          dummy_vectors[:5, :], dummy_vectors[
                              5:, :], dataset, negSamplingLossAndGradient)))
    print("Expected Result: Value should approximate these:")
    print("""Loss: 16.15119285363322
Gradient wrt Center Vectors (dJ/dV):
 [[ 0.          0.          0.        ]
 [ 0.          0.          0.        ]
 [-4.54650789 -1.85942252  0.76397441]
 [ 0.          0.          0.        ]
 [ 0.          0.          0.        ]]
 Gradient wrt Outside Vectors (dJ/dU):
 [[-0.69148188  0.31730185  2.41364029]
 [-0.22716495  0.10423969  0.79292674]
 [-0.45528438  0.20891737  1.58918512]
 [-0.31602611  0.14501561  1.10309954]
 [-0.80620296  0.36994417  2.81407799]]
    """)


if __name__ == "__main__":
    test_word2vec()

""

