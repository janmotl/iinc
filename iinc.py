import numpy as np
import scipy.spatial

def iinc(X, y, X_t, prior_weight='raw'):
    nrow = np.size(X, 0)
    nrow_t = np.size(X_t, 0)
    nclasses = len(np.unique(y))

    # Prior normalization one-vs-rest
    prior = np.zeros(nclasses, float)
    for row_t in range(nrow):
        prior[y[row_t]] += 1
    prior = prior / len(y)
    prior_ovr = 1.-prior

    # Prior normalization one-vs-one
    prior_A = np.zeros((nclasses, nclasses))
    for c in range(nclasses-1):
        prior_A[c, c] = prior[c]
        prior_A[c, c + 1] = -prior[c+1]
    prior_A[-1, :] = 1

    prior_b = np.zeros(nclasses)
    prior_b[-1] = 1

    prior_ovo = np.linalg.solve(prior_A, prior_b)

    # Score testing data
    d = scipy.spatial.distance_matrix(X_t, X)
    probabilities = np.zeros((nrow_t, nclasses))

    for row_t in range(nrow_t):
        indexes = np.argsort(d[row_t, :])

        for row in range(nrow):
            if prior_weight == 'raw':
                w = 1.
            elif prior_weight == 'ovr':
                w = prior_ovr[y[indexes[row]]]
            elif prior_weight == 'ovo':
                w = prior_ovo[y[indexes[row]]]
            else:
                raise Exception

            probabilities[row_t, y[indexes[row]]] += w * (1. / (row+1.))

    # Normalize
    probabilities = (probabilities.T / np.sum(probabilities, axis=1)).T

    # Prediction
    prediction = np.argmax(probabilities, axis=1)

    return probabilities, prediction
