import numpy as np
from scipy.stats import chi2
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from logger import setup_logger  # NOQA


__all__ = ('inverse_cor', 'estimate', 'deviance_and_p')


def inverse_cor(cor):
    cor_inv = np.linalg.inv(cor)
    diag = np.diag(1 / np.sqrt(np.diag(cor_inv)))
    return diag.dot(cor_inv).dot(diag)


def estimate(sample_cor, cond_ind_pairs, error_torelance=1e-4, verbose=1):
    '''Estimate correlation matrix from sample correlation and pairs of indices with conditional independence

    Parameters
    ----------
    sample_cor : 2d-array
        sample correlation matric
    arg2 : list of pairs of int
        list of pairs of indices.
        if (i, j) is in the list, this means that the i-th and j-th variables are conditionally independent given all the other variables.
    error_torelance : float
        torelance of error
    verbose : int
        verbose

    Returns
    -------
    estimated_cor : 2d-array
        estimated correlation matrix

    '''
    logger = setup_logger(__name__)
    dim = sample_cor.shape[0]
    sample_cor = np.array(sample_cor)

    estimated_cor = sample_cor.copy()
    error = 1
    if verbose > 0:
        counter = 0

    while error > error_torelance:
        if verbose > 0:
            logger.info('========', counter, '-th iteration', '========')
            counter = counter + 1

        estimated_cor_before = estimated_cor.copy()
        for i, j in cond_ind_pairs:
            estimated_cor_inv = np.linalg.inv(estimated_cor)
            new_ij = estimated_cor[i][j] + estimated_cor_inv[i][j] / (
                estimated_cor_inv[i][i] * estimated_cor_inv[j][j] - estimated_cor_inv[i][j]**2)
            estimated_cor[i][j] = new_ij
            estimated_cor[j][i] = new_ij

        error = np.abs(estimated_cor - estimated_cor_before).max().max()

    return estimated_cor


def deviance_and_p(original_cor, estimated_cor, df):
    dim = original_cor.shape[0]
    dev = dim * (np.log(np.linalg.det(estimated_cor)) -
                 np.log(np.linalg.det(original_cor)))
    p = 1 - chi2.cdf(dev, df)

    return dev, p
