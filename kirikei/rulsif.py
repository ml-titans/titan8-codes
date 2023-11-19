import numpy as np
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import mean_squared_error
from itertools import product


def compute_gaussian_kernel(x1, x2, sigma):
    """calculate Gussian Kernel
    
    Args:
        x1, x2(np.ndarray)
    Return:
        np.ndarray(x1.shape[0], x2.shape[0])
        - x_ij: k(x1_i, x2_j)
    
    """
    kernel = RBF(length_scale=sigma)
    return kernel(x1, x2)


def compute_w_alpha_x(x, xdash, alpha, sigma=1.0):
    """calc w_alpha(x)

    Args:
        x(np.ndarray): train data
        xdash(np.ndarray): test data
        alpha(np.ndarray): weight of training data respect to test data
    """
    
    w_alpha = alpha @ compute_gaussian_kernel(x, xdash, sigma).T # (ndash) @ (ndash, n)

    return w_alpha # (n)


def calculate_J(alpha_hat, h_hat, G_hat, lda):
    """calculate J(\alpha)
    """ 
    J = alpha_hat.T @ G_hat @ alpha_hat - 2 * h_hat.T @ alpha_hat + lda * alpha_hat.T @ alpha_hat
    return J


def estimate_alpha(x, xdash, beta, lda, sigma):
    """estimate alpha_hat

    Args:
        x(np.ndarray): train data
        xdash(np.ndarray): test data
        beta(float[0, 1]): relative coefficient. 
            0 means p(x) and p(x') is the same.

    """
    ndash = xdash.shape[0]
    n = x.shape[0]
    
    phi_x = compute_gaussian_kernel(x, xdash, sigma) # (n, ndash)
    phi_xdash = compute_gaussian_kernel(xdash, xdash, sigma) # (ndash, ndash)

    G_hat = ((beta / ndash) * phi_xdash.T @ phi_xdash) + \
                        (((1-beta) / n) * phi_x.T @ phi_x)
    
    assert G_hat.shape == (ndash, ndash) , f'{G_hat.shape}'

    h_hat = phi_xdash.mean(axis=0)
    assert h_hat.shape[0] == ndash, f'{h_hat.shape}, {ndash}'

    I = np.identity(G_hat.shape[0])

    alpha_hat = np.linalg.inv((G_hat + lda * I)) @ h_hat

    # 0以上でないとエラーになるので、マイナスになったものは0で埋める
    alpha_hat[alpha_hat < 0] = 0
    assert (alpha_hat >= 0).all(), f'beta = {beta}, lda = {lda}'
    
    return alpha_hat


def search_min_score(x, y, xdash, sigma_range, lda_range, beta_range, cv=None, model=None):
    """search hyper-parameters by importance weighted cross validation

    Args:
        x(np.ndarray): explanatory variables of train data
        y(np.ndarray): an objective variable of train data
        xdash(np.ndarray): explanatory variables of test data
        sigma_range(List): range of sigma of RBF Kernel
        lda_range(List): range of lambda of regularization strength
        beta_range(List): range of relative coefficient
        cv(sklearn.model_selection): a cross validation instance such as KFold, TimeseriesValidation.
        model(sklearn.models): a model instance for calc validation score such as Ridge, Lasso
    """
    
    score = np.inf
    alpha_hat = None
    min_sigma = None
    min_lda = None
    min_beta = None
    
    for sigma, lda, beta in product(sigma_range, lda_range, beta_range):
        
        sum_score = 0
        
        # w(x) by xdash
        alpha_hat_dash = estimate_alpha(x, xdash, beta, lda, sigma)
        
        if sum(alpha_hat_dash) < 1e-7:
            print('sum of alpha is almost zero. skip this iteration')
            continue
        
        # start validation
        for train_index, valid_index in cv.split(x):
            xtr = x[train_index]
            xvl = x[valid_index]
            ytr = y[train_index]
            yvl = y[valid_index]
            
            # calc w_alpha by tr data and w(x)_test
            w_alpha_tmp = compute_w_alpha_x(xtr, xdash, alpha_hat_dash, sigma)
            
            if sum(w_alpha_tmp) < 1e-7:
                print('sum of train w alpha is almost zero. skip this iteration')
                sum_score = np.inf
                break
            
            model.fit(xtr, ytr, sample_weight=w_alpha_tmp)
            yhat = model.predict(xvl)
            
            # calc w_alpha by val data and w(x)_test
            w_alpha_v_tmp = compute_w_alpha_x(xvl, xdash, alpha_hat_dash, sigma)
            
            if sum(w_alpha_v_tmp) < 1e-7:
                print('sum of valid w alpha is almost zero. skip this iteration')
                sum_score = np.inf
                break
            
            sum_score += mean_squared_error(yvl, yhat, sample_weight=w_alpha_v_tmp)
            
        mean_score = sum_score / cv.n_splits

        if mean_score < score:
            score = mean_score
            min_sigma = sigma
            min_lda = lda
            min_beta = beta
            alpha_hat = alpha_hat_dash
        
        if mean_score != np.inf:
            print(f'sigma = {sigma}, lda = {lda}, beta = {beta}, validation score = {mean_score}')
        
    print('-------')
    print(f'Result: sigma = {min_sigma}, lda = {min_lda}, beta = {min_beta}, validation score = {score}')
        
    return min_sigma, min_lda, min_beta, alpha_hat