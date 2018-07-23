"""
    experiment-v4.py
    
    BKJ modified optimizer, w/ updated hyperparameterS
    SGD + momentum algorithm here is identical to pytorch.optim.SGD
"""

import sys
import pickle
import numpy as np
from time import time
from functools import partial
from collections import defaultdict

from rsub import *
import matplotlib.pyplot as plt

sys.path.append('/home/bjohnson/software/autograd/')
sys.path.append('/home/bjohnson/software/hypergrad')

from funkyyak import grad, kylist, getval
from funkyyak import grad, Differentiable

from hypergrad.data import load_data_dicts
from hypergrad.util import RandomState
from hypergrad.nn_utils import VectorParser
from hypergrad.exact_rep import ExactRep

start_time = time()

# --
# Utils

def logit(x): 
    return 1 / (1 + np.exp(-x))

def inv_logit(x): 
    return -np.log(1 / x - 1)

def logsumexp(X, axis):
    max_X = np.max(X)
    return max_X + np.log(np.sum(np.exp(X - max_X), axis=axis, keepdims=True))

def fill_parser(parser, items):
    partial_vects = [np.full(parser[name].size, items[i]) for i, name in enumerate(parser.names)]
    return np.concatenate(partial_vects, axis=0)

# --
# Fixed params

layer_sizes = [784, 50, 50, 50, 10]
batch_size  = 200
N_iters     = 100
N_classes   = 10
N_train     = 10000
N_valid     = 10000
N_tests     = 10000

# --
# Initial values of learned hyper-parameters

init_log_L2_reg      = -100.0
init_log_alphas      = -1.0
init_invlogit_betas  = inv_logit(0.5)
init_log_param_scale = -3.0

# --
# Superparameters

seed = 0

# --
# Optimizers

def sgd_parsed(L_grad, hypers, parser, callback=None, forward_pass_only=True):
    x0, alphas, betas, meta = hypers
    X, V = ExactRep(x0), ExactRep(np.zeros(x0.size))
    iters = zip(range(len(alphas)), alphas, betas)
    
    for i, alpha, beta in iters:
        g = L_grad(X.val, meta, i)
        
        if callback:
            callback(X.val, V.val, g, i)
        
        cur_alpha_vect = fill_parser(parser, alpha)
        cur_beta_vect  = fill_parser(parser, beta)
        V.mul(cur_beta_vect).sub(g)
        X.add(cur_alpha_vect * V.val)
    
    x_final = X.val
    
    if forward_pass_only:
        return x_final
    
    # Hypergradient calculation
    def hypergrad(outgrad):
        d_x = outgrad
        d_alphas, d_betas = np.zeros(alphas.shape), np.zeros(betas.shape)
        d_v, d_meta = np.zeros(d_x.shape), np.zeros(meta.shape)
        
        grad_proj  = lambda x, meta, d, i: np.dot(L_grad(x, meta, i), d)
        L_hvp_x    = grad(grad_proj, 0)
        L_hvp_meta = grad(grad_proj, 1)
        
        for i, alpha, beta in iters[::-1]:
            
            # build alpha and beta vector
            cur_alpha_vect = fill_parser(parser, alpha)
            cur_beta_vect  = fill_parser(parser, beta)
            for j, (_, (ixs, _)) in enumerate(parser.idxs_and_shapes.iteritems()):
                d_alphas[i,j] = np.dot(d_x[ixs], V.val[ixs])
            
            # Exactly reverse SGD
            X.sub(cur_alpha_vect * V.val)
            g = L_grad(X.val, meta, i)
            V.add(g).div(cur_beta_vect)
            
            d_v += d_x * cur_alpha_vect
            
            for j, (_, (ixs, _)) in enumerate(parser.idxs_and_shapes.iteritems()):
                d_betas[i,j] = np.dot(d_v[ixs], V.val[ixs])
                
            d_x    -= L_hvp_x(X.val, meta, d_v, i)
            d_meta -= L_hvp_meta(X.val, meta, d_v, i)
            d_v    *= cur_beta_vect
        
        assert np.all(ExactRep(x0).val == X.val)
        return d_x, d_alphas, d_betas, d_meta
        
    return x_final, [None, hypergrad]


sgd_parsed = Differentiable(sgd_parsed, partial(sgd_parsed, forward_pass_only=False))


def adam(grad, x, callback=None, num_iters=100, step_size=0.1, b1=0.1, b2=0.01, eps=10**-4, lam=10**-4):
    m = np.zeros(len(x))
    v = np.zeros(len(x))
    for i in xrange(num_iters):
        b1t = 1 - (1 - b1) * (lam ** i)
        g = grad(x, i)
        
        if callback: 
            callback(x, i, g)
        
        m = b1t * g + (1-b1t) * m
        v = b2 * (g ** 2) + (1 - b2) * v
        mhat = m/(1 - (1 - b1) ** (i + 1))
        vhat = v/(1 - (1 - b2) ** (i + 1))
        x -= step_size * mhat / (np.sqrt(vhat) + eps)
    
    return x

# --
# Make NN functions

parser = VectorParser()
for i, shape in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
    parser.add_shape(('weights', i), shape)
    parser.add_shape(('biases', i), (1, shape[1]))


def pred_fun(W_vect, X):
    """Outputs normalized log-probabilities."""
    W = parser.new_vect(W_vect)
    cur_units = X
    N_iter = len(layer_sizes) - 1
    for i in range(N_iter):
        cur_W = W[('weights', i)]
        cur_B = W[('biases',  i)]
        cur_units = np.dot(cur_units, cur_W) + cur_B
        if i == (N_iter - 1):
            cur_units = cur_units - logsumexp(cur_units, axis=1)
        else:
            cur_units = np.tanh(cur_units)
    
    return cur_units


def loss_fun(W_vect, X, T, L2_reg=0.0):
    log_prior = -np.dot(W_vect * L2_reg, W_vect)
    log_lik = np.sum(pred_fun(W_vect, X) * T) / X.shape[0]
    return - log_prior - log_lik


def frac_err(W_vect, X, T):
    preds = np.argmax(pred_fun(W_vect, X), axis=1)
    return np.mean(np.argmax(T, axis=1) != preds)


# --
# Run

train_data, valid_data, test_data = load_data_dicts(N_train, N_valid, N_tests)

N_weight_types = len(parser.names)

hyperparams = VectorParser()
hyperparams['log_alphas']      = np.full((N_iters, N_weight_types), init_log_alphas)
hyperparams['invlogit_betas']  = np.full((N_iters, N_weight_types), init_invlogit_betas)

fixed_hyperparams = VectorParser()
fixed_hyperparams['log_param_scale'] = np.full(N_weight_types, init_log_param_scale)
fixed_hyperparams['log_L2_reg'] = np.full(N_weight_types, init_log_L2_reg)

cur_primal_results = {}

def primal_optimizer(hyperparams_vect, meta_epoch):
    
    def indexed_loss_fun(w, L2_vect, i_iter):
        rs = RandomState((seed, meta_epoch, i_iter)) # Deterministic seed needed for backwards pass.
        idxs = rs.randint(N_train, size=batch_size)
        return loss_fun(w, train_data['X'][idxs], train_data['T'][idxs], L2_vect)
    
    cur_hyperparams = hyperparams.new_vect(hyperparams_vect)
    
    rs = RandomState((seed, meta_epoch))
    
    # Randomly initialize weights
    W0 = fill_parser(parser, np.exp(fixed_hyperparams['log_param_scale'])) 
    W0 *= rs.randn(W0.size)
    # Init regularization term
    L2_reg = fill_parser(parser, np.exp(fixed_hyperparams['log_L2_reg']))
    # Set step sizes
    alphas = np.exp(cur_hyperparams['log_alphas'])
    # Momentum terms
    betas  = logit(cur_hyperparams['invlogit_betas'])
    
    # Train model
    W_opt = sgd_parsed(grad(indexed_loss_fun), kylist(W0, alphas, betas, L2_reg), parser)
    
    cur_primal_results['weights'] = getval(W_opt).copy()
    return W_opt

# hyperparams_vect = hyperparams.vect
# meta_epoch = 0

def hyperloss(hyperparams_vect, meta_epoch):
    W_opt = primal_optimizer(hyperparams_vect, meta_epoch)
    return loss_fun(W_opt, **train_data)


def meta_callback(hyperparams_vect, meta_epoch, metagrad=None):
    # !! Probably don't need
    cur_hyperparams = hyperparams.new_vect(hyperparams_vect.copy())
    for field in cur_hyperparams.names:
        meta_results[field].append(cur_hyperparams[field])
    
    # Compute metrics
    train_loss = loss_fun(cur_primal_results['weights'], **train_data)
    valid_loss = loss_fun(cur_primal_results['weights'], **valid_data)
    test_loss  = loss_fun(cur_primal_results['weights'], **test_data)
    test_err   = frac_err(cur_primal_results['weights'], **test_data)
    
    # Logging
    meta_results['train_loss'].append(train_loss)
    meta_results['valid_loss'].append(valid_loss)
    meta_results['test_loss'].append(test_loss)
    meta_results['test_err'].append(test_err)
    
    print "Meta Epoch {0} Train loss {1:2.4f} Valid Loss {2:2.4f} Test Loss {3:2.4f} Test Err {4:2.4f} Time {5:2.4f}".format(
        meta_epoch, 
        train_loss,
        valid_loss,
        test_loss,
        test_err,
        time() - start_time
    )


meta_alpha  = 0.04 # Meta-learning rate
meta_epochs = 50   # Number of epochs

meta_results = defaultdict(list)

final_result = adam(
    grad=grad(hyperloss), 
    x=hyperparams.vect, 
    callback=meta_callback, 
    num_iters=meta_epochs, 
    step_size=meta_alpha
)

meta_callback(final_result, meta_epochs)
parser.vect = None
pickle.dump((meta_results, parser), open('results.pkl', 'w'))

# --
# Plot

meta_results, parser = pickle.load(open('./results.pkl'))

for cur_results, name in zip(meta_results['log_alphas'][-1].T, parser.names):
    if name[0] == 'weights':
        _ = plt.plot(np.exp(cur_results), 'o-', label=name[1] + 1)


_ = plt.legend(loc='upper right')
show_plot()
