import torch
import math
import pyro
import pyro.infer
import pyro.optim
import pyro.distributions as dist
import math
import torch
from torch import distributions

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.linear_model import BayesianRidge
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO

# data
t_sampler = dist.Bernoulli(0.5).sample([1000])
x_sampler = dist.Normal(0,1).sample([1000])

# must generate fake data by specifying known values of b0, b1 etc to generate real u and y
real_b0 = 0.70
real_b1 = 1.4
real_lambda = 0.1
real_gamma0 = 0.5
real_gamma1 = 0.4
real_eta = 0.1
real_E = 0.2
def return_y_logits(t, b0, b1, lamb, eta, u, x):
  return b0 + b1* t + lamb*u + eta*x

def return_u_logits(t, gamma0, gamma1, E, x):
  return gamma0 + gamma1*t + E*x

# once we have ground truth data, we can start to try and guess the parameters using guide and model

u_logits = return_u_logits(t_sampler, real_gamma0, real_gamma1, real_E, x_sampler)
real_u = dist.Bernoulli(logits=u_logits).sample()

y_logits = return_y_logits(t_sampler, real_b0, real_b1, real_lambda, real_eta, real_u, x_sampler)
real_y = dist.Bernoulli(logits=y_logits).sample()

def return_y_coeffs(x, t,  y, u):
    model = LinearRegression()
    X = torch.cat([x.unsqueeze(-1), t.unsqueeze(-1), u.unsqueeze(-1)], dim=-1)
    model.fit(X=X, y=y)
    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)
    b0 = model.intercept_
    eta = model.coef_[0]
    b1 = model.coef_[1]
    lamb = model.coef_[2]
    # figure out correct order here!!!
    return b0, b1, lamb, eta

def return_u_coeffs(x, t,  u):
    model = LinearRegression()
    import pdb
    pdb.set_trace
    X = torch.cat([x.unsqueeze(-1), t.unsqueeze(-1)], dim=-1)
    model.fit(X=X, y=u)
    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)
    E = model.coef_[1]
    gamma1 = model.coef_[0]
    gamma0 = model.intercept_
    return gamma0, gamma1, E

gammas = dist.MultivariateNormal(torch.tensor([0.0,0.0]), torch.tensor([[1.0,0.0], [0.0,1.0]])).sample()

def model(data):
    # starting values for params
    # a bunch of p_u's that is determined by latent variables gamma0 etc
    # each data point x, t has a different u
    t, x, y = data[0], data[1], data[2]
    beta0 = torch.tensor(0.5)
    beta1 = torch.tensor(1.2)
    eta = torch.tensor(1.0)
    gamma0 = gammas[0]
    gamma1 = gammas[1]
    lamb = dist.Normal(loc=0.0, scale=math.log(6) / 1.96).sample()
    E = dist.Normal(loc=0.0, scale=math.log(6) / 1.96).sample()

    with pyro.plate("data", t.shape[0]):
        u_logits = torch.zeros(t.shape[0])
        p_u = pyro.sample("p_u", dist.Bernoulli(logits=u_logits))
        y_logits = return_y_logits(t, beta0, beta1, lamb, eta, p_u, x)
        pyro.sample("obs", dist.Bernoulli(logits=y_logits), obs=y)

def guide(data):
    # the guide samples latent variables from the prior,
    # priors for u
    # need to do prior.param etc
    t, x, y = data[0], data[1], data[2]
    # specify all latent parameters
    gamma0 = pyro.param("gamma0", torch.tensor(1.0))
    gamma1 = pyro.param("gamma1", torch.tensor(0.2))
    lamb = pyro.param("lamb", torch.tensor(0.3))
    eta = pyro.param("eta", torch.tensor(1.3))
    beta0 = pyro.param("beta0", torch.tensor(0.2))
    beta1 = pyro.param("beta1", torch.tensor(0.3))
    E = pyro.param("E", torch.tensor(0.3))
    with pyro.plate("data", t.shape[0]):
        u_logits = return_u_logits(t, gamma0, gamma1, E, x)
        pyro.sample("p_u", dist.Bernoulli(logits=u_logits))


# need to figure out where the model fitting comes in.
# probably in the model()
adam_params = {"lr": 0.1, "betas": (0.90, 0.999)}
optimizer = Adam(adam_params)

# setup the inference algorithm
svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

# do gradient steps
n_steps=1000
data = (t_sampler, x_sampler, real_y)
for step in range(n_steps):
    svi.step(data)
    if step % 100 == 0:
        print('.', end='')

# grab the learned variational parameters
beta_0 = pyro.param("beta0").item()
beta_1 = pyro.param("beta1").item()
print(beta_0)
print(beta_1)
