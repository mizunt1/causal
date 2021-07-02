import math
import torch
# from torch import distributions

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.linear_model import BayesianRidge

import pyro
from pyro import nn
from pyro import infer
from pyro import distributions
from pyro import poutine
from pyro.infer import SVI, Trace_ELBO

def u_model(data, u=None):
    # starting points for params
    x, t = data["x"], data["t"]
    gamma_loc = torch.tensor([0.0,0.0])
    gamma_scale = torch.tensor([[1.0,0.0], [0.0,1.0]])
    big_e_loc = torch.tensor(0.0)
    big_e_scale = torch.tensor(1.0)
    gammas = pyro.sample(
        "gammas",
        distributions.MultivariateNormal(gamma_loc, gamma_scale)
    )
    big_e = pyro.sample(
        "big_e",
        distributions.Normal(big_e_loc, big_e_scale) # or whatever prior
    )
    logits = big_e * x + gammas[1] * t + gammas[0]
    with pyro.plate("data", len(x)):
        u = pyro.sample(
            "obs",
            distributions.Bernoulli(logits=logits),
            obs=u,
        )

def u_guide(data, u=None):
    gamma_loc = pyro.param(
        'gamma_loc',
        torch.tensor([0.0,0.0])
    )
    gamma_scale = pyro.param(
        'gamma_scale',
        torch.tensor([[1.0,0.0], [0.0,1.0]])
    )
    big_e_loc = pyro.param(
        'big_e_loc',
        torch.tensor(0.0)
    )
    big_e_scale = pyro.param(
        'big_e_scale',
        torch.tensor(1.0)
    )
    gammas = pyro.sample(
        "gammas",
        distributions.MultivariateNormal(gamma_loc, gamma_scale)
    )
    big_e = pyro.sample(
        "big_e",
        distributions.Normal(big_e_loc, big_e_scale) # or whatever prior
    )

def y_model(data, y=None):
    x, t, u_ = data["x"], data["t"], data["u"]
    # starting points for params
    betas_mu_loc = torch.tensor([0.0, 0.0, 0.0])
    betas_mu_scale = torch.tensor([[1.0,0.0, 0.0], [0.0,1.0,0.0], [0.0,0.0,1.0]])
    lambda_loc = torch.tensor(0)
    lambda_scale = torch.tensor(1)
    betas_mu = pyro.sample("betas_mu", distributions.MultivariateNormal(betas_mu_loc, betas_mu_scale))
    lambda_ = pyro.sample("lambda", distributions.Normal(lamda_loc, lamda_scale))
    logits = betas_mu[0] + betas_mu[1]*t + lamda_*u + betas_mu[2]*x
    y_sigma = pyro.sample("sigma", distributions.Uniform(0., 10.))
    with pyro.plate("data", len(x)):
        y = pyro.sample(
            "obs",
            distributions.Normal(logits, y_sigma),
            obs=y,
        )

def y_guide(data, y=None):
    betas_mu_loc = pyro.param(
        "betas_mu_loc",
        torch.tensor([0.0,0.0,0.0])
    )
    betas_mu_scale = pyro.param(
        "betas_mu_scale",
        torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    )
    lambda_loc = pyro.param(
        "lambda_loc", torch.tensor(0.0)
    )
    lambda_scale = pyro.param(
        "lambda_scale", torch.tensor(1.0)
    )
    betas_mu  = pyro.sample(
        "betas_mu",
        distributions.MultivariateNormal(betas_mu_loc, betas_mu_scale)
    )

    lambda_ = pyro.sample(
        "lambda",
        distributions.Normal(lambda_loc, lambda_scale) # or whatever prior
    )

class RegressionModelY(nn.PyroModule):
    def __init__(self, in_features, out_features):
        super().__init__()
        # y model
        self.linear = nn.PyroModule[torch.nn.Linear](in_features, out_features)
        # Beta1, Lambda, eta
        self.linear.weight = nn.PyroSample(distributions.Normal(0., 1.))

        # Beta0
        self.linear.bias = nn.PyroSample(
            distributions.Normal(0., 10.).expand([out_features]).to_event(1)
        )
        # priors for the coefficients for regressing on Y. U is input as data.

    def forward(self, data, y=None):
        x, t, u_ = data["x"], data["t"], data["u"]
        x = x.unsqueeze(1)
        t = t.unsqueeze(1)
        u_ = u_.unsqueeze(1)
        # Infer y params
        y_inputs = torch.cat([t, u_, x], dim=1)
        y_sigma = pyro.sample("sigma", distributions.Uniform(0., 10.))
        #y_sigma = distributions.Uniform(0., 10.).sample()
        y_mean = self.linear(y_inputs).squeeze(1)
        with pyro.plate("data_y", x.shape[0]):
            obs = pyro.sample(
                "obs_y", distributions.Normal(y_mean, y_sigma), obs=y)
        return y_mean

class RegressionModelU(nn.PyroModule):
    # u is discrete so it should be a logistic regression model
    def __init__(self, in_features, out_features):
        super().__init__()
        # u model
        self.linear = nn.PyroModule[torch.nn.Linear](in_features, out_features)
        # gamma1, and big E
        self.linear.weight = nn.PyroSample(
            distributions.Normal(0., 1.).expand([out_features, in_features]).to_event(2)
        )
        # gamma 0
        self.linear.bias = nn.PyroSample(
            distributions.Normal(0., 10.).expand([out_features]).to_event(1)
        )

    def forward(self, data, u_=None):
        x, t = data["x"], data["t"]
        # model takes in the whole data set.
        # plate manages the batching etc
        # Sample u
        x = x.unsqueeze(1)
        t = t.unsqueeze(1)
        u_inputs = torch.cat([t, x], dim=1)
        # u_sigma = distributions.Uniform(0., 10.).sample()
        u_mean = self.linear(u_inputs).squeeze(1)
        # Possibly missing u stuff
        with pyro.plate("data_u", x.shape[0]):
            obs = pyro.sample("obs_u", distributions.RelaxedBernoulliStraightThrough(
                temperature=torch.tensor(2/3), logits=u_mean), obs=u_)
        return u_mean

def train(modelY, modelU, guideY, guideU, data, u_guess, iters=10, svi_iters=10, burn_in=0.2):
    beta0s = []
    beta1s = []
    lambdas = []
    etas = []
    big_es = []
    gamma1s = []
    gamma0s = []
    # train takes in whole data set and iterates for each data point in batch
    optimiserY = pyro.optim.Adam({"lr": 0.1})
    optimiserU = pyro.optim.Adam({"lr": 0.01})
    sviY = SVI(modelY, guideY, optimiserY, loss=Trace_ELBO())
    sviU = SVI(modelU, guideU, optimiserU, loss=Trace_ELBO())
    first = True
    for i in range(iters):
        if first:
            u_ = u_guess
            first = False
        pyro.clear_param_store()
        for j in range(svi_iters):
            loss = sviU.step(data, u_)
            gamma_loc = pyro.param("gamma_loc")
            gamma0 = gamma_loc[0]
            gamma1 = gamma_loc[1]
            big_e = pyro.param("big_e_loc")

        print("**************")
        print("gamma1", gamma1)
        print("gamma0", gamma0)
        print("big e", big_e)
        # now we have some updated model parameters for the u model
        # so we sample another u:
        if i > burn_in*iters:
            gamma1s.append(gamma1.item())
            gamma0s.append(gamma0.item())
            big_es.append(big_E.item())

        #new_u_logits = modelU(data)
        new_u_logits  = return_u_logits(data["t"], gamma0, gamma1, big_e, data["x"])
        u_sampler = distributions.Bernoulli(logits=new_u_logits)
        u_ = u_sampler.sample()

        y = data["y"]
        data = {"x": data["x"], "t": data["t"], "y": data["y"], "u": u_}
        pyro.clear_param_store()
        for j in range(svi_iters):
            loss = sviY.step(data, y)
            #print("lossy", loss)
        betas_mu = pyro.param("betas_mu_loc")
        beta0, beta1, mu = betas_mu[0], betas_mu[1], betas_mu[2]
        lambda_ = pyro.param("lambda_loc")
        if i > burn_in*iters:
            beta1s.append(beta1.item())
            beta0s.append(beta0.item())
            lambdas.append(lambda_.item())
            etas.append(eta.item())
        print("beta1s", beta1)
        print("beta0", beta0)
        print("lambda", lambda_)

    dict_ = {"beta0s": beta0s, "beta1s": beta1s, "lambdas": lambdas,
            "etas": etas, "gamma0s": gamma0s, "gamma1s": gamma1s, "big_es": big_es}
    return dict_



def return_y_logits(t, b0, b1, lamb, eta, u, x):
  return b0 + b1* t + lamb*u + eta*x

def return_u_logits(t, gamma0, gamma1, E, x):
  return gamma0 + gamma1*t + E*x

real_latents = {"beta0s": 0.3, "beta1s": -0.1, "lambdas": 0,
            "etas": -0.4, "gamma0s": -0.3, "gamma1s": 0.4, "big_es": 0.2}

t_sampler = distributions.Bernoulli(probs=0.5)
x_sampler = distributions.Normal(0, 1)
u_guess_sampler = distributions.Bernoulli(probs=0.5)

t_real = t_sampler.sample_n(1000)
x_real = x_sampler.sample_n(1000)

u_guess = u_guess_sampler.sample_n(1000)


u_logits = return_u_logits(t_real, real_latents["gamma0s"], real_latents["gamma1s"], real_latents["big_es"], x_real)
u_real_sampler = distributions.Bernoulli(logits = u_logits)
u_real = u_real_sampler.sample()

y_logits = return_y_logits(x=x_real, t=t_real, u=u_real, b0=real_latents["beta0s"],
                           eta=real_latents["etas"], b1=real_latents["beta1s"], lamb=real_latents["lambdas"])
y_sampler = distributions.Normal(loc=y_logits, scale=1.0)
y_real = y_sampler.sample()
data = {"x": x_real, "t": t_real, "y": y_real}
dataY = {"x": x_real, "t": t_real, "u": u_guess}

modelY = RegressionModelY(3, 1)
modelU = RegressionModelU(2, 1)
#trace = poutine.trace(modelY).get_trace(dataY,y_real)
#print(trace.format_shapes())
guideY = infer.autoguide.AutoDiagonalNormal(modelY)
guideU = infer.autoguide.AutoDiagonalNormal(modelU)

modelY = y_model
modelU = u_model
#trace = poutine.trace(modelY).get_trace(dataY,y_real)
#print(trace.format_shapes())
guideY = y_guide
guideU = u_guide

dict_is = train(modelY, modelU, guideY, guideU, data, u_guess, iters=400, svi_iters=200)

for key, value in dict_is.items():
    print("plot for {} should be {}".format(key, real_latents[key]))
    plt.hist(value)
    plt.show()
