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


class RegressionModelY(nn.PyroModule):
    def __init__(self, in_features, out_features):
        super().__init__()
        # y model
        self.linear = nn.PyroModule[torch.nn.Linear](in_features, out_features)
        # Beta1, Lambda, eta
        self.linear.weight = nn.PyroSample(
            distributions.Normal(0., 1.).expand([out_features, in_features]).to_event(2)
        )
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
        y_sigma = distributions.Uniform(0., 10.).sample()
        y_mean = self.linear(y_inputs).squeeze(1)
        with pyro.plate("data_y", x.shape[0]):
            obs = pyro.sample(
                "obs_y", distributions.Normal(y_mean, y_sigma), obs=y)
        return y_mean

class RegressionModelU(nn.PyroModule):
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
        u_sigma = distributions.Uniform(0., 10.).sample()
        u_mean = self.linear(u_inputs).squeeze(1)
        # Possibly missing u stuff
        with pyro.plate("data_u", x.shape[0]):
            obs = pyro.sample("obs_u", distributions.Normal(u_mean, u_sigma), obs=u_)
        return u_mean


def monte_carlo_elbo(model, guide, batch,  *args, **kwargs):
    # assuming batch is a dictionary, we use poutine.condition to fix values of observed variables
    conditioned_model = poutine.condition(model, data=batch)
    # we'll approximate the expectation in the ELBO with a single sample:
    # first, we run the guide forward unmodified and record values and distributions
    # at each sample site using poutine.trace
    guide_trace = poutine.trace(guide).get_trace(*args, **kwargs)

    # we use poutine.replay to set the values of latent variables in the model
    # to the values sampled above by our guide, and use poutine.trace
    # to record the distributions that appear at each sample site in in the model
    model_trace = poutine.trace(
        poutine.replay(conditioned_model, trace=guide_trace)
    ).get_trace(*args, **kwargs)

    elbo = 0.
    for name, node in model_trace.nodes.items():
        if node["type"] == "sample":
            elbo = elbo + node["fn"].log_prob(node["value"]).sum()
            if not node["is_observed"]:
                elbo = elbo - guide_trace.nodes[name]["fn"].log_prob(node["value"]).sum()
    return -elbo


def train(modelY, modelU, guideY, guideU, data, u_guess, iters=100, svi_iters=100):
    beta0s = []
    beta1s = []
    lambdas = []
    etas = []
    big_es = []
    gamma1s = []
    gamma0s = []
    # train takes in whole data set and iterates for each data point in batch
    optimiserY = pyro.optim.Adam({"lr": 0.03})
    optimiserU = pyro.optim.Adam({"lr": 0.03})
    sviY = SVI(modelY, guideY, optimiserY, loss=Trace_ELBO())
    sviU = SVI(modelU, guideU, optimiserU, loss=Trace_ELBO())
    first = True
    for i in range(iters):
        if first:
            u_ = u_guess
            first = False
        pyro.clear_param_store()
        for j in range(svi_iters):
            sviU.step(data, u_)

        # now we have some updated model parameters for the u model
        # so we sample another u:
        gamma1, big_E = modelU.linear.weight[0][0], modelU.linear.weight[0][1]
        gamma0 = modelU.linear.bias
        gamma1s.append(gamma1)
        gamma0s.append(gamma0s)
        big_es.append(big_E)
        new_u_logits  = return_u_logits(data["t"], gamma0, gamma1, big_E, data["x"])
        u_sampler = distributions.Bernoulli(logits=new_u_logits)
        u_ = u_sampler.sample()
        y = data["y"]
        data = {"x": data["x"], "t": data["t"], "y": data["y"], "u": u_}
        pyro.clear_param_store()
        for j in range(svi_iters):
            sviY.step(data, y)
        beta1, lambda_, eta = modelY.linear.weight[0][0], modelY.linear.weight[0][1], modelY.linear.weight[0][2]
        beta0 = modelY.linear.bias
        beta1s.append(beta1)
        beta0s.append(beta0)
        lambdas.append(lambda_)
        etas.append(eta)
    dict_ = {"beta0s": beta0s, "beta1s": beta1s, "lambdas": lambdas,
            "etas": etas, "gamma0s": gamma0s, "gamma1s": gamma1s, "big_es": big_es}
    return dict_



def return_y_logits(t, b0, b1, lamb, eta, u, x):
  return b0 + b1* t + lamb*u + eta*x

def return_u_logits(t, gamma0, gamma1, E, x):
  return gamma0 + gamma1*t + E*x

real_latents = {"beta0s": 0.2, "beta1s": 1.2, "lambdas": 0.1,
            "etas": 0.1, "gamma0s": 0.4, "gamma1s": 0.3, "big_es": 0.5}

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

adamY = pyro.optim.Adam({"lr": 0.03})
adamU = pyro.optim.Adam({"lr": 0.03})
dict_is = train(modelY, modelU, guideY, guideU, data, u_guess)

for key, value in dict_is.items():
    print("plot for {} should be {}".format(key, real_latents[key]))
    plt.hist(value)
