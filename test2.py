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

    def forward(self, data_obs, y=None):
        x, t, u  = data_obs["x"], data_obs["t"], data_obs["u"]
        # Infer y params
        y_inputs = torch.cat([x, t, u], dim=-1)
        y_sigma = pyro.sample("sigma_y", distributions.Uniform(0., 10.))
        y_mean = self.linear(y_inputs).squeeze(-1)
        with pyro.plate("data_y", x.shape[0]):
            obs = pyro.sample("obs_y", distributions.Normal(y_mean, y_sigma), obs=y)
        return y_mean

class RegressionModelU(nn.PyroModule):
    def __init__(self, in_features, out_features):
        super().__init__()
        # u model
        self.linear = nn.PyroModule[torch.nn.Linear](2, 1)
        # gamma1, and big E
        self.linear.weight = nn.PyroSample(
            distributions.Normal(0., 1.).expand([1, 2]).to_event(2)
        )
        # gamma 0
        self.linear.bias = nn.PyroSample(
            distributions.Normal(0., 10.).expand([1]).to_event(1)
        )

    def forward(self, data_obs, u=None):
        x, t = data_obs["x"], data_obs["t"]
        # model takes in the whole data set.
        # plate manages the batching etc
        # Sample u
        u_inputs = torch.cat([x, t], dim=-1)
        u_sigma = pyro.sample("sigma_u", distributions.Uniform(0., 10.))
        u_mean = self.linear(u_inputs).squeeze(-1)
        # Possibly missing u stuff
        with pyro.plate("data_u", x.shape[0]):
            obs = pyro.sample("obs_u", distributions.Normal(u_mean, u_sigma), obs=u)
        return u_mean


def monte_carlo_elbo(model, guide, batch,  *args, **kwargs):
    # assuming batch is a dictionary, we use poutine.condition to fix values of observed variables
    import pdb
    pdb.set_trace()
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


def train(modelY, modelU, guideY, guideU, data, u_guess):
    # train takes in whole data set and iterates for each data point in batch
    optimizerY = pyro.optim.Adam({})
    optimizerU = pyro.optim.Adam({})
    first = True
    for i in range(len(data["x"])):
        x, t, y = data["x"][i], data["t"][i], data["y"][i]
        if first:
            u = u_guess[i]
            first = False
        # only first element of u guess is used this is weird
        # should probably be using the whole of u_guess?
        # this poutine.trace will record all of the parameters
        # that appear in the model and guide
        # during the execution of monte_carlo_elbo

        with poutine.trace() as param_capture:
            # we use poutine.block here so that only parameters appear in the trace above
            with poutine.block(hide_fn=lambda node: node["type"] != "param"):
                # modify batch to include u
                data_point = {"data_obs": {"x": x, "t": t}, "u": u}
                import pdb
                pdb.set_trace()
                lossU = monte_carlo_elbo(modelU, guideU, data_point)
        lossU.backward()
        params = set(node["value"].unconstrained()
                     for node in param_capture.trace.nodes.values())
        optimizerU.step(params)
        pyro.infer.util.zero_grads(params)

        # now we have some updated model parameters for the u model
        # so we sample another u:
        gamma1, big_E = modelU.linear.weight[0], modelU.linear.weight[1]
        gamma0 = modelU.linear.bias
        new_u_logits  = return_u_logits(t, gamma0, gamma1, big_E, x)
        u_sampler = distributions.Bernoulli(logits=new_u_logits)
        u = u_sampler.sample()
        with poutine.trace() as param_capture:
            # we use poutine.block here so that only parameters appear in the trace above
            with poutine.block(hide_fn=lambda node: node["type"] != "param"):
                # modify batch to include u
                data_point = {"data_obs":{"x": x, "y": y, "t":t}, "u": u}
                lossY = monte_carlo_elbo(modelY, guideY, data_point)
        lossY.backward()
        # the above gets some gradients
        params = set(node["value"].unconstrained()
                     for node in param_capture.trace.nodes.values())
        # change latent params so that elbo is maximised.
        # what are these latent params,
        # seems like currently they are weights of network? not sure

        optimizerY.step(params)
        pyro.infer.util.zero_grads(params)


def return_y_logits(t, b0, b1, lamb, eta, u, x):
  return b0 + b1* t + lamb*u + eta*x

def return_u_logits(t, gamma0, gamma1, E, x):
  return gamma0 + gamma1*t + E*x

t_sampler = distributions.Bernoulli(probs=0.5)
x_sampler = distributions.Normal(0, 1)
u_sampler = distributions.Bernoulli(probs=0.5)

t_real = t_sampler.sample_n(1000)
x_real = x_sampler.sample_n(1000)

u_guess = u_sampler.sample_n(1000)
y_logits = return_y_logits(x=x_real, t=t_real, u=u_guess, b0=0.2, eta=0.1, b1=1.2, lamb=0.1)
y_sampler = distributions.Normal(loc=y_logits, scale=1.0)
y_real = y_sampler.sample()

data = {"x": x_real, "t": t_real, "y": y_real}

modelY = RegressionModelY(3, 1)
modelU = RegressionModelU(2, 1)
guideY = infer.autoguide.AutoDiagonalNormal(modelY)
guideU = infer.autoguide.AutoDiagonalNormal(modelU)

adamY = pyro.optim.Adam({"lr": 0.03})
adamU = pyro.optim.Adam({"lr": 0.03})
train(modelY, modelU, guideY, guideU, data, u_guess)