import math
import torch
import os
# from torch import distributions
import numpy as np

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.linear_model import BayesianRidge

import pyro
from pyro import nn
from pyro import infer
from pyro import distributions
from pyro import poutine
from pyro.infer import SVI, Trace_ELBO

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

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
    c4 = math.log(3, 10)/1.96
    c3 = c4
    gamma_loc = pyro.param(
        'gamma_loc',
        torch.tensor([0.0,0.0])
    )
    gamma_scale = pyro.param(
        'gamma_scale',
        torch.tensor([[c4,c3/2], [c3/2,c4]]), constraint=distributions.constraints.positive_definite)
    big_e_loc = pyro.param(
        'big_e_loc',
        torch.tensor(0.0)
    )
    big_e_scale = pyro.param(
        'big_e_scale',
        torch.tensor(c4), constraint=distributions.constraints.positive
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
    betas_eta_loc = torch.tensor([0.0, 0.0, 0.0])
    betas_eta_scale = torch.tensor([[1.0,0.0, 0.0], [0.0,1.0,0.0], [0.0,0.0,1.0]])
    lambda_loc = torch.tensor(0)
    lambda_scale = torch.tensor(1)
    betas_eta = pyro.sample("betas_eta", distributions.MultivariateNormal(betas_eta_loc, betas_eta_scale))
    lambda_ = pyro.sample("lambda", distributions.Normal(lambda_loc, lambda_scale))
    logits = betas_eta[0] + betas_eta[1]*t + lambda_*u_ + betas_eta[2]*x
    y_sigma = pyro.sample("sigma", distributions.Uniform(0., 10.))
    with pyro.plate("data", len(x)):
        y = pyro.sample(
            "obs",
            distributions.Normal(logits, y_sigma),
            obs=y,
        )

def y_guide(data, y=None):
    c1 = math.log(6, 10) / 1.96
    # base 10??
    # uninformative prior for betas and eta
    betas_eta_loc = pyro.param(
        "betas_eta_loc",
        torch.tensor([0.0,0.0,0.0])
    )
    betas_eta_scale = pyro.param(
        "betas_eta_scale",
        torch.tensor([[1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0],
                      [0.0, 0.0, 1.0]]),
        constraint=distributions.constraints.positive_definite)
    lambda_loc = pyro.param(
        "lambda_loc", torch.tensor(0.0)
    )
    lambda_scale = pyro.param(
        "lambda_scale", torch.tensor(c1), constraint=distributions.constraints.positive
    )
    #print("det", np.linalg.det(betas_eta_scale.detach().numpy()))
    try:
        betas_eta  = pyro.sample(
            "betas_eta",
            distributions.MultivariateNormal(betas_eta_loc, betas_eta_scale)
        )


    except:
        import pdb
        pdb.set_trace()

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

def train(modelY, modelU, guideY, guideU, data, u_guess, real_latents, iters=2, svi_iters=5, burn_in=0.2):
    beta0s = []
    beta1s = []
    lambdas = []
    etas = []
    big_es = []
    gamma1s = []
    gamma0s = []
    # train takes in whole data set and iterates for each data point in batch
    optimiserY = pyro.optim.Adam({"lr": 0.1})
    optimiserU = pyro.optim.Adam({"lr": 0.1})
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
            gamma = pyro.param("gamma_loc")
            gamma0 = gamma[0]
            gamma1 = gamma[1]
            big_e = pyro.param("big_e_loc")
            if j % 100 == 0:
                print("u loss: {}".format(loss))

        print("**************")
        print("gamma1 pred: {:.2f} gamma1 actual: {:.2f}".format(
            gamma1, real_latents["gamma1s"]))
        print(
            "gamma0 pred: {:.2f} gamma0 actual: {:.2f}".format(gamma0, real_latents["gamma0s"]))
        print("big e pred: {:.2f} big e actual: {:.2f}".format(big_e, real_latents["big_es"]))
        # now we have some updated model parameters for the u model
        # so we sample another u:
        if i > burn_in*iters:
            gamma1s.append(gamma1.item())
            gamma0s.append(gamma0.item())
            big_es.append(big_e.item())

        #new_u_logits = modelU(data)
        new_u_logits  = return_u_logits(data["t"], gamma0, gamma1, big_e, data["x"])
        u_sampler = distributions.Bernoulli(logits=new_u_logits)
        u_ = u_sampler.sample()

        y = data["y"]
        data = {"x": data["x"], "t": data["t"], "y": data["y"], "u": u_}
        pyro.clear_param_store()
        for j in range(svi_iters):
            loss = sviY.step(data, y)
            if j % 100 == 0:
                print("y loss: {}".format(loss) )
        #print("lossy", loss)
        betas_eta = pyro.param("betas_eta_loc")
        beta0, beta1, eta = betas_eta[0], betas_eta[1], betas_eta[2]
        lambda_ = pyro.param("lambda_loc")
        if i > burn_in*iters:
            beta1s.append(beta1.item())
            beta0s.append(beta0.item())
            lambdas.append(lambda_.item())
            etas.append(eta.item())
        print("beta1s pred: {:.2f} beta1s actual: {:.2f}".format(beta1, real_latents["beta1s"]))
        print("beta0s pred: {:.2f} beta0s actual: {:.2f}".format(beta0, real_latents["beta0s"]))
        print("lambda pred: {:.2f} lambda actual: {:.2f}".format(lambda_, real_latents["lambdas"]))

    dict_ = {"beta0s": beta0s, "beta1s": beta1s, "lambdas": lambdas,
            "etas": etas, "gamma0s": gamma0s, "gamma1s": gamma1s, "big_es": big_es}
    return dict_



def return_y_logits(t, b0, b1, lamb, eta, u, x):
  return b0 + b1* t + lamb*u + eta*x

def return_u_logits(t, gamma0, gamma1, E, x):
  return gamma0 + gamma1*t + E*x

def return_real_latents(set_gammas=None, set_lambda=None):
    c1 = math.log(6) / 1.96
    betas_eta_dist = distributions.MultivariateNormal(
        torch.tensor([0.0, 0.0, 0.0]), torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))
    betas_eta = betas_eta_dist.sample()
    beta0 = betas_eta[0]
    beta1 = betas_eta[1]
    eta = betas_eta[2]
    lambda_dist = distributions.Normal(torch.tensor(0.0), torch.tensor(c1))
    lambda_ = lambda_dist.sample()
    gammas_dist = distributions.MultivariateNormal(torch.tensor([0.0,0.0]), torch.tensor([[c1, c1/2], [c1/2, c1]]))
    gammas = gammas_dist.sample()
    gamma0 = gammas[0]
    gamma1 = gammas[1]
    big_e_dist = distributions.Normal(torch.tensor(0.0), torch.tensor(c1))
    big_e = big_e_dist.sample()
    if set_gammas != None:
        gamma0 = set_gammas[0]
        gamma1 = set_gammas[1]
    if set_lambda != None:
        lambda_ = set_lambda
    return {"beta0s": beta0, "beta1s": beta1, "lambdas": lambda_,
            "etas": eta, "gamma0s": gamma0, "gamma1s": gamma1, "big_es": big_e}




real_latents = return_real_latents(set_lambda=0)
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
modelY = y_model
modelU = u_model
guideY = infer.autoguide.AutoDiagonalNormal(modelY)
guideU = infer.autoguide.AutoDiagonalNormal(modelU)



#trace = poutine.trace(modelY).get_trace(dataY,y_real)
#print(trace.format_shapes())
guideY = y_guide
guideU = u_guide

dict_is = train(modelY, modelU, guideY, guideU, data, u_guess, real_latents, iters=100, svi_iters=400, burn_in=0.2)

exp_name = "lambda_zero"
os.mkdir(os.path.expanduser("~/causal_stuff/results/"+exp_name))
for key, value in dict_is.items():
    print("plot for {} should be {}".format(key, real_latents[key]))
    plt.hist(value)
    plt.savefig(os.path.expanduser("~/causal_stuff/results/{}/{}_{:.2f}.jpg".format(exp_name, key, real_latents[key])))
    plt.show()
