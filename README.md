# Gpytorch
import numpy as np
import pandas as pd


import math
import torch
import gpytorch
from matplotlib import pyplot as plt


# read data
train=pd.read_csv('/Users/wudanni/Downloads/train.csv')
test = pd.read_csv('/Users/wudanni/Downloads/test.csv')


#training and test
train_x = torch.linspace(0, 95, 96)
train_y =torch.FloatTensor(train.num_thousand_passengers)
test_x = torch.linspace(96,143, 48)
test_y = torch.FloatTensor(test.num_thousand_passengers)


class SpectralMixtureGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(SpectralMixtureGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=7,batch_size=1)
        self.covar_module.initialize_from_data(train_x, train_y)

    def forward(self,x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = SpectralMixtureGPModel(train_x, train_y, likelihood)

# Find optimal model hyperparameters
model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.9)

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

training_iter = 200
for i in range(training_iter):
    optimizer.zero_grad()
    output = model(train_x)
    loss = -mll(output, train_y)
    loss.backward()
    print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iter, loss.item()))
    optimizer.step()



# Get into evaluation (predictive posterior) mode
model.eval()
likelihood.eval()


with torch.no_grad(), gpytorch.fast_pred_var():
    test_x = test_x
    observed_pred = likelihood(model(test_x))
    
with torch.no_grad():
    # Initialize plot
    f, ax = plt.subplots(1, 1, figsize=(4, 3))

    # Get upper and lower confidence bounds
    lower, upper = observed_pred.confidence_region()
    # Plot training data in blue
    ax.plot(train_x.numpy(), train_y.numpy(),"b")
    # Plot testoing data in red
    ax.plot(test_x.numpy(),test_y.numpy(),"r")
    # Plot SM kernel predictive means as blue line
    ax.plot(test_x.numpy(), observed_pred.mean.numpy(), 'k')
    
    
    # Shade between the lower and upper confidence bounds
    ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
    ax.legend(['Training Data','Test Data', 'Mean', 'Confidence'])
    
   
