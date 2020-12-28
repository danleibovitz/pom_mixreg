import pandas
import random
import numpy as np
import scipy
import warnings

from pomegranate import *


class LinearReg():
    def __init__(self, coefs, std):
        self.coefs = coefs
        self.std = std
        self.parameters = (self.coefs, self.std)
        self.d = len(coefs) - 1
        # TODO how long must self.summaries be?
        self.summaries = np.zeros(3)

    def log_probability(self, Y, X=None):
        if X is None:
            warnings.warn("log_probability() called with no covariate (X) "
                          "argument calculates conditional mean Y as if all covariates = 0")
            X = np.zeros(self.d)
        if len(X) != self.d:
            message = "Sample covariates must be same dimension as LinearReg instance, in this case: " + str(self.d)
            raise ValueError(message)
        return scipy.stats.norm.logpdf(Y, np.dot(np.concatenate(([1], X)), self.coefs), self.std)

    def summarize(self, Y, X, w=None):
        if w is None:
            w = np.ones(X.shape[0])
        if X.shape[1] != self.d:
            raise ValueError("Covariates are of incorrect dimension")
        X = X.reshape(X.shape[0])
        self.summaries[0] += w.sum()
        self.summaries[1] += X.dot(w)
        self.summaries[2] += (X ** 2.).dot(w)

    def from_summaries(self, inertia=0.0):
        self.mu = self.summaries[1] / self.summaries[0]
        self.std = self.summaries[2] / self.summaries[0] - self.summaries[1] ** 2 / (self.summaries[0] ** 2)
        self.std = np.sqrt(self.std)
        self.parameters = (self.mu, self.std)
        self.clear_summaries()

    def clear_summaries(self, inertia=0.0):
        # TODO replace with default summaries
        self.summaries = np.zeros(3)

    @classmethod
    def from_samples(cls, Y, X, weights=None):
        # instantiate a linear model with 0-coefs of length = number of columns in X plus 1
        d = LinearReg(np.zeros(X.shape[1] + 1), 0)
        d.summarize(Y, X, weights)
        d.from_summaries()
        return d

    @classmethod
    def blank(cls):
        return NormalDistribution2(0, 0)