#!/usr/bin/env python3
"""BayesianOptimization"""

import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """BayesianOptimization"""

    def __init__(self, f, X_init, Y_init, bounds, ac_samples, l=1, sigma_f=1, xsi=0.01, minimize=True):
        """init"""
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        self.X_s = np.linspace(bounds[0], bounds[1], ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """acquisition"""
        mu_s, sigma_s = self.gp.predict(self.X_s)

        if self.minimize:
            best_y = np.min(self.gp.Y)
            imp = best_y - mu_s - self.xsi
        else:
            best_y = np.max(self.gp.Y)
            imp = mu_s - best_y - self.xsi

        with np.errstate(divide='warn'):
            Z = imp / sigma_s
            cdf_Z = norm.cdf(Z)
            pdf_Z = norm.pdf(Z)

        ei = imp * cdf_Z + sigma_s * pdf_Z

        return self.X_s[np.argmax(ei)], ei
