#!/usr/bin/env python3
"""BayesianOptimization"""

from scipy.stats import norm
import numpy as np
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

        tol = 1e-9
        Z = imp / (sigma_s + tol)
        cdf_Z = norm.cdf(Z)
        pdf_Z = norm.pdf(Z)

        ei = imp * cdf_Z + sigma_s * pdf_Z

        return self.X_s[np.argmax(ei)], ei

    def optimize(self, iterations=100):
        """optimize"""
        for _ in range(iterations):
            X_next, _ = self.acquisition()

            if X_next in self.gp.X:
                break

            Y_next = self.f(X_next)
            self.gp.update(X_next, Y_next)

        idx_opt = np.argmin(
            self.gp.Y) if self.minimize else np.argmax(self.gp.Y)

        X_opt = self.gp.X[idx_opt]
        Y_opt = self.gp.Y[idx_opt]

        X_opt = np.array([X_opt])
        Y_opt = np.array([Y_opt])

        return X_opt, Y_opt
