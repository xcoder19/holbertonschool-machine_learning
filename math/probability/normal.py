#!/usr/bin/env python3

"""Normal"""


class Normal:
    """Normal"""

    def __init__(self, data=None, mean=0., stddev=1.):
        if data is None:
            if not (isinstance(mean, (int, float))):
                raise TypeError("mean must be a number")
            if not (stddev > 0):
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.mean = sum(data) / len(data)
            self.stddev = (
                (sum((x - self.mean) ** 2 for x in data) / len(data)) ** 0.5)

    def z_score(self, x):
        """returns z_score value"""

        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """returns x_value of a give z_score"""

        return self.mean + z * self.stddev

    def pdf(self, x):
        """returns pdf value"""
        pi = 3.1415926536
        exp_term = 2.7182818285**(-0.5 * ((x - self.mean) / self.stddev) ** 2)
        pdf_value = (1 / (self.stddev * (2 * pi)**0.5)) * exp_term
        return pdf_value

    def cdf(self, x):
        """returns cdf"""

        z = self.z_score(x)
        cdf_value = 0.5 * (1 + self.erf(z / (2)**0.5))
        return cdf_value

    def erf(self, x):
        """returns erf value"""

        pi = 3.1415926536
        return ((2 / pi ** 0.5) * (
            x - (x ** 3 / 3) +
            (x ** 5 / 10) -
            (x ** 7 / 42) +
            (x ** 9 / 216)
        ))
