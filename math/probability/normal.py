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
