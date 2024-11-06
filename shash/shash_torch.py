"""Shash module for pytorch.

Classes
---------
Shash()

"""

__author__ = "Randal J. Barnes and Elizabeth A. Barnes"
__date__ = "03 February 2024"

import torch
import numpy as np
import scipy
import scipy.stats


SQRT_TWO = 1.4142135623730950488016887
ONE_OVER_SQRT_TWO = 0.7071067811865475244008444
TWO_PI = 6.2831853071795864769252868
SQRT_TWO_PI = 2.5066282746310005024157653
ONE_OVER_SQRT_TWO_PI = 0.3989422804014326779399461


class Shash:
    """sinh-arcsinh normal distribution w/o using tensorflow_probability or torch.

    Functions
    ---------
    cdf(x, mu, sigma, gamma, tau=None)
        cumulative distribution function (cdf).

    log_prob(x, mu, sigma, gamma, tau=None)
        log of the probability density function.

    mean(mu, sigma, gamma, tau=None)
        distribution mean.

    median(mu, sigma, gamma, tau=None)
        distribution median.

    prob(x, mu, sigma, gamma, tau=None)
        probability density function (pdf).

    quantile(pr, mu, sigma, gamma, tau=None)
        inverse cumulative distribution function.

    rvs(mu, sigma, gamma, tau=None, size=1)
        generate random variates.

    stddev(mu, sigma, gamma, tau=None)
        distribution standard deviation.

    variance(mu, sigma, gamma, tau=None)
        distribution variance.

    get_params(pred)
        get mu, sigma, gamma, tau

    get_median_prediction(x_input,model)
        get the deterministic median prediction

    get_mean_prediction(x_input,model)
        get the deterministic mean prediction


    Notes
    -----
    * This module uses only pytorch.

    * The sinh-arcsinh normal distribution was defined in [1]. A more accessible
    presentation is given in [2].

    * The notation and formulation used in this code was taken from [3], page 143.
    In the gamlss.dist/CRAN package the distribution is called SHASHo.

    * There is a typographical error in the presentation of the probability
    density function on page 143 of [3]. There is an extra "2" in the denomenator
    preceeding the "sqrt{1 + z^2}" term.

    References
    ----------
    [1] Jones, M. C. & Pewsey, A., Sinh-arcsinh distributions,
    Biometrika, Oxford University Press, 2009, 96, 761-780.
    DOI: 10.1093/biomet/asp053.

    [2] Jones, C. & Pewsey, A., The sinh-arcsinh normal distribution,
    Significance, Wiley, 2019, 16, 6-7.
    DOI: 10.1111/j.1740-9713.2019.01245.x.
    https://rss.onlinelibrary.wiley.com/doi/10.1111/j.1740-9713.2019.01245.x

    [3] Stasinopoulos, Mikis, et al. (2021), Distributions for Generalized
    Additive Models for Location Scale and Shape, CRAN Package.
    https://cran.r-project.org/web/packages/gamlss.dist/gamlss.dist.pdf

    """

    def __init__(self, params):
        """
        mu : float (batch size x 1) Tensor
            The location parameter.

        sigma : float (batch size x 1) Tensor
            The scale parameter. Must be strictly positive.

        gamma : float (batch size x 1) Tensor
            The skewness parameter.

        tau : float (batch size x 1) Tensor
            The tail-weight parameter. Must be strictly positive. If tau is None then the default value of tau=1 is used.
        """

        self.mu = params[:, 0]
        self.sigma = params[:, 1]
        self.gamma = params[:, 2]
        self.tau = params[:, 3]

        # Only necessary during evaluation, not during training.
        # Need to check after params has been separated, otherwise
        # it is not differentiable.
        if not torch.is_tensor(self.mu):
            self.mu = torch.tensor(self.mu)
            self.sigma = torch.tensor(self.sigma)
            self.gamma = torch.tensor(self.gamma)
            self.tau = torch.tensor(self.tau)

    def _jones_pewsey_P(self, q):
        """P_q function from page 764 of [1].

        Arguments
        ---------
        q : float, array like

        Returns
        -------
        P_q : array like of same shape as q.

        Notes
        -----
        * The formal equation is

                jp = 0.25612601391340369863537463 * (
                    scipy.special.kv((q + 1) / 2, 0.25) +
                    scipy.special.kv((q - 1) / 2, 0.25)
                )

            The strange constant 0.25612... is "sqrt( sqrt(e) / (8*pi) )" computed
            with a high-precision calculator.  The special function

                scipy.special.kv

            is the Modified Bessel function of the second kind: K(nu, x).

        * But, we cannot use the scipy.special.kv function during tensorflow
            training.  This code uses a 6th order polynomial approximation in
            place of the formal function.

        * This approximation is well behaved for 0 <= q <= 10. Since q = 1/tau
            or q = 2/tau in our applications, the approximation is well behaved
            for 1/10 <= tau < infty.

        """
        # A 6th order polynomial approximation of log(_jones_pewsey_P) for the
        # range 0 <= q <= 10.  Over this range, the max |error|/true < 0.0025.
        # These coefficients were computed by minimizing the maximum relative
        # error, and not by a simple least squares regression.

        # coeffs = [
        #     9.37541380598926e-06,
        #     -0.000377732651131894,
        #     0.00642826706073389,
        #     -0.061281078712518,
        #     0.390956214318641,
        #     -0.0337884356755193,
        #     0.00248824801827172,
        # ]

        # val = (
        #     coeffs[0] * q**6
        #     + coeffs[1] * q**5
        #     + coeffs[2] * q**4
        #     + coeffs[3] * q**3
        #     + coeffs[4] * q**2
        #     + coeffs[5] * q**1
        #     + coeffs[6] * q**0
        # )
        # return torch.exp(val)

        jp = 0.25612601391340369863537463 * (
            scipy.special.kv((q + 1) / 2, 0.25) + scipy.special.kv((q - 1) / 2, 0.25)
        )
        return jp

    def prob(self, x):
        """Probability density function (pdf).

        Parameters
        ----------
        x : float (batch size x 1) Tensor
            The values at which to compute the probability density function.

        Returns
        -------
        f : float (batch size x 1) Tensor.
            The computed probability density function evaluated at the values of x.
            f has the same shape as x.

        Notes
        -----
        * This code uses the equations on page 143 of [3], and the associated
        notation.

        """

        if not torch.is_tensor(x):
            if hasattr(x, "__len__"):
                x = torch.tensor(x[:, None])
            else:
                x = torch.tensor(x)

        y = (x - self.mu) / self.sigma
        y = torch.divide(torch.subtract(x, self.mu), self.sigma)

        if self.tau is None:
            rsqr = torch.square(torch.sinh(torch.asinh(y) - self.gamma))
            return (
                ONE_OVER_SQRT_TWO_PI
                / self.sigma
                * torch.sqrt((1 + rsqr) / (1 + torch.square(y)))
                * torch.exp(-rsqr / 2)
            )

        else:
            rsqr = torch.square(torch.sinh(self.tau * torch.asinh(y) - self.gamma))
            return (
                ONE_OVER_SQRT_TWO_PI
                * (self.tau / self.sigma)
                * torch.sqrt((1 + rsqr) / (1 + torch.square(y)))
                * torch.exp(-rsqr / 2)
            )

    def log_prob(self, x):
        """Log-probability density function.

        Parameters
        ----------
        x : float (batch size x 1) Tensor
            The values at which to compute the probability density function.

        Returns
        -------
        f : float (batch size x 1) Tensor.
            The natural logarithm of the computed probability density function
            evaluated at the values of x.  f has the same shape as x.

        Notes
        -----
        * This function is included merely to emulate the tensorflow_probability
        distributions.

        """
        return torch.log(self.prob(x))

    def cdf(self, x):
        """Cumulative distribution function (cdf).

        Parameters
        ----------
        x : float (batch size x 1) Tensor
            The values at which to compute the probability density function.

        mu : float (batch size x 1) Tensor
            The location parameter. Must be the same shape as x.

        sigma : float (batch size x 1) Tensor
            The scale parameter. Must be strictly positive. Must be the same
            shape as x.

        gamma : float (batch size x 1) Tensor
            The skewness parameter. Must be the same shape as x.

        tau : float (batch size x 1) Tensor or None
            The tail-weight parameter. Must be strictly positive. Must be the same
            shape as x. If tau is None then the default value of tau=1 is used.

        Returns
        -------
        F : float (batch size x 1) Tensor.
            The computed cumulative probability distribution function (cdf)
            evaluated at the values of x.  F has the same shape as x.

        Notes
        -----
        * This function uses the tensorflow.math.erf function rather than the
        tensorflow_probability normal distribution functions.

        """
        if not torch.is_tensor(x):
            x = torch.tensor(x)

        y = (x - self.mu) / self.sigma

        if self.tau is None:
            z = torch.sinh(torch.asinh(y) - self.gamma)
        else:
            z = torch.sinh(self.tau * torch.asinh(y) - self.gamma)

        return 0.5 * (1.0 + torch.erf(ONE_OVER_SQRT_TWO * z))

    def mean(self):
        """The distribution mean.

        Arguments
        ---------
        mu : float (batch size x 1) Tensor
            The location parameter.

        sigma : float (batch size x 1) Tensor
            The scale parameter. Must be strictly positive. Must be the same
            shape as mu.

        gamma : float (batch size x 1) Tensor
            The skewness parameter. Must be the same shape as mu.

        tau : float (batch size x 1) Tensor
            The tail-weight parameter. Must be strictly positive. Must be the same
            shape as mu. If tau is None then the default value of tau=1 is used.

        Returns
        -------
        x : float (batch size x 1) Tensor.
            The computed distribution mean values.

        Notes
        -----
        * This equation for evX can be found on page 764 of [1].

        """
        if self.tau is None:
            evX = torch.sinh(self.gamma) * 1.35453080648132
        else:
            evX = torch.sinh(self.gamma / self.tau) * self._jones_pewsey_P(
                1.0 / self.tau
            )

        return self.mu + self.sigma * evX

    def median(self):
        """The distribution median.

        Arguments
        ---------
        mu : float (batch size x 1) Tensor
            The location parameter.

        Returns
        -------
        x : float (batch size x 1) Tensor.
            The computed distribution mean values.

        Notes
        -----
        * This code uses the basic formula:

            E(a*X + b) = a*E(X) + b

        * The E(X) is computed using the moment equation given on page 764 of [1].

        """
        if self.tau is None:
            return self.mu + self.sigma * torch.sinh(self.gamma)
        else:
            return self.mu + self.sigma * torch.sinh(self.gamma / self.tau)

    def quantile(self, pr):
        """Inverse cumulative distribution function.

        Arguments
        ---------
        pr : float (batch size x 1) Tensor.
            The probabilities at which to compute the values.

        Returns
        -------
        x : float (batch size x 1) Tensor.
            The computed values at the specified probabilities. f has the same
            shape as pr.

        """
        if not torch.is_tensor(pr):
            pr = torch.tensor(pr)

        z = torch.special.ndtri(pr)

        if self.tau is None:
            return self.mu + self.sigma * torch.sinh(torch.asinh(z) + self.gamma)
        else:
            return self.mu + self.sigma * torch.sinh(
                (torch.asinh(z) + self.gamma) / self.tau
            )

    def rvs(self, size=1, random_state=42):
        """Generate an array of random variates.

        Arguments
        ---------
        size : int or tuple of ints, default=1.
            The number of random variates.

        Returns
        -------
        x : double ndarray of size=size
            The generated random variates.

        """
        # xi=mu, eta=sigma, epsilon=gamma, delta=tau
        z = torch.tensor(scipy.stats.norm.rvs(size=size, random_state=random_state))

        if self.tau is None:
            return self.mu + self.sigma * torch.sinh(torch.arcsinh(z) + self.gamma)
        else:
            return self.mu + self.sigma * torch.sinh(
                (torch.arcsinh(z) + self.gamma) / self.tau
            )

    def std(self):
        """The distribution standard deviation.

        Arguments
        ---------

        Returns
        -------
        x : float (batch size x 1) Tensor.
            The computed distribution standard deviation values.

        """
        return torch.sqrt(self.var())

    def var(self):
        """The distribution variance.

        Arguments
        ---------

        Returns
        -------
        x : float (batch size x 1) Tensor.
            The computed distribution variance values.

        Notes
        -----
        * This code uses two basic formulas:

            var(X) = E(X^2) - (E(X))^2
            var(a*X + b) = a^2 * var(X)

        * The E(X) and E(X^2) are computed using the moment equations given on
        page 764 of [1].

        """
        if self.tau is None:
            evX = torch.sinh(self.gamma) * 1.35453080648132
            evX2 = (torch.cosh(2 * self.gamma) * 3.0 - 1.0) / 2
        else:
            evX = torch.sinh(self.gamma / self.tau) * self._jones_pewsey_P(
                1.0 / self.tau
            )
            evX2 = (
                torch.cosh(2 * self.gamma / self.tau)
                * self._jones_pewsey_P(2.0 / self.tau)
                - 1.0
            ) / 2

        return torch.square(self.sigma) * (evX2 - torch.square(evX))

    def skewness(self):
        """The distribution skewness. Named as such to not overwrite the "skewness" parameter.

        Returns
        -------
        x : Tensor of same dtype and shape as loc specified at initialization.
            The computed distribution skewness values.

        Notes
        -----
        * The E(X), E(X^2), and E(X^3) are computed using the moment equations
        given on page 764 of [1].

        """

        # raise Warning("This code is not correct.")
        # xi=mu, eta=sigma, epsilon=gamma, delta=tau
        # https://www.randomservices.org/random/expect/Skew.html

        evX = torch.sinh(self.gamma / self.tau) * self._jones_pewsey_P(1.0 / self.tau)
        evX2 = (
            torch.cosh(2.0 * self.gamma / self.tau)
            * self._jones_pewsey_P(2.0 / self.tau)
            - 1.0
        ) / 2.0
        evX3 = (
            torch.sinh(3.0 * self.gamma / self.tau)
            * self._jones_pewsey_P(3.0 / self.tau)
            - 3.0
            * torch.sinh(self.gamma / self.tau)
            * self._jones_pewsey_P(1.0 / self.tau)
        ) / 4.0

        term_1 = evX3
        term_2 = -3.0 * evX * evX2
        term_3 = 2.0 * evX * evX * evX
        denom = torch.pow(torch.sqrt(evX2 - evX * evX), 3)

        return (term_1 + term_2 + term_3) / denom
