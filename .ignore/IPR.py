# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 16:30:59 2016

@author: Marcelo

IPR code and classes

In it's simpler form, an IPR is simply a flow vs. P equation. A flow can be
either a vector or a scalar. Pressure and flow units are not specified

Analytical IPR is a child of IPR with kh, skin, re and rw as inputs. Also,
a pseudo pressure function (deltaP / mu.FVF vs Pwf and Pws) is input,
together with a dimensionless pressure drop.

The IPR object (and its childs) can output either a vector or a scalar.
This is signaled in .isScalar boolean, which is set at initialization,
based on a test on the rate_function

DataFitIPR is scalar only

AnalyticIPR and FluidIPR can input a volume function to output a vector.
The simplest example is an Rs from a Live Oil


TODO

-rever metodos ahora que no hay mas inheritance de cc.Globals
-rever fits
"""

# =============================================================================
# ---------- Part 0: Dependancies ----------#
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import derivative
from scipy.integrate import quad
from scipy.interpolate import LSQUnivariateSpline
from functools import partial


from ..misc import aux_func as af
from ..misc import units as cc
#from .. import fluids

# =============================================================================
# ---------- Part 1: Globals ----------#
# =============================================================================


# =============================================================================
# ---------- Part 2: Functions and auxiliary classes ----------#
# =============================================================================
# --- Part 2a: Empiric Rate Functions ---#
def rate_vogel(Pwf, Pws, IP, a=0.8, pb=None):
    """
    Vogel equation
    if pb (bubble pressure) is omitted, returns a normal Vogel Curve
    Otherwise, uses a linear equation until pb and quadratic afterwards
    'a' corrisponds to the coefficient in the quadratic equation (default: 0.8)
    """
    if pb is None or pb > Pws:
        pb = Pws
    c0 = (Pwf <= Pws)
    if not np.all(c0):
        print('Note: Dynamic pressure is higher than static pressure. ' +
              'Rate will be set equal to 0')
    c1 = (Pwf > pb)
    c2 = (Pwf <= pb)
    r1 = IP * (Pws - Pwf)
    r2 = IP * ((Pws - pb) + pb / (1 + a) *
               (1 - (Pwf / pb) * (1 - a) - (Pwf / pb)**2 * a))
    return c0 * (r1 * c1 + r2 * c2)


def rate_fetkovich(Pwf, Pws, C, n):
    """
    Fetkovich equation
    """
    return C * (Pws**2 - Pwf**2) ** n


def rate_quadratic(Pwf, Pws, A, B):
    dp_sq = (Pws**2 - Pwf**2)
    if B == 0:
        return dp_sq / A
    else:
        return (np.sqrt(A**2 + 4 * B * dp_sq) - A) / (2 * B)


# --------------------------------------------
# --- Part 2b: Pseudo pressure functions ---#

def _rate_analytic_mix_fit(Pwf, Pws, A, n, fluid, temperature=None, **kwargs):
    """ for fitting A and n to a pseudo pressure"""
    c = cc.Globals.get_transmissivity_constant()
    return c * A * np.vectorize(fluid.get_mP)(Pws, temperature, Pwf,
                                          kr_function=lambda s: s**n, **kwargs)


def _rate_analytic_single_fit(Pwf, Pws, A, fluid, temperature=None, B=0,
                              **kwargs):
    """ for fitting a constant to a pseudo pressure"""
    c = cc.Globals.get_transmissivity_constant()
    return c * A * np.vectorize(fluid.get_mP)(Pws, temperature, Pwf, **kwargs)


# ---------------------------------------------------
# --- Part 2c: Dimensionless pressure drop Functions ---#
def steadyState_PD(rw, re, skin, a=-0.75):
    """
    Steady state or pseudo steady state dimensionless pressure drop.
    For calculation based on average reservoir pressure and FBHP
    """
    return np.log(re/rw) + skin + a


# =============================================================================
# ---------- Part 3: Classes ----------#
# =============================================================================
class GenericIPR():
    """
    Generic IPR class

    Inputs:
        - a rate function (callable), of Pwf and Pws (in that order)*
        - Default value for Pws and Pwf for calculations.

    *Rate function can have additional arguments as needed. It can output
    a vector or a scalar.

    A flag (self._isScalar) is initialized according to rate_function shape.

    Pws and Pwf should however be scalars or 0-D vectors

    methods:
        __call__: a wrapper for the rate function, defaults Pwf and Pws
        get_AOF: rate at standard pressure
        plot: plots the curve (selected phase if more phases are inputted)
    """

    def __init__(self, rate_function, Pws, Pwf=None,
                 rate_function_derivative=None):
        """ """
        if cc.Globals.verbose:
            print('--IPR created--')

        self.Pws = float(Pws)
        self.Pwf = self.Pws / 2 if Pwf is None else float(Pwf)
        self._fRate = rate_function

        if rate_function_derivative is None:
            def rate_function_derivative(Pwf, *args, **kwargs):
                def rf(P):
                    return self(P, self.Pws, *args, **kwargs)
                return derivative(rf, Pwf)
        self.rate_function_derivative = rate_function_derivative

        r, PI = self._initial_test()
        assert len(r.shape) <= 1, 'Only scalars or 1-d vectors accepted as rate function output'
        self._isScalar = r.shape == ()
        self._phases = r.size

    @property
    def phases(self):
        return self._phases

    def _initial_test(self):
        r = np.array(self(Pwf=self.Pwf, Pws=self.Pws))
        PI = self.get_PI(self.Pwf)
        if cc.Globals.verbose:
            print('\tTesting Rate and IP with Pwf = {:.2f}, Pws = {:.2f}'
                  ''.format(self.Pwf, self.Pws))
            print('\tRate = ', r)
            print('\tPI = ', PI)
            print()
        return r, PI

    def _default_P(self, Pwf=None, Pws=None):
        Pwf = self.Pwf if Pwf is None else Pwf
        Pws = self.Pws if Pws is None else Pws
        return Pwf, Pws

    def __call__(self, Pwf=None, Pws=None, *args, **kwargs):
        Pwf, Pws = self._default_P(Pwf, Pws)
        return self._fRate(Pwf, Pws, *args, **kwargs)

    def get_AOF(self, Pws=None, *args, **kwargs):
        return self(cc.Globals.units['P_STD'], Pws, *args, **kwargs)

    def get_PI(self, Pwf=None, *args, **kwargs):
        """
        returns the rate derivative at the input Pwf.
        """
        Pwf = self.Pws if Pwf is None else float(Pwf)
        return -self.rate_function_derivative(Pwf, *args, **kwargs)

    def plot(self, Pwf=None, Pws=None, func_args=None, func_kwargs=None,
             phase=0, axes=None,
             print_current=False, rate_scale_factor=1, 
             label=None, xlabel=None, ylabel=None,
             plot_args=None, plot_kwargs=None):
        """
        Plots IPR for a given phase

        Inputs:
        - Pwf: for printing the current rate (see below)
        - Pws: Static pressure
        -*args for rate function
        -phase: index of the phase to be input in the case of a vector rate
                formula. Ignored if the rate function returns a scalar
        -Print_current: flag - prints the current rate point (default True)
        -pressure_scale_factor: pressure multiplier
        -rate_scale_factor: rate multiplier
        -label: data label (default: 'IPR')
        -plot_args: args to be passed to plt.plot method (if any)
        -plot_kwargs: kwargs to be passed to plt.plot method (if any)
        -Additional **kwargs to be passed to the rate function

        returns the plt.plot output
        """
        current_rate = self(Pwf, Pws) * rate_scale_factor
        Pwf, Pws = self._default_P(Pwf, Pws)
        pstd = cc.Globals.units['P_STD']
        p_data = np.linspace(pstd, Pws, 15)
        args = list() if func_args is None else list(func_args)
        kwargs = dict() if func_kwargs is None else dict(func_kwargs)
        q_data = np.array([self(p, Pws, *args, **kwargs) for p in p_data])
        q_data *= rate_scale_factor
        if not self._isScalar:
            q_data = q_data[:, phase]
            current_rate = current_rate[phase]

        axes, lines = cc.Globals.plot(q_data, p_data, axes=axes, label=label,
                                      plot_args=plot_args,
                                      ylabel=ylabel, xlabel=xlabel,
                                      plot_kwargs=plot_kwargs)

        handles, labels = lines, [] if label is None else [label]
        if print_current:
#            axes.plot(q_data, np.zeros_like(p_data) + Pwf, 'r--')
            handles.extend(axes.plot(current_rate, Pwf, 'ro'))
            labels.append('Current rate')
            axes.annotate('({:,.1f}, {:,.2f})'.format(current_rate, Pwf),
                          (current_rate, Pwf * 1.05))
            axes.legend(handles, labels, loc='best')
        axes.set_ylim(bottom=0)
        axes.set_xlim(left=0)
        return axes, lines

    def create_fit(self, pdata=None, knots=[], Pws=None, k=3, rate_args=[],
                   **rate_kwargs):
        """
        creates a new IPR from self using an LSQUnivariateSPline object,
        both for the rate and for the IP (rate derivative)
        """
        if pdata is None:
            pdata = np.linspace(cc.Globals.units['P_STD'], self.Pws, 10)
        if knots is None:
            knots = np.array([cc.Globals.units['P_STD'], np.average(pdata), self.Pws])
        else:
            knots = np.asarray(knots)
            knots.sort()
        rate_data = np.asarray([-self(p, Pws, *rate_args, **rate_kwargs)
                                for p in pdata])
        if self._isScalar:
            rate_fit = LSQUnivariateSpline(pdata, rate_data, knots, k=k)
            fIP = rate_fit.derivative()
            IP_fitA = fIP.antiderivative()

            def fRate(Pwf, Pws):
                return IP_fitA(Pws) - IP_fitA(Pwf)

        else:
            IP_list, rate_list = [], []
            for p in range(self.phases):
                rate = rate_data[:, p]
                rate_fit = LSQUnivariateSpline(pdata, rate, knots, k=k)
                fIP = rate_fit.derivative()
                IP_list.append(fIP)
                rate_list.append(fIP.antiderivative())

            def fRate(Pwf, Pws):
                return np.array([r(Pws) - r(Pwf) for r in rate_list])

            def IP_fit(pressure):
                return np.array([r(pressure) for r in IP_list])

        return GenericIPR(fRate, Pwf=self.Pwf, Pws=self.Pws,
                          rate_function_derivative=IP_fit)


class DataFitIPR(GenericIPR):
    """
    Fits an IPR to Pwf vs rate data

    Inputs:
        Pws
        rate_data
        p_data
        equation name:
            'fetkovich': fits C-n
            'vogel': fits the IP
            'quadratic': fits A and B in A*Q + B*Q**2 = (Pws**2 - Pwf**2)
            any other custom equations of the form f(Pwf, Pws, etc.)
            'analytic'
        additional kwargs to be set

    See tutorial for examples

    """
    def __init__(self, Pws, rate_data, p_data, equation, bounds=dict(),
                 Pwf=None, initial_guess=None, **kwargs):
        # default bounds: bounds for equation parameters not specified by user
        # p0: initial guess for fit parameters
        default_bounds = dict()
        if equation == 'fetkovich':
            func = rate_fetkovich
            if 'n' not in kwargs.keys():
                default_bounds['n'] = (0, 1)
                p0 = [1, 1]
            else:
                p0 = [1]
        elif equation == 'vogel':
            func = rate_vogel
            p0 = [1]
            if 'a' not in kwargs.keys():
                default_bounds['a'] = (0.1, 1)
                p0.append(1)
        elif equation == 'quadratic':
            func = rate_quadratic
            default_bounds['A'] = default_bounds['B'] = (0, np.inf)
            p0 = [1, 1]
        elif equation == 'analytic':
            p0 = [1]
            fluid = kwargs.get('fluid')
            if not fluid:
                raise ValueError('Analytic equation must specify a fluid')
            elif fluid.is_mix():
                default_bounds['n'] = (0.2, 4)
                p0.append(1)
                func = _rate_analytic_mix_fit
            else:
                func = _rate_analytic_single_fit
        elif hasattr(equation, '__call__'):
            func = equation
            p0 = initial_guess
            if p0 is None:
                raise ValueError('If a custom equation is input, an '
                                 'initial_guess (vector with length of the '
                                 'args to be fit) must be specified')
        else:
            raise ValueError("Input function '{}' not valid. See DataFitIPR "
                             "documentation for a list of valid inputs"
                             "".format(equation))
        default_bounds.update(bounds)
        kwargs['Pws'] = Pws
        fitObj = af.FitFunc(func, p_data, rate_data, bounds=default_bounds,
                            p0=p0, fixed_parameters=kwargs)
        ratef = fitObj.get_partial()
        ratef.keywords.pop('Pws')
        self.function_fit_args = ratef.keywords

        super().__init__(ratef, Pws, Pwf)
        self.test_points = rate_data, p_data

    def plot_fit(self):
        if self.test_points:
            self.plot(print_current=False, label='Fit')
            plt.plot(self.test_points[0], self.test_points[1], 'o',
                     label='test_points')
            plt.legend(loc='best')
            plt.title('Curve fit and test points')
            plt.xlabel('Rate'), plt.ylabel('Pressure')
        else:
            raise ValueError('Object was generated without test points!!')

    def get_fit_parameters(self):
        return dict(self.function_fit_args)


class AnalyticIPR(GenericIPR):
    """
    Analytic IPR

    Rate function (and derivative) calculated using a pseudo-pressure function,
    a k.h value and a (default) dimensionless pressure drop

    Inputs:
    - mP_factor: pressure-based formula or value for kr(S) / mu(P) / FVF(P)
    - mP_function: integral of the mP_factor of the type lambda Pws, Pwf: value
                   Note the order of the pressure arguments.
                   (If omitted a numerical integration function will be created
                   on initialization)
    - Pws and Pwf default values
    - kh: float value
    - PD: default value for dimensionless pressure drop. PD becomes an argument
          of the IPR __call__ method to allow a transient rate calculation
    - vol_function (optional): Pws-based volume function for multiphase IPR.
                               e.g.: an Rs(P) value for a live oil or an LGR(P)
                               for a wet gas. It should return either a float
                               or a numpy array

    e.g.:
    >>mu, FVF = 1.25, 1
    >>mpf = 1 / FVF / mu
    >>ipr = AnalyticIPR(mpf, Pws=200, Pwf=100, kh=10, PD=10,
                        vol_function=lambda P: np.array(80, P]))
    >>ipr()
        Out: array([   4.2860452 ,  342.88361605,  857.20904013])
    >>ipr.get_volume_ratio(150)
        Out: array([80, 150])
    """
    def __init__(self, Pws, mP_factor=1, mP_function=None, Pwf=None, kh=1,
                 PD=1, vol_function=None, D=0):
        self.kh = float(kh)
        self.PD = float(PD)
        self.D = float(D)

        # Create mP function
        if callable(mP_factor):
            self.fmP_derivative = mP_factor
            if mP_function is None:
                mP_function = lambda Pmax, Pmin: quad(mP_factor, Pmin, Pmax,
                                                      epsabs=0.01)[0]
        else:
            mP_factor = float(mP_factor)
            self.fmP_derivative = lambda P: mP_factor
            if mP_function is None:
                mP_function = lambda Pmax, Pmin: (Pmax - Pmin) * mP_factor
            else:
                raise ValueError('an mP_factor has not been defined as formula'
                                 ' but an mP integral has been defined')
        if not callable(mP_function):
            raise TypeError('mP_function must be a callable or None')
        else:
            self.fmP = mP_function

        # Create mP function
        if vol_function is None:
            self.vol_function = None
        elif callable(vol_function):
            self.vol_function = vol_function
        else:
            raise ValueError('vol_function must be a callable or None')

        def rate_func(Pwf, Pws, PD=None):
            """
            rate function for calculation
            """
            mP = self.fmP(Pws, Pwf)
            q = self.get_mP_multiplier(PD) * mP
            vf = self.get_volume_ratio(Pws)
            if cc.Globals._debug_mode:
                print('Check mode enabled. Rate calculation:\n\tPwf = {:.2f}. '
                      'Pws = {:.2f}\n\tPD = {:.2f}, mP = {}\n\tRate = {}'
                      '\n\tPD = {}'.format(Pwf, Pws, mP, PD, q, PD))
                if self.vol_function:
                    print('\n\tvolume fraction = {}'.format(vf))
                print()
            return q * vf

        def rate_func_der(Pwf, PD=None, Pws=None):
            mPf = self.fmP_derivative(Pwf)
            vf = self.get_volume_ratio(Pws)
            return -self.get_mP_multiplier(PD) * mPf * vf

        super().__init__(rate_func, Pws, Pwf, rate_func_der)

        if cc.Globals._init_test and cc.Globals.verbose:
            # test mP
            mPff = self.fmP_derivative(self.Pwf)
            mPfs = self.fmP_derivative(self.Pws)
            mP_test = self.fmP(self.Pws, self.Pwf)
            print('--Creating Analytic IPR. Testing mP functions:--')
            print('\tPws = {:.2f}, Pwf= {:.2f}'.format(self.Pws, self.Pwf))
            print('\tmP factor at Pws: {}'.format(mPfs))
            print('\tmP factor at Pwf: {}'.format(mPff))
            print('\tmP integral between Pwf and Pws: {}'.format(mP_test))

    def get_mP(self, Pwf=None, Pws=None):
        """
        returns the pseudo pressure integral based in mP factor input function
        """
        Pwf, Pws = self._default_P(Pwf, Pws)
        return self.fmP(Pws, Pwf)

    def get_mP_factor(self, Pwf=None):
        """
        """
        Pwf = self._default_P(Pwf)[0]
        return self.fmP_derivative(Pwf)

    def get_volume_ratio(self, Pws=None):
        """
        returns the ratio of all secondary phases to a unit rate
        of the main phase (if vol_function has been inputted)
        """
        if self.vol_function:
            Pws = self.Pws if Pws is None else float(Pws)
            return np.append(1, self.vol_function(Pws))
        else:
            return 1

    def get_PD(self):
        return self.PD

    def get_mP_multiplier(self, PD=None):
        """
        returns the porous media dependant factor for rate:
            k.h / PD * a
        Where:
            kh: permeability * thickness
            PD: dimensionless pressure drop (defaulted to self.PD)
            a: rate constant
        """
        PD = self.PD if PD is None else float(PD)
        return cc.Globals.rate_constant() * self.kh / self.PD
        cc.Globals.rate_constan()


class FluidIPR(AnalyticIPR):
    """
    IPR created from analytical inflow calculation and a fluid object

    Inputs:
    - Fluid
    - kh: in mD.m or mD.ft (see "units")
    - Pwf and Pws for default rate calculation
    - temperature (default to fluid.ref_T)
    - kr_function for pseudo pressure integration (mixes only)
    - units: if set to "field gas" or 'field oil', transmissivity constant
      will be calculated to output Mscfpd or bpd respectively,
      else it will output sm3/d (default '')
    - PD: default dimensionless pressure drop (default 1)
    - full_output: flag (for mixes only - default False).*
    
    On initialization, reads the arguments for PD and mP and writes stores them

    On __call__: besides Pwf and Pws, keyword arguments are passed to either
    mP, PD or both

    *If set to True (for a mix fluid), a volume relationship function will be created
                   between the main phase and the secondary phases as follows:

                   q1 = quality vector at Pws
                   qs = quality vector at PSTD
                   dsc = standard density of the fluids from the mix
                   x = mass ratio between secondary phases and main phase
                   x = (qs-q1)[1:] / qs[0]
                   v = volume ratio between secondary phases and main phase
                   v = x * dsc[0] / dsc[1:]

            Note that [0] is the main phase and [1:] are all secondary phases

    """

    def __init__(self, fluid, Pws, kh=1, Pwf=None, temperature=None,
                 PD=1, kr_function=None, full_output=False):
        self.fluid = fluid

        # Create mP function
        if temperature is None:
            T = self.fluid.ref_T
        else:
            T = float(temperature)
        mP_args = {'temperature': T}
        if fluid.is_mix():
            mP_args['kr_function'] = kr_function
            mP_args['phase'] = 0
        else:
            full_output = False
        fmP = partial(fluid.get_mP, **mP_args)
        fmP_derivative = partial(fluid.get_mP_factor, **mP_args)

        if self.fluid.is_mix() and full_output:
            if hasattr(fluid, 'get_Rs'):
                rate_ratio = partial(fluid.get_Rs,
                                     temperature=temperature)
            elif fluid.phases >= 2:
                def rate_ratio(P):
                    T = self.fmP.keywords['temperature']
                    xs = self.fluid.get_std_mass_fraction()
                    x = self.fluid.get_mass_fraction(P, T)
                    ds = self.fluid.get_densitySC_vector()
                    x[0] = 0
                    ratio = (xs[1:] - x[1:]) / xs[0] * ds[0] / ds[1:]
                    if cc.Globals._debug_mode:
                        print('Check mode enabled. Checking calculations')
                        print('\tfraction at Pws = {} - {}\n\tfraction at Pstd'
                              ' - {}\n\tratio - {}\n'.format(P, x, xs, ratio))
                    return ratio
        else:
            rate_ratio = None

        super().__init__(Pws, fmP_derivative, fmP, Pwf=Pwf, kh=kh, PD=PD,
                         vol_function=rate_ratio, D=0)
        if fluid.ref_T != T and cc.Globals.verbose:
            print('NOTE: input temperature is different than fluid ref_T. '
                  'Calculations will be adjusted for input temperature')
        elif cc.Globals.verbose and temperature is None:
            print('Temperature defaulted to fluid ref_T')

    def get_temperature(self):
        """ returns temperature used for calculating fluid properties"""
        return self.fmP.keywords['temperature']

    def set_temperature(self, new_value):
        """ sets a new value for temperature """
        self.fmP.keywords['temperature'] = float(new_value)
        self.fmP_derivative.keywords['temperature'] = float(new_value)

# ---------------------------------------------------
# --- Part 4: Defaults ---#



