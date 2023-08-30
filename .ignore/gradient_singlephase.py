# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 10:21:43 2016


Contains the base classes and calculations for flow performance (FP) curves

TODOs:
-check friction calculations / gravity against a web source or similar
-rate VFP construction
-sign management for MD / TVD data

-TESTS!!!

CHEQUEAR:
    signos rate
    unidades
    integrales / formulas
CORREGIR 
    HACER defaults como property


Changelog:
    jun 2018 - put all gradient methods in a class to use global units
@author: Marcelo
"""

# =============================================================================
# ---------- Part 0: Dependancies ----------#
# =============================================================================

# External:

import scipy.constants as SPC
from scipy.integrate import odeint
#from scipy.optimize import brentq
from scipy.interpolate import InterpolatedUnivariateSpline as IUS
#from scipy.misc import derivative
import numpy as np
import matplotlib.pyplot as plt
#from functools import partial

# Internal:
from .. import fluids
from ..misc import aux_func as af
from ..misc import constants
from . import gradient_functions as gf

# =============================================================================
# --- Part 1: Globals ---#
# =============================================================================

__all__ = ['Trayectory']
__all__ += ['GenericGradient', 'Base_ODE_Gradient',
            'PipeGenericGradient', 'PipeCompressibleGradient',
            'PipeIncompressibleGradient'
           ]

verbose = constants.Globals.verbose


# %%
# =============================================================================
# --- Part 2: Functions and auxiliary classes ---#
# =============================================================================
class Trayectory():
    """
    Trayectory class

    Initialized from a pipe length vs elevation data points, provides basic
    tools for plotting and estimating properties such as inclination angles,
    factors and depth vs x in integration routines

    Inputs:
        -s_data: length points
        -y_data: elevation points defaulted to s_data
        -step_points: additional steps to be included in integration routines
                      (if any), such as high discontinuities
        -temperature_data: 2D array (x, T), scalar or None
        -temperature_profile: 'x' or 'y' (default 'y').
                              'x': length based, 'y': depth based.
                              Ignored if data is scalar or None

    Methods:
        __call__: returns elevation for a given x
        get_angle: returns elevation angle
        get_inclination: returns sin(elevation angle)
        plot: plots the trayectory
    """
    _class_name = 'Trayectory'

    @staticmethod
    def _check_arrays(x, y, build_spline=True):
        # 1) same shape
        assert np.shape(x) == np.shape(y), ('y_data and s_data shape mismatch')

        # 2) input is 1d array or float
        assert len(np.shape(x)) <= 1, 's_data must be float or 1d array'

        # 3) Convert floats to array, assert numerical input data
        if np.isscalar(x):
            x = np.append(0, x)
            y = np.append(0, y)
        else:
            x = np.asarray_chkfinite(x, dtype='<f4')
            y = np.asarray_chkfinite(y, dtype='<f4')

        if (np.gradient(x) <= 0).any():
            raise ValueError('pipe length data must be sorted')
        if build_spline:
            return x, y, IUS(x, y, k=1)
        return x, y

    def __init__(self, xy_points, xD_points,
                 xT_points=None, yT_points=None,
                 step_points=None):
        # trayectory
        x, y = xy_points
        x, y, self.pipe_profile = self._check_arrays(x, y)
        self._trayectory = dict()
        dl, dy = x[1:] - x[:-1], y[1:] - y[:-1]
        if not (np.abs(dl) >= np.abs(dy)).all():    # inclination check
            raise ValueError('Values found for x and y imply sin(alpha) > 1')
        self._trayectory = {'l': x, 'y': y,
                            'x': np.cumsum(np.append(0, np.sqrt(dl**2 - dy**2)))}

        # temperature vs depth or vs length
        x = y = T = ius = None
        if xT_points is not None and yT_points is not None:
            raise ValueError('Temperature profile cannot be vs y and vs. x. at the same time')
        elif xT_points is not None:
            x, T, ius = self._check_arrays(*xT_points, True)
            self.Tflag = 'x'
        elif yT_points is not None:
            y, T, ius = self._check_arrays(*yT_points, True)
            self.Tflag = 'y'
        else:
            self.Tflag = False
        self.temperature_profile = ius
        self._temperature_data = {'x': x, 'y': y, 'T': T}


        if step_points is not None:
            self.step_points = np.asarray_chkfinite(step_points, dtype='<f4')
        else:
            self.step_points = None
        self.diameter_data = diameters
        if verbose:
            print('\n-----------')
            print('Trayectory created')

    def __call__(self, s_data):
        """
        returns the elevation corrisponding to the input length points (s_data)
        """
        return self.pipe_profile(s_data)


    def get_diameters(self, s_data):
        if diameters is None:
            raise ValueError('No diameter data available')
        return np.interp(s_data, self.s_data, self.diameter_data)

    def set_diameters(self, s_data, diameter_data):
        pass

    def plot_trayectory(self, axes=None, plot_args=None, plot_kwargs=None):
        """
        plots the pipe trayectory, i.e. elevation vs LD (Lateral Displacement)

        Inputs:
            axes: plt.axes object (if any). Default None
            plot_args: list of arguments for axes.plot or plt.plot method
            plot_kwargs: dict of kwargs for axes.plot or plt.plot method
        """
        x, y = self.l_data, self.y_data
        xl, yl = 'horizontal length', 'elevation'
        return constants.Globals.plot(x, y, axes=axes, xlabel=xl, ylabel=yl,
                                      plot_args=plot_args,
                                      plot_kwargs=plot_kwargs)

    def plot_inclination(self, axes=None, plot_args=None, plot_kwargs=None):
        """
        plots the pipe inclination graph, i.e. elevation vs pipe length

        Inputs:
            axes: plt.axes object (if any). Default None
            plot_args: list of arguments for axes.plot or plt.plot method
            plot_kwargs: dict of kwargs for axes.plot or plt.plot method
        """
        x, y = self.s_data, self.y_data
        xl, yl = 'pipe length', 'elevation'
        return constants.Globals.plot(x, y, axes=axes, xlabel=xl, ylabel=yl,
                                      plot_args=plot_args,
                                      plot_kwargs=plot_kwargs)

    def get_inclination(self, x):
        """
        returns the inclination at the given point in the form sin alpha.
        """
        return self.pipe_profile.derivative()(x)

    def get_angle(self, x):
        """
        returns the inclination at the given point as an angle in radians
        """
        return np.arcsin(self.get_inclination(x))

    def get_step_points(self):
        return self.step_points

#%%
# =============================================================================
# ------ Part 3: classes ---#
# =============================================================================

class GlobalMethods():
    """
    Global methods for all gradient classes
    """
    def __init__(self, default_rate=None, default_P0=None, **defaults):
        if default_rate is None:
            default_rate = 0
        self.phases = int(np.size(default_rate))
        if default_P0 is None:
            p_u, = constants.Globals.get_unit_conversions('pressure')
            default_P0 = 10 * 1e5 / p_u
        defaults.update({'rate': default_rate, 'P0': float(default_P0)})
        self.defaults = defaults

    def _update_defaults(self, names, values):
        try:
            return (self.defaults[k] if v is None else v
                    for k, v in zip(names, values))
        except TypeError:
            return self.defaults[names] if values is None else values

    def get_curve(self, rate_points, *args, **kwargs):
        """
        returns the gradient calculation for a set of rate points
        """
        return np.array([self(r, *args, **kwargs) for r in rate_points])

    def plot(self, rate_points=None, axes=None, phase=0,
             ylabel='BHP', xlabel='Rate', title=None, label=None,
             plot_args=None, plot_kwargs=None, **kwargs):
        """
        plots the FP curve for the given max_rate and min_rate interval
        """
        if rate_points is None:
            rate_points = np.linspace(0, 1, 10) * 3 * self.defaults['rate']
        y = self.get_curve(rate_points, **kwargs)
        x = np.atleast_2d(rate_points)[phase, :]
        return constants.Globals.plot(x, y, axes=axes, ylabel=ylabel,
                                      xlabel=xlabel, title=title, label=label,
                                      plot_args=plot_args,
                                      plot_kwargs=plot_kwargs)


class GenericGradient(GlobalMethods):
    """
    Generic Gradient.
    Inputs a bhp_function of the form lambda rate, P0: P1
    """
    def __init__(self, bhp_function, default_rate=None, default_P0=None):
        self._f_deltaP = bhp_function
        super().__init__(default_rate, default_P0, x0=0, x1=0)

    def __call__(self, rate=None, P0=None, func_args=None, func_kwargs=None):
        rate, P0 = self._update_defaults(['rate', 'P0'], (rate, P0))
        if func_args is None:
            func_args = []
        if func_kwargs is None:
            func_kwargs = dict()
        return self._f_deltaP(rate, P0, *func_args, **func_kwargs)


class Base_ODE_Gradient(GlobalMethods):
    """
    A Generic integrator class for a gradient ODE of the form:
        dP/dx = f(P, x, rate, *args, **kwargs)

    For a more specific pipe gradient use PipeGenericGradient

    Provides the basic methods for integration along the well path and plotting

    inputs:
        - gradient function of the form f(P, x, rate, **kwargs). See notes
        - x0, x1: default integration values
        - default_P0, default_rate: optional defaults for initial pressure
                                    and rate for calculation
        - hstep: integration step
        - inverseFlow: boolean, if set to True, positive rates will be taken
                       as going in decreasing x
        - jacobian: (optional) - gradient jacobian
        - tcrit: critical points for integration (optional)

    NOTE 1: the gradient should return a vector of the form (dPg, dPf, dPv) -
            this is used for discriminating the components of the pressure
            loss in the calculation.

            Pressure and x should be expressed in chosen custom units
            for gradient output (typically bar and m, or psi and ft)

    NOTE 2: The rate argument can be either a scalar or a vector

    """
    p_low_limit = None

    def __init__(self, gradient_function, default_P0=None, default_rate=None,
                 x0=None, x1=None, hstep=None, jacobian=None,
                 inverseFlow=False, tcrit=None):

        self.gradient_function = gradient_function
        self.inverseFlow = bool(inverseFlow)
        self.hstep = float(hstep if hstep is not None else 10)
        if x0 is None:
            x0 = 0
        if x1 is None:
            x1 = 0
        super().__init__(default_rate, default_P0, x0=x0, x1=x1)
        self.odeint_kwargs = {'Dfun': jacobian, 'tcrit': tcrit}

        if constants.Globals.verbose:
            print('\n---Base Gradient Curve created---')
        self._results = None

    def _test(self, *args, **kwargs):
        x0, P0, rate = self._update_defaults(('x0', 'P0', 'rate'), [None] * 3)
        dp = self._adjusted_dP(P0, x0, rate, *args, **kwargs)
        phases = np.size(rate)
        print('Initial test - gradient at defaults conditions')
        print('x0 = {:.2f}, P = {:,.0f}, rate = '
              '{}'.format(x0, P0, rate))
        print('phases: {}'.format(phases))
        print('Calculated mass_rate: {} kg/s'.format(rate))
        print('Calculated gradient: {}'.format(dp))
        print()
        assert np.size(dp) == 3, 'Expected a gradient of 3 components'
        return dp

    def _adjusted_dP(self, pressure, x, rate, *f_args, **f_kwargs):
        """
        wrapper for gradient function integrator in __call__
        """
        return self.gradient_function(pressure, x, rate, *f_args, **f_kwargs)

    def get_gradient(self, P, x, rate, func_args=None, func_kwargs=None):
        """
        returns the pressure gradient at the specified conditions
        """
        if func_args is None:
            func_args = []
        if func_kwargs is None:
            func_kwargs = dict()
        return self._adjusted_dP(P, x, rate, *func_args, **func_kwargs)

    def get_integration_steps(self, x0=None, x1=None):
        """ returns the integration steps used for integration """
        x0, x1 = self._update_defaults(['x0', 'x1'], [x0, x1])
        return np.linspace(x0, x1, np.abs(x1 - x0) // self.hstep + 1)

    @property
    def results(self):
        """
        returns the results from the last calculation (if any)
        """
        return self._results

    @results.deleter
    def del_results(self):
        del self._results

    def __call__(self, rate=None, P0=None, x0=None, x1=None,
                 full_output=0, func_args=None, func_kwargs=None):
        """
        calculates the delta P from x0 to x1 integrating the pressure gradient
        along the x path

        Returns the pressure plot with the gravity and friction components
        Inputs:
        - fluid rate expressed as volume at standard conditions / time
        - P0, x0 (start of calculation) and x1 (end of calculation)
        - full_output: 0: returns only the end point pressure
                       1: returns depth and pressure
                       2: returns depth, pressure, gravity gradient and
                          friction gradient
        - func_args, func_kwargs: additional arguments to be passed to
                                  gradient function

        Returns: end pressure or table (depending on full_output value)

        Notes: use self.odeint_kwargs to set additional arguments to be passed
        to scipy.integrate.odeint routine
        """
        # defaults and checks
        x0, x1, rate, P0 = self._update_defaults(['x0', 'x1', 'rate', 'P0'],
                                                 [x0, x1, rate, P0])
        if self.inverseFlow:
            rate = -rate
        assert full_output in (0, 1, 2), 'full_output must be 0, 1 or 2'

        # integration steps
        l_steps = self.get_integration_steps(x0, x1)

        # gradient function settings
        Pinit = np.array([P0, 0, 0, 0])

        # wrap gradient function as f(P, x) and build output shape
        if func_args is None:
            func_args = []
        if func_kwargs is None:
            func_kwargs = dict()

        def der_fun(p_vector, x):
            P = p_vector[0]
            dP = self._adjusted_dP(P, x, rate, *func_args, **func_kwargs)
            # pressure control (optional)
            if self.p_low_limit:
                if P < self.p_low_limit:
                    ermsg = ('Pressure has gone below STD at x = {:.0f} '
                             ', rate = {}'.format(x, rate))
                    raise ValueError(ermsg)
            return [sum(dP)] + list(dP)

        # Additional kwargs for odeint routine
        odeint_kwargs = self.odeint_kwargs

        if constants.Globals._debug_mode:
            print('Debug mode:')
            print('x0 = {}, P0={}'.format(x0, P0))
            print('Pinit: {}'.format(Pinit))
            print('Function args:', func_args)
            print('Function kwargs:', func_kwargs)
            print('other odeint args:', odeint_kwargs)
            print('Initial gradient: {}'.format(der_fun(Pinit, x0)))

        # solve integration !!
        if x0 != x1:
            sol = odeint(der_fun, Pinit, l_steps, **odeint_kwargs)
            P, dpg, dpf, dpv = sol.T
            P1 = P[-1]
            P0 = P[0]
        else:
            P1 = P0
            P = np.atleast_1d(P0)
            l_steps = np.atleast_1d(x0)
            dpg = dpf = dpv = np.atleast_1d(0)

        self._results = [l_steps, P, dpg, dpf, dpv, rate, x0, x1]
        if constants.Globals.verbose or constants.Globals._debug_mode:
            print('Calculation ended:')
            self.print_calculation_output()

        if full_output == 2:
            return l_steps, P, dpg, dpf
        elif full_output == 1:
            return l_steps, P
        else:
            return P1

    def print_calculation_output(self):
        if self._results is None:
            print('No result data available')
            return None
        l_steps, P, dpg, dpf, dpv, rate, x0, x1 = self._results
        P0, P1 = P[0], P[-1]
        print('\tRate: {}'.format(rate))
        print('\tStart pressure: {:.3f} at x0 = {:.1f}'.format(P0, x0))
        print('\tEnd pressure: {:.3f} at x1 = {:.1f}'.format(P1, x1))
        print()
        print('\tFriction losses: {:.3f}'.format(dpf[-1]))
        print('\tGravity losses: {:.3f}'.format(dpg[-1]))
        print('\tvelocity losses: {:.3f}'.format(dpv[-1]))
        print('\tTotal losses: {:.3f}'.format(P1 - P0))
        print('\tAvg gradient [pres/length]: {:.3f} '.format(
            0 if x0 == x1 else (P1 - P0)/(x1 - x0)))

    def plot_gradient(self, xlabel='Pressure', ylabel='x', **kwargs):
        if self.results is None:
            print('No results available, run the gradient')
            return None
        else:
            x, P = self.get_results()[:2]
        fig, axes = plt.subplots()
        axes, lines = constants.Globals.plot(x, P, axes=axes, xlabel=xlabel,
                                             ylabel=ylabel, **kwargs)
        return axes, lines

    def create_fit(self, rate_points, *args, k=3, **kwargs):
        """
        creates a GenericGradient using an interpolated spline
        inputs:
            rate_points
            *args: gradient function args
            k: spline degree (default 3)
            **kwargs: gradient function kwargs
        """
        p_points = self.get_curve(rate_points, *args, **kwargs)
        ius = IUS(rate_points, p_points, k=k)
        return GenericGradient(ius)


class PipeGenericGradient(Base_ODE_Gradient):
    """
    ODE Integrator with a generic gradient function of the form:
    dPg, dPf, dPv = f(rate, sin_alpha(x), D, eps, pressure, T(x)=None,
                      *args, **kwargs)

    Where sin_alpha and temperature are determined by a trajectory object,
    and rate may be mass, std volumetric, scalar or vector.

    Inputs:
        - gradient_function: gradient function as specified above

    Additional Inputs:
        - trayectory: trayectory object*
        - D: pipe diameter in custom length units
        - eps: pipe roughness in custom length units
        - default_P0, default_rate: optional defaults for initial pressure
                                    and rate for calculation
        - hstep: integration step
        - inverseFlow: boolean, if set to True, positive rates will be taken
                       as going in decreasing x

    *Trayectory object should have:
        -Temperature (if no temperature is provided, fluid ref_T is used)
        -Critical points for integration (if any)
        -initial and final x values appropriate for integration

    """
    def __init__(self, gradient_function, trayectory, D, eps,
                 default_P0=None, default_rate=None,
                 hstep=None, inverseFlow=False):
        af.check_class_type(trayectory, 'Trayectory', 'trayectory')
        self.pipe_path = trayectory
        self._fdP_args = [float(D), float(eps)]
        x0, x1 = self.pipe_path.s_data[[0, -1]]
        tcrit = self.pipe_path.get_step_points()

        super().__init__(gradient_function, default_P0, default_rate, x0, x1,
                         tcrit=tcrit, inverseFlow=inverseFlow, hstep=hstep)

    def _adjusted_dP(self, pressure, x, rate, *args, **kwargs):
        """
        arguments match form
        dP = f(std_rate, sin alpha, D, eps, P, T=None, *args, **kwargs)
        """
        sin_alpha = self.get_inclination(x)
        T = self.get_temperature(x)
        return self.gradient_function(rate, sin_alpha, *self._fdP_args,
                                      *[pressure, T], *args, **kwargs)

    def get_temperature(self, x):
        return self.pipe_path.get_temperature(x)

    def get_inclination(self, x):
        return self.pipe_path.get_inclination(x)

    @property
    def D(self):
        """ pipe diameter """
        return self._fdP_args[0]

    @D.setter
    def D(self, new_value):
        self._fdP_args[0] = float(new_value)

    @property
    def eps(self):
        return self._fdP_args[1]

    @eps.setter
    def eps(self, new_value):
        self._fdP_args[1] = float(new_value)

    def get_area(self):
        """
        returns pipe cross sectional area in m2
        """
        return self.D ** 2 * np.pi / 4


class _PipeFluidGradient(PipeGenericGradient):
    """
    base methods
    """
    def get_density(self):
        """ returns fluid density"""
        pass

    def get_viscosity(self):
        """ returns fluid density"""
        pass

    def get_mass_rate(self, rate=None, densitySC=None):
        """
        returns mass rate in kg/s from a given input volume rate
        measured in custom units at standard conditions
        """
        rate = self._update_defaults('rate', rate)
        if densitySC is None:
            densitySC = self.get_density()
        r_u, d_u = constants.Globals.get_unit_conversions('rate', 'density')
        return rate * r_u * densitySC * d_u

    def get_velocity(self, rate=None, FVF=1):
        """
        returns the velocity in m/s for a given rate in std_vol/time
        """
        r_u, l_u = constants.Globals.get_unit_conversions('rate', 'length')
        rate = self._update_defaults('rate', rate) * r_u
        A = self.get_area() * (l_u ** 2)
        return rate / A * FVF

    def get_friction_factor(self, rate=None, rho=None, mu=None):
        """
        returns the darcy-weibasch friction factor (Fanning * 4)
        """
        rate = self._update_defaults('rate', rate)
        if mu is None:
            mu = self.get_viscosity()
        if rho is None:
            rho = self.get_density()
        return gf.GradientFunctions.get_friction_factor(rate, self.D,
                                                        self.eps, rho, mu)

    def _adjusted_dP(self, pressure, x, rate):
        sin_alpha = self.get_inclination(x)
        return self.gradient_function(rate, sin_alpha, *self._fdP_args)


class PipeCompressibleGradient(_PipeFluidGradient):
    """
    PipeGenericGradient subclass constructed on a single phase fluid gradient,
    density and viscosity variables based on temperature and pressure

    gradient becomes of the form:
        dP/dx = f(rate, sin_alpha(x), D, eps, rho(P, T(x)), mu(P, T(x)))

    Inputs:
        - fluid: single phase fluid object

    Additional Inputs:
        - trayectory: trayectory object*
        - D: pipe diameter in custom length units
        - eps: pipe roughness in custom length units
        - default_P0, default_rate: optional defaults for initial pressure
                                    and rate for calculation
        - hstep: integration step
        - inverseFlow: boolean, if set to True, positive rates will be taken
                       as going in decreasing x

    *Trayectory object should have:
        -Temperature (if no temperature is provided, fluid ref_T is used)
        -Critical points for integration (if any)
        -initial and final x values appropriate for integration

    """

    def __init__(self, fluid, trayectory, D, eps, default_P0=None,
                 default_rate=None, hstep=None, inverseFlow=False):
        f_dP = gf.GradientFunctions.single_phase_gradient
        self.fluid = fluid
        super().__init__(f_dP, trayectory, D, eps, default_P0,
                         default_rate, hstep, inverseFlow)

    def _adjusted_dP(self, pressure, x, rate):
        """
        arguments match gradient form
        dP = f(rate, sin alpha, D, eps, rho, mu, rhoSC)
        """
        sin_alpha = self.get_inclination(x)
        T = self.get_temperature(x)
        rho = self.get_density(pressure, T)
        mu = self.get_viscosity(pressure, T)
        rhoSC = self.get_densitySC()
        return self.gradient_function(rate, sin_alpha, *self._fdP_args,
                                      rho, mu, rhoSC)

    def get_compressibility(self, pressure, temperature=None):
        """
        returns the fluid compressibility, used for jacobian in ODE integration
        """
        return self.fluid.get_compressibility(pressure, temperature)

    def get_velocity(self, rate=None, pressure=None, temperature=None):
        """
        returns the velocity in m/s at the given pressure and temperature
        for a given rate in std_vol/time
        """
        FVF = self.get_densitySC() / self.get_density(pressure, temperature)
        return super().get_velocity(rate, FVF)

    def get_densitySC(self):
        """ returns the density at standard conditions """
        return self.fluid.get_densitySC()

    @af.doc_inherit
    def get_density(self, pressure, temperature=None):
        return self.fluid.get_density(pressure, temperature)

    @af.doc_inherit
    def get_viscosity(self, pressure, temperature=None):
        return self.fluid.get_viscosity(pressure, temperature)

    @af.doc_inherit
    def get_friction_factor(self, rate=None, pressure=None,
                            temperature=None):
        """
        returns the darcy-weibasch friction factor (Fanning * 4)
        """
        rho = self.get_density(pressure, temperature)
        mu = self.get_viscosity(pressure, temperature)
        return super().get_friction_factor(rate, rho, mu)

    @af.doc_inherit
    def get_mass_rate(self, rate=None):
        return super().get_mass_rate(rate, self.get_densitySC())


class PipeIncompressibleGradient(_PipeFluidGradient):
    """
    PipeGenericGradient subclass constructed on a single phase fluid gradient,
    density and viscosity constant

    gradient becomes of the form:
        dP/dx = f(rate, sin_alpha, D, eps, rho(constant), mu(constant))

    Inputs:
        - density (float)
        - viscosity (float)

    Additional Inputs:
        - trayectory: trayectory object*
        - D: pipe diameter in custom length units
        - eps: pipe roughness in custom length units
        - default_P0, default_rate: optional defaults for initial pressure
                                    and rate for calculation
        - hstep: integration step
        - inverseFlow: boolean, if set to True, positive rates will be taken
                       as going in decreasing x

    *Trayectory object should have:
        -Temperature (if no temperature is provided, fluid ref_T is used)
        -Critical points for integration (if any)
        -initial and final x values appropriate for integration
    """
    def __init__(self, density, viscosity, trayectory, D, eps,
                 default_P0=None, default_rate=None, hstep=None,
                 inverseFlow=False):
        super().__init__(gf.GradientFunctions.single_phase_gradient,
                         trayectory, D, eps, default_P0, default_rate,
                         hstep, inverseFlow)
        self._fdP_args.extend([float(density), float(viscosity)])

    @property
    def fluid_viscosity(self):
        """ fluid viscosity """
        return self._fdP_args[-1]

    @fluid_viscosity.setter
    def fluid_viscosity(self, new_value):
        self._fdP_args[-1] = float(new_value)

    @property
    def fluid_density(self):
        """ fluid density """
        return self._fdP_args[-2]

    @fluid_density.setter
    def fluid_density(self, new_value):
        self._fdP_args[-2] = float(new_value)

    @af.doc_inherit
    def get_density(self):
        return self.fluid_density

    @af.doc_inherit
    def get_viscosity(self):
        return self.fluid_viscosity
