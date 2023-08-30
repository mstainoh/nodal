'''
Main equation
dP / dx = f(P, x):

IVP:
Initial pressure (up or bot)

Data:
-mass rates (constant)
-inc(x)
-density(P)
-viscosity(P)
-T(x)
-sigma
'''
from scipy.interpolate import InterpolatedUnivariateSpline as IUS
import numpy as np
from typing import Union, Callable
import scipy.constants as SPC
from scipy.integrate import solve_ivp

import gradient_functions as gf
import fluids

class Tubing:
    def __init__(self, D, eps, x_points, z_points=None, k=3):
        self.D = D
        self.eps = eps
        if z_points is None:
            z_points = x_points
        self.trayectory =  IUS(x_points, z_points, k=k)
        self.get_inclination = self.trayectory.derivative()

    def get_area(self):
        return self.D ** 2 * np.pi / 4
    
    def get_end_points(self):
        return self.trayectory.get_knots()

class SinglePhaseVLP:
    verbose = False
    def __init__(self, tubing: Tubing,
                 fluid: fluids.Fluid,
                 T: Union[Callable[[float], float], float]):
        self.tubing = tubing
        self.fluid = fluid
        if callable(T):
            self.T_function = T
        else:
            self.T_function = lambda x: T

    def get_temperature(self, x):
        return self.T_function(x)
    
    def get_density(self, P, x):
        return self.fluid.get_density(P, self.get_temperature(x))
    
    def get_viscosity(self, P, x):
        return self.fluid.get_viscosity(P, self.get_temperature(x))
    
    def get_compressibility(self, P, x):
        return self.fluid.get_compressibility(P, self.get_temperature(x))

    def get_inclination(self, x):
        return self.tubing.get_inclination(x)
    
    def get_area(self):
        return self.tubing.get_area()

    def get_velocity(self, P, x, mass_rate):
        return mass_rate / self.get_area() / self.get_density(P, x)

    def get_gradient(self, P, x, mass_rate):
        D, eps = self.tubing.D, self.tubing.eps
        density = self.get_density(P, x)
        viscosity = self.get_viscosity(P, x)
        inc = self.get_inclination(x)
        compressibility = self.get_compressibility(P, x)
        return gf.single_phase_gradient(mass_rate=mass_rate, D=D, density=density, viscosity=viscosity, inc=inc, eps=eps, compressibility=compressibility)

    def get_solver(self, mass_rate, BHP=None, THP=None):
        assert np.logical_xor(BHP is None, THP is None), 'one and only one of BHP or THP needs to be specified'
        
        bottom, top = self.tubing.get_end_points()
        if BHP is not None:
            tspan = [bottom, top]
            P0 = BHP
        else:
            tspan = [top, bottom]
            P0 = THP

        def func(t, y):
            x = t
            P = y[0] 
            grad = self.get_gradient(P, x, mass_rate)
            return grad
        
        y0 = [P0, 0, 0, 0]
        return solve_ivp(func, tspan, y0)
    
    def get_total_gradient(self, mass_rate, BHP=None, THP=None):
        solver = self.get_solver(mass_rate, BHP, THP)
        return solver.y[:,-1]
    
    def get_end_pressure(self, mass_rate, BHP=None, THP=None):
        return self.get_total_gradient(mass_rate, BHP, THP)[0]
    

class BiPhaseVLP(SinglePhaseVLP):
    def __init__(self, tubing, liquid: fluids.Fluid, gas: fluids.Fluid, T, sigma_function: Callable[[float, float], float],
                 payne_correction=False, holdup_adj=1):
        self.tubing = tubing
        self.gas = gas
        self.liquid = liquid
        if callable(T):
            self.T_function = T
        else:
            self.T_function = lambda x: T
        self.sigma_function = sigma_function
        self.payne_correction = payne_correction
        self.holdup_adj = holdup_adj

    def get_interfacial_tension(self, P, x):
        T = self.get_temperature(x)
        return self.sigma_function(P, T)

    def get_density(self, P, x):
        raise AttributeError('only valid for single fluids')
    
    def get_viscosity(self, P, x):
        raise AttributeError('only valid for single fluids')

    def get_compressibility(self, P, x):
        raise AttributeError('only valid for single fluids')

    def get_velocity(self, P, x, mass_rate):
        raise AttributeError('only valid for single fluids')

    def get_gradient(self, P, x, mass_rate):
        D, eps = self.tubing.D, self.tubing.eps
        liquid_mass_rate, gas_mass_rate = mass_rate
        inc = self.get_inclination(x)
        T = self.get_temperature(x)
        rho_liquid = self.liquid.get_density(P, T)
        rho_gas = self.gas.get_density(P, T)
        mu_liquid = self.liquid.get_viscosity(P, T)
        mu_gas = self.gas.get_viscosity(P, T)
        sigma = self.get_interfacial_tension(P, T)
        mix_compressibility = (
            liquid_mass_rate / rho_liquid * self.liquid.get_compressibility(P, T) + 
            gas_mass_rate / rho_gas * self.gas.get_compressibility(P, T)
        ) / (liquid_mass_rate / rho_liquid + gas_mass_rate / rho_gas)

        return gf.beggs_brill_correlation(liquid_mass_rate, gas_mass_rate,
                            rho_liquid=rho_liquid, rho_gas=rho_gas,
                            mu_liquid=mu_liquid, mu_gas=mu_gas,
                            D=D, inc=inc, eps=eps, sigma=sigma,
                            mix_compressibility=0,
                            holdup_adj=self.holdup_adj, payne_correction=self.payne_correction,
                            full_output=False, verbose=self.verbose)

if __name__ == '__main__':
    sep = '\n' + '-' * 20 + '\n'
    
    print(sep + 'Tubing test')
    # tubing data
    D = 2.441 * SPC.inch
    eps = 0.045e-3
    x_points = z_points = [-1200,0]
    tubing = Tubing(D,eps,x_points, z_points, k=1)
    print(tubing.get_end_points())
    print(tubing.get_inclination(-100))
 
    print(sep + 'Single phase test')
    # single phase gas
    gas = fluids.Gas({'C1': .99, 'C2': .01})
    
    # VLP
    ref_T = 24 + 273.15
    vlp1 = SinglePhaseVLP(tubing, T=ref_T, fluid=gas)
    P = 100 * SPC.bar
    x = -500

    # test input
    gas_mass_rate = 1000 * gas.get_standard_density() / SPC.day #100000 sm3
    print('mass rate:', gas_mass_rate, 100000 * 0.0008 / SPC.day)
    print('area:', vlp1.get_area())
    print('velocity:', vlp1.get_velocity(P, x, gas_mass_rate))
    print('density:',vlp1.get_density(P, x))
    print('viscosity:',vlp1.get_viscosity(P, x))
    print('compressibility:',vlp1.get_compressibility(P, x))
    
    print('test gradient [Pa/m]', np.array(vlp1.get_gradient(P, -500, gas_mass_rate)))
    print(SPC.g * vlp1.get_density(P, x) * 1)

    THP = 10 * SPC.bar
    BHP = 100 * SPC.bar
    slv = vlp1.get_solver(gas_mass_rate, BHP=BHP)
#    print(slv)
    print(vlp1.get_total_gradient(gas_mass_rate, THP=THP) / SPC.bar)
    print('Test delta P:', vlp1.get_end_pressure(gas_mass_rate, THP=THP) / SPC.bar)
    print('grav gradient:', THP / SPC.bar - vlp1.get_density(THP, -500) * x_points[0] * SPC.g / SPC.bar)


    print(sep + 'Bi phase test')
    rho_l, mu_l, sigma = 1000, 1e-3, 30
    liquid = fluids.Fluid(density_function=lambda P, T: rho_l, viscosity_function=lambda P, T: mu_l)
    fsigma = lambda P, T: sigma
    vlp2 = BiPhaseVLP(tubing, liquid, gas, ref_T, fsigma)
    liquid_mass_rate = 0.00001 * rho_l / SPC.day
    print('Test gradient:', vlp2.get_gradient(BHP, -500, [liquid_mass_rate, gas_mass_rate]) / SPC.bar)

    print('Test delta P', vlp2.get_end_pressure( [liquid_mass_rate, gas_mass_rate], THP=THP) / SPC.bar)
