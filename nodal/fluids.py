# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 14:23:08 2016

@author: Marcelo
"""
# =============================================================================
# ---------- Part 0: Dependancies ----------#
# =============================================================================

from scipy.optimize import newton
import scipy.constants as SPC
import numpy as np
import functools

try:
    from scipy.misc import derivative
except:
    def derivative(func, x, args, delta=0.1):
        fminus = func(x - delta, *args)
        fplus = func(x + delta, *args)
        return (fplus - fminus) / (2 * delta)

# =============================================================================
# ---------- Part 1: Globals ----------#
# =============================================================================
component_aliases = {
    'CarbonDioxide': ['R744', 'co2', 'CO2', 'carbondioxide', 'CARBONDIOXIDE',
                      'CarbonDioxide'],
    'Ethane': ['ethane', 'ETHANE', 'R170', 'Ethane', 'c2', 'C2'],
    'Hydrogen': ['hydrogen', 'HYDROGEN', 'H2', 'R702', 'Hydrogen', 'h2'],
    'HydrogenSulfide': ['H2S', 'HYDROGENSULFIDE', 'HydrogenSulfide', 'h2s'],
    'IsoButane': ['isobutane', 'Isobutane', 'ISOBUTANE', 'R600A', 'R600a',
                  'ISOBUTAN', 'IsoButane', 'iC4', 'ic4'],
    'IsoButene': ['Isobutene', 'ISOBUTENE', 'IBUTENE', 'IsoButene'],
    'Isohexane': ['ihexane', 'ISOHEXANE', 'Isohexane', 'iC6', 'ic6'],
    'Isopentane': ['ipentane', 'R601a', 'ISOPENTANE', 'IPENTANE', 'Isopentane',
                   'iC5', 'ic5'],
    'Methane': ['CH4', 'methane', 'METHANE', 'R50', 'Methane', 'c1', 'C1'],
    'Neopentane': ['neopentn', 'NEOPENTANE', 'Neopentane'],
    'Nitrogen': ['nitrogen', 'NITROGEN', 'N2', 'R728', 'Nitrogen'],
    'Oxygen': ['oxygen', 'OXYGEN', 'O2', 'o2', 'R732', 'Oxygen'],
    'Water': ['water', 'WATER', 'H2O', 'h2o', 'R718', 'Water'],
    'n-Butane': ['nButane', 'butane', 'BUTANE', 'N-BUTANE', 'R600', 'n-Butane',
                 'nC4', 'nc4'],
    'n-Decane': ['Decane', 'decane', 'DECANE', 'N-DECANE', 'n-Decane'],
    'n-Heptane': ['nHeptane', 'Heptane', 'HEPTANE', 'N-HEPTANE', 'n-Heptane', 'nc7'],
    'n-Hexane': ['nHexane', 'Hexane', 'HEXANE', 'N-HEXANE', 'n-Hexane', 'nc6'],
    'n-Nonane': ['nonane', 'NONANE', 'N-NONANE', 'n-Nonane', 'nc9'],
    'n-Octane': ['nOctane', 'Octane', 'OCTANE', 'N-OCTANE', 'n-Octane', 'nc8'],
    'n-Pentane': ['nPentane', 'Pentane', 'PENTANE', 'N-PENTANE', 'R601',
                  'n-Pentane', 'nC5', 'nc5'],
    'n-Propane': ['Propane', 'propane', 'R290', 'C3H8', 'PROPANE',
                  'N-PROPANE', 'n-Propane', 'c3', 'C3', 'nC3']
    }

component_props = {
    'CarbonDioxide': {'M': 0.0440098, 'Pcrit': 7377300.0, 'Tcrit': 304.1282},
    'Ethane': {'M': 0.03006904, 'Pcrit': 4872200.0, 'Tcrit': 305.322},
    'Hydrogen': {'M': 0.00201588, 'Pcrit': 1296400.0, 'Tcrit': 33.145},
    'HydrogenSulfide': {'M': 0.03408088, 'Pcrit': 9000000.0, 'Tcrit': 373.1},
    'IsoButane': {'M': 0.0581222, 'Pcrit': 3629000.0, 'Tcrit': 407.817},
    'IsoButene': {'M': 0.05610632, 'Pcrit': 4009800.0, 'Tcrit': 418.09},
    'Isohexane': {'M': 0.08617536, 'Pcrit': 3040000.0, 'Tcrit': 497.7},
    'Isopentane': {'M': 0.07214878, 'Pcrit': 3378000.0, 'Tcrit': 460.35},
    'Methane': {'M': 0.0160428, 'Pcrit': 4599200.0, 'Tcrit': 190.564},
    'Neopentane': {'M': 0.07214878, 'Pcrit': 3196000.0, 'Tcrit': 433.74},
    'Nitrogen': {'M': 0.02801348, 'Pcrit': 3395800.0, 'Tcrit': 126.192},
    'Oxygen': {'M': 0.0319988, 'Pcrit': 5043000.0, 'Tcrit': 154.581},
    'Water': {'M': 0.018015268, 'Pcrit': 22064000.0, 'Tcrit': 647.096},
    'n-Butane': {'M': 0.0581222, 'Pcrit': 3796000.0, 'Tcrit': 425.125},
    'n-Decane': {'M': 0.14228168, 'Pcrit': 2103000.0, 'Tcrit': 617.7},
    'n-Heptane': {'M': 0.100202, 'Pcrit': 2736000.0, 'Tcrit': 540.13},
    'n-Hexane': {'M': 0.08617536, 'Pcrit': 3034000.0, 'Tcrit': 507.82},
    'n-Nonane': {'M': 0.1282551, 'Pcrit': 2281000.0, 'Tcrit': 594.55},
    'n-Octane': {'M': 0.1142285, 'Pcrit': 2497000.0, 'Tcrit': 569.32},
    'n-Pentane': {'M': 0.07214878, 'Pcrit': 3370000.0, 'Tcrit': 469.7},
    'n-Propane': {'M': 0.04409562, 'Pcrit': 4251200.0, 'Tcrit': 369.89}
    }


for k, v in component_aliases.items():
    component_props.update(dict().fromkeys(v, component_props[k]))

# =============================================================================
# ---------- Part 2: Functions and auxiliary classes ----------#
# =============================================================================

def z_DAK(Pr, Tr, tol=1e-3, verbose=False):
    """
    calculates the compressibility of natural gas
    Uses Dranchuk Abbou-Kassem correlation
    z = 1 + (A1 + A2 / Tr + A3 / Tr^3 + A4 / Tr^4 + A5 / Tr^5) * pr +
    (A6 + A7 / Tr + A8 / Tr^2) * pr^2 - A9 * (A7 / Tr + A8 / Tr^2) * pr^5 +
    A10 * (1 + A11 * pr^2) * (pr^2 / Tr^3)*exp(-A11 * pr^2)
    where pr: pseudoreduced density = 0.27 * Pr / (z . Tr)

    Inputs:
        Pr, Tr, tolerance
    """
    A1 = 0.3265
    A2 = -1.07
    A3 = -0.5339
    A4 = 0.01569
    A5 = -0.05165
    A6 = 0.5475
    A7 = -0.7361
    A8 = 0.1844
    A9 = 0.1056
    A10 = 0.6134
    A11 = 0.721
    C1 = A1 + A2 / Tr + A3 / (Tr ** 3) + A4 / (Tr ** 4) + A5 / (Tr ** 5)
    C2 = A6 + A7 / Tr + A8 / (Tr ** 2)
    C3 = -A9 * (A7 / Tr + A8 / (Tr ** 2))
    C4 = A10 / Tr**3
    C5 = A10 * A11 / Tr**3

    def zf(z):
        """
        basic correlation: z is a function of r,
        where r is a function of z,
        therefore this correlation requires an iterative method
        """
        r = 0.27 * Pr / z / Tr
        return (1 + C1 * r + C2 * r**2 + C3 * r**5 +
                (C4 * r**2 + C5 * r ** 4) * np.exp(-A11 * r**2))

    def dzf(z):
        """
        computes the derivative of zf respect to z,
        using the derivative respect to r and the chain rule
        """
        r = 0.27 * Pr / z / Tr
        dfdr = (C1 + C2 * 2 * r + C3 * 5 * r ** 4 +
                C4 * (2 * r - 2 * A11 * r) * np.exp(-A11 * r**2) +
                C5 * (4 * r ** 3 - 2 * A11 * r) * np.exp(-A11 * r ** 2))
        drdz = - 0.27 * Pr / Tr / z ** 2
        return dfdr * drdz

    testf = lambda x: zf(x) - x    # test function for Newton algorithm
    testdf = lambda x: dzf(x) - 1    # derivative of test function

    try:
        sol = newton(testf, 1, testdf, tol=tol, maxiter=30)
    except RuntimeError as e:
        e.args += ('Cannot perform calculation on Pr = {:.3f}, '
                   'Tr = {:.3f}'.format(Pr, Tr),)
        raise
    if verbose:
        print('Result', sol, 'test', zf(sol), 'error', sol - zf(sol))
    return sol


def z_brill_beggs(Pr, Tr):
    """
    """
    F = 0.3106 - 0.49 * Tr + 0.1824 * Tr ** 2
    E = 9 * (Tr - 1)
    D = 10 ** F
    C = 0.132 - 0.32 * np.log10(Tr)
    B = (0.62 - 0.23 * Tr) * Pr + (
            0.066 / (Tr - 0.86) - 0.037 + 0.32 / 10**E) * Pr**2
    A = 1.39 * (Tr - 0.92) ** 0.5 - 0.36 * Tr - 0.1
    return A + (1 - A) / np.exp(B) + C * Pr ** D


def gas_density(pressure, temperature, z, MW):
    """
    inputs: pressure and temperature un Pa and K, MW in g/mol
    output: density in g/m3
    """
    return (pressure * (MW / 1000) / (SPC.R * temperature))


def gas_visc_LeeGonzalez(pressure, temperature, Pcrit, Tcrit, MW,
                         N2=0, H2S=0, CO2=0):
    """
    Calculates the viscosity of a gas based on pressure, temperature,
    gravity and composition.  Using Lee Gonzalez correlation

    Inputs:
        Pressure, Temperature, Pcrit, Tcrit and MW in Pa, °K and g/mol
        impurities ignored
    Output: viscosity in Pa.s
    """
    T_rank = temperature * 9/5
    z = z_DAK(pressure / Pcrit, temperature / Tcrit)
    rho_SI = gas_density(pressure, temperature, z, MW)
    K = 1e-4 * ((9.4 + 0.02 * MW) * (T_rank)**1.5 / (209 + 19 * MW + T_rank))
    X = 3.5 + 986 / T_rank + 0.01 * MW
    Y = 2.4 - 0.2 * X
    # print(K, X, Y)
    return K * np.exp(X * (rho_SI / 1000) ** Y) * 0.001


def gas_visc_CKB(pressure, temperature, Pcrit, Tcrit, MW, N2=0, H2S=0, CO2=0):
    """
    Carr, Kobayashi, and Burrows correlation for viscosity of natural gas
    Inputs:
        Pressure, Temperature, Pcrit, Tcrit and MW in Pa, °K and g/mol
        N2, CO2 and H2S fractions
    Output: viscosity in Pa.s
    """
    SG = MW / 28.97
    Pr = pressure / Pcrit
    Tr = temperature / Tcrit
    x = -2.4621182
    x += 2.97054714 * Pr
    x += -0.286264054 * Pr ** 2
    x += 0.008054205 * Pr ** 3
    x += 2.80860949 * Tr
    x += -3.49803305 * Tr * Pr
    x += 0.36037302 * Tr * (Pr ** 2)
    x += -0.01044324 * Tr * (Pr ** 3)
    x += -0.793385684 * (Tr ** 2)
    x += 1.39643306 * (Tr ** 2) * Pr
    x += -0.149144925 * (Tr ** 2) * (Pr ** 2)
    x += 0.004410155 * (Tr ** 2) * (Pr ** 3)
    x += 0.083938718 * (Tr ** 3)
    x += -0.186408848 * (Tr ** 3) * Pr
    x += 0.020336788 * (Tr ** 3) * (Pr ** 2)
    x += -0.000609579 * (Tr ** 3) * (Pr ** 3)

    tF = SPC.convert_temperature(temperature, 'K', 'F')
    visc1 = ((1.709e-5 - 2.062e-6 * SG) * tF + 8.188e-3 - 6.15e-3 *
             np.log10(SG))
    d1 = N2 * (8.48 * np.log10(SG) + 9.59) / (10**6)
    d2 = CO2 * (9.08 * np.log10(SG) + 6.24) / (10**6)
    d3 = H2S * (8.49 * np.log10(SG) + 3.73) / (10**6)
    visc1 += d1 + d2 + d3
    return visc1 / Tr * np.exp(x) * 0.001


# --- water ---#
# def cw_osif(pressure, temperature, salinity):
#    """
#    osif correlation for water compressibility
#    inputs: pressure and temperature in Pa and °K
#    output: compressibility
#    """
#    dp = 7.033 / SPC.psi * (pressure)
#    ds = 541.5 * salinity
#    T_F = SPC.convert_temperature(temperature, 'K', 'F')
#    dt = -537 * T_F + 403.3
#    return (dp + ds + dt)**-1


def muw_Shell_McCain(pressure, temperature, salinity):
    """
    calculates the water viscosity
    Uses Shell correlation + McCain pressure correction
    inputs: pressure and temperature in Pa and K
    output: viscosity in Pa.s
    """
    B = 11.897 - 5.943e-2 * temperature + 6.422e-5 * (temperature ** 2)
    muw = 1 + 2.765e-3 * salinity * np.exp(B)
    P_psi = pressure / SPC.psi
    p_correction = 0.9994 + 4.0295e-5 * P_psi + 3.1062e-9 * P_psi**2
    return muw * p_correction * 0.001


def dw_chierici(pressure, temperature, salinity):
    """
    Chierici correlation for density.

    Input: salinity in g/l, pressure and temperature in SI (Pa and K)

    Output: density in kg/m3
    """
    T_K = temperature
    P_MPa = pressure / 1e6
    return (730.6 + 2.025 * T_K - 3.8e-3 * T_K**2 +
            (2.362 - 1.197e-2 * T_K + 1.835e-5 * T_K**2) * P_MPa +
            (2.374 - 1.024e-2 * T_K + 1.49e-5 * T_K**2 - 5.1e-4 * P_MPa) *
            salinity)


def cw_chierici(pressure, temperature, salinity):
    """
    chierici compressibility based on density correlation derivative

    Input: salinity in g/l, pressure and temperature in Pa and °K
    Output: compressibility
    """
    P_der = 1e-6
    rho_SI = dw_chierici(pressure, temperature, salinity)
    rho_der_SI = ((2.362 - 1.197e-2 * temperature + 1.835e-5 *
                   temperature**2) * P_der + (- 5.1e-4 * P_der) * salinity)
    return rho_der_SI / rho_SI

# =============================================================================
# ---------- Part 3: Classes ----------#
# =============================================================================
class Fluid:
    def __init__(self, density_function, viscosity_function, compressibility_function=None, ref_T=288.15):
        self.density_function = density_function
        self.viscosity_function = viscosity_function
        if compressibility_function is None:
            self.compressibility_function = lambda P,T: 0
        else:
            self.compressibility_function = compressibility_function
        self.ref_T = ref_T

    def get_density(self, P, T=None):
        if T is None:
            T = self.ref_T
        return self.density_function(P,T)

    def get_viscosity(self, P, T=None):
        if T is None:
            T = self.ref_T
        return self.viscosity_function(P,T)

    def get_compressibility(self, P, T=None):
        if T is None:
            T = self.ref_T
        return self.compressibility_function(P,T)


class Water(Fluid):
    def __init__(self, salinity:float, ref_T:float=288.15):
        self.salinity = salinity
        self.ref_T = ref_T

    @property
    def salinity(self):
        return self._salinity
    
    @salinity.setter
    def salinity(self, new_value:float):
        self._salinity = new_value
        self.viscosity_function = functools.partial(muw_Shell_McCain, salinity=new_value)
        self.compressibility_function = functools.partial(cw_chierici, salinity=new_value)
        self.density_function = functools.partial(dw_chierici, salinity=new_value)


class Gas(Fluid):
    def __init__(self, composition, ref_T=288.15, viscosity_correlation=0, z_correlation=0):
        
        self.ref_T = ref_T
        self.composition = dict(composition)
        checksum = sum(self.composition.values())
        assert checksum==1, 'composition must add up to 1. It adds to {}'.format(checksum)

        MW = Pcrit = Tcrit = 0
        for k, v in self.composition.items():
            MW += v * component_props[k]['M']
            Pcrit +=  v * component_props[k]['Pcrit']
            Tcrit +=  v * component_props[k]['Tcrit']
        self.MW = MW * 1e3
        self.Tcrit = Tcrit
        self.Pcrit=Pcrit

        
        if viscosity_correlation==0:
            self.viscosity_function = gas_visc_LeeGonzalez
        elif viscosity_correlation == 1:
            self.viscosity_function = gas_visc_CKB
        else:
            raise ValueError ('viscosity correlation must be 0 or 1')
        
        if z_correlation == 0:
            self.z_function = z_brill_beggs
        elif z_correlation == 1:
            self.z_function = z_DAK
        else:
            raise ValueError('z_correlation must be 0 or 1')

    def get_z(self, P, T=None):
        if T is None:
            T = self.ref_T
        Pr, Tr = P/self.Pcrit, T/self.Tcrit
        return self.z_function(Pr, Tr)

    def get_density(self, P, T=None):
        if T is None:
            T = self.ref_T
        z = self.get_z(P,T)
        MW = self.MW
        return gas_density(P, T, z, MW)
    
    def get_standard_density(self):
        return self.get_density(SPC.atm, 288.15)

    def get_impurities(self):
        out = dict().fromkeys(['H2S', 'CO2', 'N2'], 0)
        for k, v in self.composition.items():
            if k in component_aliases['HydrogenSulfide']:
                out['H2S'] += v
            elif k in component_aliases['CarbonDioxide']:
                out['CO2'] += v
            elif k in component_aliases['Nitrogen']:
                out['N2'] += v
        return out

    def get_viscosity(self, P, T=None):
        if T is None:
            T = self.ref_T
        return self.viscosity_function(P, T, self.Pcrit, self.Tcrit, self.MW, **self.get_impurities())

    def get_compressibility(self, P, T=None):
        if T is None:
            T = self.ref_T
        dzdp = derivative(self.get_z, P, args=[T])
        z = self.get_z(P, T)
        return 1 / P - 1 / z * dzdp

if __name__ == '__main__':
    composition = {'c1': 0.99, 'CO2': 0.01}
    mygas = Gas(composition)
    P, T = 100 *SPC.bar, SPC.convert_temperature(100, 'C', 'K')
    print('z', mygas.get_z(P,T))
    print('rho', mygas.get_density(P,T))
    print('mu cP', mygas.get_viscosity(P,T) * 1000)
    print('compressibility bar-1', mygas.get_compressibility(P,T) * SPC.bar)
    
