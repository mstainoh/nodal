# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 12:18:50 2018

@author: mstainoh
"""

# =============================================================================
# ---------- Part 0: Dependancies ----------#
# =============================================================================

import numpy as np
import scipy.constants as SPC

#from ..misc import constants

# ===========================================================================
# ---------- Part 1: Globals ----------#
# =============================================================================
_debug = False

roughness_values = {
    'Cast Iron': 0.26e-3,
    'Galvanized Iron': 0.15e-3,
    'Asphalted Cast Iron': 0.12e-3,
    'Commercial or Welded Steel': 0.045e-3,
    'PVC, Glass, Other Drawn Tubing': 0.0015e-3}


# ==========================================================================
# ---------- Part 2: Functions and auxiliary classes ----------#
# =============================================================================
# SI input and output

# ---------- General funcions ----------#
def reynolds(v, D, density, viscosity):
    """
    Returns the reynolds number v * D * rho / mu

    Parameters
    ----------
    v: float
        velocity in m/s
    D: float
        diameter in m
    density: float
        density in kg/m3
    viscosity: float
        viscosity in Pa.s
        

    Example
    -------
    reynolds (1, 0.0254, 1000, 0.001)
    """
    return v * D * density / viscosity


def froid(v, D):
    """
    Returns the Froid number v / srqt(g * D)

    Parameters
    ----------
    v: float
        velocity in m/s
    D: float
        diameter in m

    Example
    -------
    froid(1, 0.0254)
    """
    return v / np.sqrt(SPC.g * D)


def chen_approx(re, D, eps):
    """ chen aproximation for turbulent friction factor"""
    re = np.clip(re, 2e3, re)
    return (-4 * np.log10(0.2698 * (eps/D) - 5.0452 / re * np.log10(
            0.3539 * (eps/D)**1.1098 + 5.8506 / re ** 0.8981)))**-2


def find_friction_factor(re, D, eps, fanning=True):
    """
    Returns the friction factor for fluid flow pressure drop.
    Uses Chen approximation.

    Parameters
    ----------
    re: float
        reynolds number
    D: float
        pipe diameter (same units as eps)
    eps: float
        pipe roughness, same units as diameter
    fanning: bool
        True to return fanning friction factor, False to return Darcy Weibach
        (f_DW = f_Fanning * 4). Default True
    """
    if np.any(re < 0):
        raise ValueError('Reynolds value cannot be negative')
    f = np.zeros_like(re, dtype=float)
    m1 = (re > 0) & (re <= 2000)
    m2 = (re > 2000) & (re < 4000)
    m3 = re >= 4000
    np.putmask(f, m1, 16 / re)
    np.putmask(f, m2 | m3, chen_approx(re, D, eps))
    np.putmask(f, m2, (f * (re - 2e3) + (16 / re) * (4e3 - re)) / 2e3)
    return f * (1 if fanning else 4)


def mach(density, pressure, velocity, gamma=1.4):
    """
    returns the mach number M = v / c
    with c = speed of sound = sqrt(gamma * pressure / density)

    Parameters
    ----------
    density: float
        density in kg/m3
    pressure: float
        pressure in Pa
    velocity: float 
        flow velocity in m/s
    gamma: float
        gas gamma constant
    """
    c = np.sqrt(gamma * pressure / density)
    return velocity / c


# %%
# ---------- single phase funcions ----------#
def single_phase_gradient(mass_rate, D, density, viscosity=1e-3, inc=1,
                          eps=.15e-3, compressibility=0, verbose=False):
    """
    generic single phase fluid gradient (adiabatic)

    Parameters
    ----------
    mass_rate: float
        mass_rate in kg/s
    D: float
        diameter in m
    density: float
        fluid density in kg/m3
    viscosity: float
        fluid viscosity in Pa.s (default 1)
    inc: float between -1 and 1
        inclination (default 1)
    eps: float
        pipe roughness in m (default 0.15 mm)
    compressibility: float
        compressibility of fluid in Pa**-1 (default 0)

    Returns
    ----------
    total gradient, gravity gradient, friction gradient, momentum gradient

    Notes
    ----------
    pressure gradient due to change in momentum assumes density change from
    temperature is negligible respect to density change from pressure. i.e.:
        d(rho)/dx = d(rho)/dP * dP/dx

    Raises
    ----------
    Warning if flow is close to supersonic (rho * comp * v**2 >0.9),
    ValueError if flow is supersonic
        
    """
    assert np.all(inc <= 1) & (inc >= -1), 'inclination must be -1 < inc < 1'

    # gravity part
    dPg = - SPC.g * inc * density

    # friction part
    v = np.abs(mass_rate / density / (D**2 / 4 * np.pi))
    re = reynolds(v, D, density, viscosity)
    f = find_friction_factor(re, eps, D, fanning=False)
    dPf = - f / (2 * D) * density * v**2 * np.sign(mass_rate)

    # momentum part
    Eh = compressibility * density * v ** 2
    if np.any(Eh >= 1):
        raise ValueError('Supersonic flow encountered')
    elif np.any(Eh > 0.9):
        raise Warning('Flow is close to supersonic')
    dPv = (dPf + dPg) * Eh / (1 - Eh)

    if verbose:
        print('DEBUG MODE')
        print('Kinetic factor dPv / dPtotal : {:.5f}'.format(Eh / (1 - Eh)))
        print('Mass Rate = {:.2f} kg/s'.format(mass_rate))
        rate = mass_rate / density
        print('Actual Rate = {:.2f} m3/s'.format(rate))
        print('Inc = {:.2f}'.format(inc))
        print('D = {}, eps = {} meters'.format(D, eps))
        print('Density = {:.3f} kg/m3'.format(density))
        print('Viscosity = {:.3f} cP'.format(viscosity))
        print('Speed = {:.3f} m/s'.format(rate / (D**2 / 4 * np.pi)))
        print('friction factor = {:.5f}'.format(f))
        print('kinetic equivalent pressure')
    return dPg + dPf + dPv, dPg, dPf, dPv


# %% Beggs and brills equation
# ---------- multiphase funcions ----------#
def beggs_brill_flowmap(Cl, NFr):
    """
    returns the flow regime based on liquid fraction and Froid number
    ouput index in ['segregated', 'intermittent', 'distributed',
                    'transition']

    Parameters
    ----------
    Cl: float
        no-slip liquid fraction = volume of liquid / total volume
    Fr: float
        Froid number
    """
    L1 = 316 * (Cl ** 0.302)
    L2 = 0.0009252 * (Cl ** -2.4684)
    L3 = 0.1 * (Cl ** -1.4516)
    L4 = 0.5 * (Cl ** -6.738)

    # vectorized !
    m0 = ((Cl <= 0.01) & (NFr <= L1)) | ((Cl > 0.01) & (NFr <= L2))
    m1 = (((Cl > 0.01) & (Cl <= .4) & (NFr > L3) & (NFr <= L1)) |
          ((Cl > 0.4) & (NFr > L3) & (NFr <= L4)))
    m2 = ((Cl <= .4) & (NFr > L1)) | ((Cl > .4) & (NFr > L4))
    m3 = ((Cl > .01) & (NFr > L2) & (NFr <= L3))
    if not np.all(m3 + m2 + m1 + m0):
        m = ~(m3 + m2 + m1 + m0)
        raise ValueError('invalid values for Cl {} and NFr {}'.format(Cl[m], NFr[m]))
    return m1 * 1 + m2 * 2 + m3 * 3
#


def beggs_brill_holdup(i, Cl, NFr, Nlv, angle, verbose=False):
    """
    auxiliary function for calculating the liquid holdup
    """
    # STEP 1: horizontal liquid holdup
    if i == 0:
        a, b, c = 0.98, 0.4846, 0.0868   # segregated
    elif i == 1:
        a, b, c = 0.845, 0.5351, 0.0173  # intermittent
    elif i == 2:
        a, b, c = 1.065, 0.5824, 0.0609    # distributed
    elif i == 3:
        # recursive calculation
        L2 = 0.0009252 * (Cl ** -2.4684)
        L3 = 0.1 * (Cl ** -1.4516)
        A = (L3 - NFr) / (L3 - L2)
        H1, H2 = (beggs_brill_holdup(k, Cl, NFr, Nlv, angle)
                  for k in (1, 2))
        return A * H1 + (1 - A) * H2
    else:
        raise ValueError('i must be an integer between 0 and 3')
    El0 = a * Cl ** b / NFr ** c

    # STEP 2: generic angle holdup
    if angle > 0 and i == 2:
        C = 0
        b_theta = 1
    else:
        if angle <= 0:
            d, e, f, g = 4.7, -0.3692, 0.1244, -0.5056
        elif i == 0:
            d, e, f, g = 0.011, -3.768, 3.539, -1.614
        elif i == 1:
            d, e, f, g = 2.96, 0.305, -0.4473, 0.0978
        C = (1 - Cl) * np.log(d * (Cl ** e) * (Nlv ** f) * (NFr ** g))
        b_theta = 1 + np.clip(C, 0, np.inf) * (
                np.sin(1.8 * angle) - np.sin(1.8 * angle)**3 / 3)

    if verbose:
        print('Intermediate holdup calculations:')
        print('\t0-angle holdup: {:.3f}'.format(El0))
        print('\tC: {:.3f}'.format(C))
        print('\tangle adjustment factor: {:.3f}'.format(b_theta))
    return (El0 * b_theta)


def beggs_brill_correlation(liquid_mass_rate, gas_mass_rate,
                            rho_liquid, rho_gas,
                            mu_liquid, mu_gas,
                            D, inc=1, eps=.15e-3, sigma=30,
                            compressibility_liquid=0, compressibility_gas=0, mix_compressibility=None,
                            holdup_adj=1, payne_correction=True,
                            full_output=False, verbose=False):
    """
    Beggs and Brill correlation for multiphase (liquid-gas) flow

    Parameters
    ----------
    liquid_mass_rate: float
        liquid mass rate (kg/s)
    gas_mass_rate: float
        gas mass rate (kg/s)
    rho_liquid: float
        liquid density (kg/m3)
    rho_gas: float
        gas density (kg/m3)
    mu_liquid: float
        liquid viscosity in Pa.s
    mu_gas: float
        gas viscosity in Pa.s
    D: float
        pipe diameter in m
    inc: float
        pipe inclination, i.e. sin angle (default 1)
    sigma: float
        surface tension in dynas/cm (default 30)
    eps: float
        pipe roughness in m (default .15mm)
    compressibility_gas: float
        compressibility of the liquid part (used for calculating momentum loss) - default 0
    compressibility_liquid: float
        compressibility of the gas part (used for calculating momentum loss) - default 0
    mix_compressibility: float or None
        compressibility of the mixture: if set to None, calculates it using volumetric average (default), otherwise liquid and gas
        compressibility values are ignored
    holdup_adj: float
        multiplicator factor for adjusting correlation holdup;
        holdup will be limited to 0-1 in any case (default 1)
    payne_correction: boolean
        applies Payne et al correction to holdup (as suggested in Kermitt Brown book)
    full_output: boolean
        False (default) - outputs pressure gradient
        True - outputs a dictionary with various results
                   flow regime index*, NFr, Liquid Fraction and Liquid holdup

    * 0: 'segregated', 1: 'intermittent', 2: 'distributed', 3: 'transition'

    Returns:
        gradient in the form (dP_total, dPg, dPf, dPv=0), or dictionary

        Being in the flow direction, gradient values are negative if rate
        input is positive, and positive if rate input is negative
    """
    grad = np.zeros(4)

    # correct negative rates and check if flow is uphill or downhill
    if liquid_mass_rate > 0 and gas_mass_rate >= 0:
        inverse_flow = False
        uphill = inc > 0
    elif liquid_mass_rate <= 0 and gas_mass_rate <= 0:
        # for negative rates the inclination and rates are temporarily reversed
        inverse_flow = True
        uphill = inc < 0
        liquid_mass_rate = np.abs(liquid_mass_rate)
        gas_mass_rate = np.abs(gas_mass_rate)
        inc = -inc
    else:
        raise ValueError('counterflow not allowed (ql = {:.3f} kg/s, qg = '
                         '{:.3f} kg/s'.format(liquid_mass_rate, gas_mass_rate))

    # pass rates to m3/s, mass_rate in kg/s
    ql = liquid_mass_rate / rho_liquid
    qg = gas_mass_rate / rho_gas
    Cl = ql / (ql + qg)
    total_mass_rate = liquid_mass_rate + gas_mass_rate

    # mix no slip velocity in m/s
    A = np.pi * D ** 2 / 4
    v_mix = (ql + qg) / A

    # No Slip - mix properties
    mu_NS = Cl * mu_liquid + (1 - Cl) * mu_gas
    rho_NS = Cl * rho_liquid + (1 - Cl) * rho_gas

    # Dimensionless numbers
    NFr = froid(v_mix, D) ** 2
    ReNs = reynolds(v_mix, D, rho_NS, mu_NS)
    vsl = ql / A
    Nlv = vsl * (rho_liquid / (0.001 * sigma * SPC.g))**0.25

    if mix_compressibility is None:
        mix_compressibility = compressibility_liquid * Cl + compressibility_gas * (1 - Cl)
    if verbose:
        print('\n' + '-' * 20)
        print('Results for begg brill calculation')
        print('NOTE: Rates are negative, gradients are in the direction of '
              'positive flow' if inverse_flow else 'Note: Rates are positive')
        print('Inputs:')
        print('\tFlow Direction: ' + ('Uphill' if uphill else 'Downhill'))
        print('\tMass rate: {:.3f} kg/s liq, {:.3f} kg/s gas, {:.3f} kg/s '
              'total'.format(liquid_mass_rate, gas_mass_rate, total_mass_rate))
        print('\tDensity: {:.3f} kg/m3 liq, {:.3f} kg/m3 gas'.format(
                rho_liquid, rho_gas))
        print('\tViscosity: {:.6f} Pa.s liq, {:.6f} Pa.s gas'.format(
                mu_liquid, mu_gas))
        print('\tSigma: {:.3f}, inc: {:.3f}'.format(sigma, inc))
        print('\tCompressibility: {:.6f}'.format(mix_compressibility))
        print('Intermediate calculations:')
        print('\tVolume rate: {:.3f} m3/s liq, {:.3f} m3/s gas, {:.3f}'
              ' m3/s total'.format(ql, qg, ql + qg))
        print('\tNo Slip velocity: {:.3f} m/s (liquid), {:.3f} m/s (gas), '
              '{:.3f} m/s (mix)'.format(ql/A, qg/A, v_mix))
        print('\tNo Slip density: {:.1f}'.format(rho_NS))
        print('\tNo Slip viscosity: {:.3f}'.format(mu_NS))
        print('Dimensionless numbers:')
        print('\tLiquid fraction (Cl): {:.3f}'.format(Cl))
        print('\tNFr (Froid**2) number: {:.3f}'.format(NFr))
        print('\tLiquid velocity number: {:.3f}'.format(Nlv))
        print('\tReynolds: {:,.1f}'.format(ReNs))

    # flow regime determination
    i = beggs_brill_flowmap(Cl, NFr)
    regime = ['segregated', 'intermittent', 'distributed', 'transition'][i]

    # Holdup correlation based on flow regime
    El = beggs_brill_holdup(i, Cl, NFr, Nlv, np.arcsin(inc), verbose=verbose)

    # Holdup corrections
    if payne_correction:
        El *= (0.924 if inc > 0 else 0.685) # Payne et al. correction
    El *= holdup_adj
    El = np.clip(El, 0, 1)

    # dPg
    rho_mix = rho_liquid * El + rho_gas * (1 - El)
    grad[1] = - rho_mix * inc * SPC.g

    # dPf
    fNs = find_friction_factor(ReNs, eps, D)
    if El == 0:
        f = fNs
    else:
        y = Cl / El ** 2
        if 1 <= y < 1.2:
            s = np.log(2.2 * y - 1.2)
        else:
            y = np.log(y)   
            s = y / (-0.0523 + 3.182 * y - 0.8725 * y**2 + 0.01853 * y**4)
        f = fNs * np.exp(s)
    grad[2] = - 2 * f / D * (v_mix ** 2) * rho_NS

    # Dimensionless kinieti energy term
    # dPv
    Eh = mix_compressibility * rho_mix * v_mix ** 2
    if np.any(Eh >= 1):
        raise ValueError('Supersonic flow encountered')
    elif np.any(Eh > 0.9):
        raise Warning('Flow is close to supersonic')
    grad[3] = (grad[1] + grad[2]) * Eh / (1 - Eh)

    if inverse_flow:    # for negative rates the gradient is reversed back
        grad = -grad
    grad[0] = grad.sum()
    if verbose:
        print('Calculations:')
        print('\tFlow Regime:', regime)
        gtext = '\t{} Gradient: {:.3f} Pa/m'
        print('\tCalculated Holdup: {:.3f}'.format(El))
        print('Results:')
        print('\tMix Density: {:.3f} (adj.)'.format(rho_mix))
        print('\tMix friction factor: {:.5f}'.format(f))
        print(gtext.format('Gravity', grad[1]))
        print(gtext.format('Friction', grad[2]))
        print(gtext.format('Momentum', grad[3]))
        if El != 0:
            print('\tlog_y factor: {:.3f}, s factor: {:.3f}'.format(y, s))
        print('\tFriction factor: {:.4f} (NS), {:.4f} (actual)'
              ''.format(fNs, f))
        print('-' * 20 + '\n')
    if full_output:
        return {'result': grad, 'flow_regime': regime, 'NFr': NFr,
                'liquid_holdup': El, 'liquid_fraction': Cl, 'fNs': fNs,
                'f': f, 'mixture_density': rho_mix,
                'liquid_velocity_number': Nlv, 'ReNs': ReNs}
    else:
        return grad

# %%
# =========================================================================== #
# ---------- Part 5: Tests ----------#
# =========================================================================== #

def test1():
    #Example 4.7 kermit brown
    # Data
    qos = 10000 * SPC.barrel / SPC.day
    qgs = 10e6 * SPC.foot**3 / SPC.day
    D = 6 * SPC.inch
    inc = 1
    P = 1700 * SPC.psi
    T = SPC.convert_temperature(180, 'F', 'K')
    
    # Fluid data
    sigma = 8.41
    eps = 6e-6 * SPC.foot
    mu_liquid = 0.97 * 1e-3
    mu_gas = 0.016 * 1e-3
    Bo = 1.197
    Bg = 0.0091
    Rs = 281 * SPC.foot**3 / SPC.barrel
    z = .853
    dos = 141.5 / (131.5 + 33) * 1000
    dgs_free = 0.70 * 1.225
    dgs_diss = 0.88 * 1.225
    
    # Density calculation
    rho_liquid = (dos + dgs_diss * Rs) / Bo
    rho_gas = dgs_free / z * (288.15 / T) * (P / SPC.atm)
    
    print('Oil density: {:.2f} (calculated), {:.2f} (actual)'
          ''.format(rho_liquid, 47.61 * SPC.pound / SPC.foot**3))
    print('Gas density: {:.2f} (calculated), {:.2f} (actual)'
          ''.format(rho_gas, 5.88 * SPC.pound / SPC.foot**3))
    
    # rate calculations
    qo = qos * Bo
    qg = (qgs - qos * Rs) * Bg
    q = np.array([qo, qg])
    A = D ** 2 * np.pi / 4
    
    print('Qo: {:.3f} ft3/s, Qg: {:.3f} ft3/s'.format(*(q / SPC.foot**3)))
    # 0.778 , 0.757
    
    print('Vsl = {:.2f} ft/s, Vsg = {:.2f} ft/s'.format(*(q / A / SPC.foot)))
    # 3.97, 3.86
    calc = beggs_brill_correlation(qo * rho_liquid, qg * rho_gas, rho_liquid, rho_gas,
                                   mu_liquid, mu_gas, D, inc, eps, sigma, full_output=True,
                                   verbose=True)
    
    # NOTA: la funcion usa el factor de friccion de fanning
    # mientras que el libro reporta el de Darcy Weybash (fanning * 4)
    
    book_results = {'NFr': 3.81, 'Liquid_fraction': 0.507,
                    'liquid_holdup_uncorrected': 0.574,
                    'flow_regime': 'intermittent',
                    'C': -0.048,
                    'liquid_holdup': 0.530,
                    'ReNs': 3.15e5,
                    'y': 1.805, 's': .3873, 'f': 0.0228 / 4, 'fNs': 0.0155 / 4,
                    'dPf': 1.17 * SPC.psi / 144 / SPC.foot,
                    'dPg': 28 * SPC.psi / 144 / SPC.foot}
    
    txt = '\tBook result: {:.3f}, Calc result: {:.6f}, error: {:.2%}'
    
    for k in ( 'NFr', 'ReNs', 'f', 'fNs', 'liquid_holdup'):
        r1, r2 = book_results[k], calc[k]
        print('{}:'.format(k))
        print(txt.format(r1, r2, np.sum((r2 - r1) / (r1))))
    
    r1, r2 = book_results['dPg'], -calc['result'][1]
    print('Gravity gradient:\n' + txt.format(r1, r2, np.sum((r2 - r1) / (r1))))
    
    r1, r2 = book_results['dPf'], -calc['result'][2]
    print('Friction gradient:\n' + txt.format(r1, r2, np.sum((r2 - r1) / (r1))))
        
#%%
def test2(D=50, eps=0.0018, alpha=90, P0=119,
          qg=9, rho_g=141.3, mu_g=0.02,
          ql=4.75, rho_l=613.8, mu_l=0.5,
          sigma=28):
    # https://www.checalc.com/fluid_flow_beggs_brill.html - no usa payne
    P0*= SPC.bar
    qg/=SPC.hour
    ql/=SPC.hour
    mu_g *= 1e-3
    mu_l *= 1e-3
    D *= 1e-3
    eps*=1e-3
    inc = np.sin(alpha * np.pi / 180)
    output = beggs_brill_correlation(ql * rho_l,
                                     qg * rho_g,
                                     rho_l, rho_g, mu_l, mu_g,
                                     D, inc, eps, sigma, payne_correction=False,
                                     full_output=True)
    return output
    
    
if __name__ == '__main__':
    test1()
    test2()