#!/usr/bin/env python
# coding: utf-8

from scipy.special.cython_special import *
import numpy as np

from astropy.constants import alpha, c, e, hbar, m_e, m_p, sigma_sb, k_B, M_sun, sigma_T, e, G
from astropy import units as u

import argparse

#-------------------------------------------------------------------------------------
# First define some useful constants

mec2 = (m_e * c ** 2).cgs


def free_fall_time(R, M):
    # Calculate the free-fall time  ( FROM CHATGPT; CHECK!!!!)
    t_ff = np.sqrt((3 * np.pi / 32) * (R**3 / (G * M))).to(u.s)

    return t_ff


def calc_gmax_cool(eta_g, r_c, B):
    '''Calculate the maximum Lorentz factor of the electrons given by t_acc = t_syn.
    The formulae are taken from Inoue+2019 (https://iopscience.iop.org/article/10.3847/1538-4357/ab2715/pdf)
    eta_g: gyrofactor (mean free path in units of R_g)
    r_c: coronal radius in units of R_s
    B: magnetic field intensity'''
    #t_sy = 7.7e4 * (B / 10 G)**-2 * (gamma/100)**-1 s
    #t_acc = 7.6e-3 * (eta_g / 100) * (r_c/40) * (B / 10 G)**-1 * (gamma/100)**-1 s ! * (eta_acc/10)
    factor = (7.7e4/7.6e-3) * (100/eta_g) * (40/r_c) * (10 * u.G/B) 
    gmax_cool = 100 * np.sqrt(factor)
    return gmax_cool


def calc_gmax_conf(R, B):
    '''Calculate the maximum Lorentz factor of the electrons given by confinement (Hillas criterium).
    The formulae are taken from Inoue+2019 (https://iopscience.iop.org/article/10.3847/1538-4357/ab2715/pdf)
    R: accelerator size
    B: magnetic field intensity'''
    # Make sure to do the maths in cgs-gauss units
    gmax_conf = e.gauss.value * B.to("G").value * R.to("cm").value / mec2.value 
    return gmax_conf
