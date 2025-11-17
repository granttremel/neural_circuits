
from dataclasses import dataclass, field
from typing import Dict, List, Any, Union
from enum import Enum, StrEnum
import string

import numpy as np

class Consts:
    kB = 1.380e-23 # J/K
    R = 8.341 # J/K mol
    F = 9.648e4 # C/mol 
    e =  1.602e-19 # C
    eps = 8.854e-12 # C2/kg m3 s2 = F/m
    
    NA = 6.0223e23 # ct / mol

    @classmethod
    def Ex(cls, T = 310):
        """
        Base voltage per valence for charges separated in an electrolytic cell. J / C
        """
        return cls.R * T / cls.F
    
    @classmethod
    def Ex_b10(cls, T=310):
        return np.log(10)*cls.R * T / cls.F

class WaterConsts:
    
    MOLAR_MASS = 18 # g/mol
    DENSITY = 0.99403 # g/cm3, 
    CONCENTRATION = 55.34 # moles/L
    DIPOLE = 1.84 # debye
    DIELECTRIC = 73.9 # at 37C, unitless
    HEAT_CAPACITY = 75.26 # at 35-40C, J/mol K
    VISCOSITY = 0.7225 # 35C, mPa s
    DIFFUSION = 2.907e-9 # self-diffusion, m2 / s
    pKw= 14 # [H]*[OH]
    """
    standard molar entropy: 69.95 J/mol K
    gibbs free energy: -237.24 kJ/mol
    heat conductivity: 623.3 mW/mK
    
    Mille 3e
    Dielectric relaxation times: 0.18 - 9.5 ps
    OH bond length: 0.957 Angstrom
    Avg nearest neighbor (O-O dist): 2.85 Angstrom
    Volume per molecule: 3. Angstrom3
    """

class Conversions:
    
    def __init__(self):
        pass
    
    def _permeability_from_diffusivity(self, D, beta, t):
        """
        Parameters
        D: diffusivity of species (fick's law)
        beta: partition coefficient of species just inside aqueous layer to just inside membrane layer
        t: thickness of membrane
        
        Returns
        p: permeability, mol s / kg
        """
        
        return D * beta / t
    
    def permeability_from_diffusivity(self, D, t):
        """
        Parameters
        D: diffusivity of species (fick's law)
        t: thickness of membrane
        
        Returns
        p: permeability, mol s / kg
        """
        beta = 1.0
        return self._permeability_from_diffusivity(D, beta, t)
    
    def mobility_from_diffusivity(self, D, T):
        """
        Parameters
        D: diffusivity, m2 / s
        T: temperature, K
        
        Returns
        u: mobility, mol s / kg
        mobility u formally relates the drift velocity of a charged species in an electrochemical potential gradient, nu = u * dmu/dx
        """
        return D / Consts.R / T

    
class Physics:
    
    def __init__(self):
        pass
    
    def _nernst(self, E_std, Q_1, Q_2, z, T):
        """
        Nernst equation
        E = E_std - RT/zF log(Q_2/Q_1)
        E is electrochemical potential for equilibrium defined by Q_1 and Q_2
        E_std is standard electrochemical potential
        Q_1 is denominator of reaction quotient, i.e. reactants or phase A, activity of phase or component from which Q_2 is referred
        Q_2 is numerator of reaction quotient, i.e. products or phase B, activity of phase or component to which Q_1 is referred
        z is valence of species
        
        """
        fact = Consts.R * T / z / Consts.F
        return E_std - fact * np.log(Q_2 / Q_1)
    
    def nernst(self, Q_1, Q_2, z, E_std = 0, T=310):
        return self._nernst(E_std, Q_1, Q_2, z, T=T)
    
    def _selectivity(self, P_A, conc_A_out, P_B, conc_B_in, z, T):
        """
        Reversal potential for a biionic potential, where the outside is purely A and inside purely B. i.e., at what potential the current changes sign, 
        
        """
        E_std = 0
        Q_1 = P_B * conc_B_in
        Q_2 = P_A * conc_A_out
        return self._nernst(E_std, Q_1, Q_2, z, T)
    
    def selectivity(self, P_A, conc_A_out, P_B, conc_B_in, z, T=310):
        return self._selectivity(P_A, conc_A_out, P_B, conc_B_in, z, T)
    
    def _boltzmann_OC(self, z_g, V_m, V_half, T):
        """
        Boltzmann statistics for ratio of open to closed channels with gating charge under potential difference
        r_OC = O/C = exp(-(w - z_g q_e E)/kB T)
        z_g is valence of gating charge
        V is potential difference across membrane
        V_half is the voltage at which half of gates should be open
        """
        return np.exp(-(z_g * Consts.e * (V_m - V_half)) / Consts.kB / T)
    
    def _boltzmann_OOC(self, z_g, V, V_half, T):
        """
        Boltzmann statistics for ratio of open to closed channels with gating charge under potential difference
        r_OOC = O/(O+C) =  1 / (1 + exp(-(w - z_g q_e E)/kB T))
        z_g is valence of gating charge
        V is potential difference across membrane
        V_half is the voltage at which half of gates should be open
        """
        return 1/(1+self._boltzmann_OC(z_g, V, V_half, T))
    
    def boltzmann(self, z_g, V, V_half, T = 310):
        return self._boltzmann_OOC(z_g, V, V_half, T)
    
    def _electrochemical_potential(self, mu_std, activity_species, z_species, V, T):
        """
        mu = mu_std + RT log(activity) + zF V
        
        Parameters:
        mu_std: standard electrochemical potential
        activity_species: activity of the ionic species (might just be concentration)
        z_species: valence of ionic species
        V: electrical potential difference
        
        Returns
        mu: electrochemical potential of the ionic species, J/mol
        """
        
        return mu_std + -Consts.R * T * np.log(activity_species) + z_species * Consts.F * V 
    
    def electrochemical_potential_difference(self, conc_A, conc_B, z_ion, V_A, V_B, T = 310):
        
        mu_std = 0
        act = conc_B / conc_A
        dV = V_B - V_A
        return self._electrochemical_potential(mu_std, act, z_ion, dV, T)
    
    def _GHK(self, conc_ions_A, conc_ions_B, p_ions_AB, z_ions, T):
        """
        Goldman Hodgkin Katz voltage equation
        
        Em = R T / F * log[(sum p_cat c2_cat + sum p_an c1_an) / (sum p_cat c2_cat + sum p_an c1_an)]
        
        Em: potential difference due to electrochemical potential gradient from region 2 to region 1
        
        conc_ions_A: concentration of ions in region 1
        conc_ions_B: concentration of ions in region 2
        p_ions_AB: permeability of ions between regions
        z_ions: valence of ions (i.e. cation vs anion)
        """
        
        num_ions = len(conc_ions_A)
        
        num = 0
        den = 0
        
        for i in range(num_ions):
            ion_p = p_ions_AB[i]
            ion_c_in = conc_ions_B[i]
            ion_c_out = conc_ions_A[i]
            ion_z = z_ions[i]
            
            if ion_z < 0:
                num += ion_p*ion_c_in
                den += ion_p*ion_c_out
            elif ion_z > 0:
                num += ion_p*ion_c_out
                den += ion_p*ion_c_in
        
        r = num / den
        
        logr = np.log(r)
        f = Consts.Ex(T)
        
        return f * logr
    
    def GHK(self, conc_ions_A, conc_ions_B, p_ions_AB, z_ions, T=310):
        return self._GHK(conc_ions_A, conc_ions_B, p_ions_AB, z_ions, T)
    
    def _nernst_planck(self, V_A, V_B, conc_A, conc_B, conc_local, dx, u, z, T):
        """
        J = -u c dmu/dx = -u R T dc/dx - u c z F dV/dx
                            diffusion   electromigration
        J: molar flux of species due to diffusion and electromigration
        c: concentration of species, f'n of space (and sometimes time)
        V: electric potential, f'n of space (time)
        u: mobility
        
        """
    
        e_cont = -Consts.F * z * conc_local * (V_B - V_A) / dx
        d_cont = -u * Consts.R * T * (conc_B - conc_A) / dx
    
        return d_cont, e_cont
    
    def nernst_planck(self, V_A, V_B, conc_A, conc_B, dx, u, z, T = 310):
        
        conc_local = (conc_A+conc_B)/2
        d_cont, e_cont = self._nernst_planck(V_A, V_B, conc_A, conc_B, conc_local, dx, u, z, T)
        return d_cont + e_cont
    
    def electrodiffusion_ratio(self, V_A, V_B, conc_A, conc_B, dx, z, T = 310):
        
        conc_local = (conc_A+conc_B)/2
        u = 1.0
        
        d_cont, e_cont = self._nernst_planck(V_A, V_B, conc_A, conc_B, conc_local, dx, u, z, T)
        return e_cont / d_cont
    
    def _reciprocal_debye_length(self, conc_ions, T):
        """
        T^-1 = sqrt( eps eps_w R T / c F2 )
        
        Parameters
        conc_ions: the concentration of all ions in solution; either cations or anions, not both. in M
        
        Returns
        Tau^-1, the reciprocal debye length, m
        """
        
        Tinv_fact = Consts.eps * Consts.R * T *  WaterConsts.DIELECTRIC / (2 * Consts.F * Consts.F)
        
        conc_cations = sum(conc_ions)
        
        Tinv = np.sqrt(Tinv_fact / conc_cations)
        
        return Tinv

    def reciprocal_debye_length(self, conc_ions, T = 310):
        return self._reciprocal_debye_length(conc_ions, T)
    
    def _probability_channel_open(self, gating_charge, dV, dV_half, T):
        """
        
        """
        
        expand = -gating_charge * (dV - dV_half) / Consts.R / T
        pinv = 1 + np.exp(expand)
        return 1/pinv
    