

from dataclasses import dataclass, field
from typing import Dict, List, Any, Union, TYPE_CHECKING
from enum import Enum, StrEnum
import string

import numpy as np

from neural_circuits.modeling.physics import Consts,  WaterConsts, Conversions, Physics
from neural_circuits.modeling.units import BaseUnit, MagUnit, Units
if TYPE_CHECKING:
    from neural_circuits.modeling.channel import IonChannel
    
class Ion(Enum):
    """
    should just be species?
    """
    
    H = 1
    OH = -1
    
    Na = 2
    K = 3
    Ca = 4
    Mg = 5
    NH4 = 6
    
    Cl = -2
    HCO3 = -3
    
    @classmethod
    def all_ions(cls):
        return list(cls._member_map_.values())
    
    @classmethod 
    def to_dict(cls):
        return cls._member_map_.copy()
    
    @property
    def is_anion(self):
        return self.value < 0
    
    @property
    def is_cation(self):
        return self.value > 0
    
    @property
    def is_neutral(self):
        return self.value == 0
    
    @classmethod
    def from_name(cls, ion_name):
        for n, ion in cls._member_map_.items():
            if n == ion_name:
                return ion
    
@dataclass
class IonData:
    ion: Ion
    m: float # molar mass, g/mol
    z: int # valence
    r: float # ionic radius, anstroms
    r_w: float # radius of hydrated ion, angstroms
    E_h: float # enthalpy of hydration, kJ/mol.. or of solution??
    
    D:float # diffusivity # m2/s
    Lam: float # molar conductivity or limiting equivalent conductivity, S*cm2/mol
    pKa: float
    pKa2: float = None
    pKa3: float = None
    
    @property
    def Lam_eq(self):
        return self.Lam / self.z
    
    def u(self, T = 310):
        """
        mobility, v = u dmu/dx (mu being electrochemical potential)
        """
        return self.D / Consts.R / T
    
    @classmethod
    def H(cls):
        return cls(Ion.H, 1, 1, 0, 2.8, -1130, 9.31e-9, 349.6, 14.0)
    @classmethod
    def OH(cls):
        return cls(Ion.OH, 17, -1, 1.76, 3.0, None, 5.27e-9, 197.9, 0.0)
    @classmethod
    def Na(cls):
        return cls(Ion.Na, 23, 1, 1.16, 3.6, -406, 1.33e-9, 50, 14.0)
    @classmethod
    def K(cls):
        return cls(Ion.K, 39.1, 1, 1.52, 3.3, -322, 1.96e-9, 73.6, 14.0)
    @classmethod
    def Cl(cls):
        return cls(Ion.Cl, 35.4, -1, 1.67, 3.3, -363, 2.03e-9, 76.2, 0.0)
    @classmethod
    def Ca(cls):
        return cls(Ion.Ca, 40.1, 2, 1.14, 4.1, -1577, 0.793e-9, 119.1, 14.0)
    @classmethod
    def Mg(cls):
        return cls(Ion.Mg, 24.3, 2, 0.86, 4.3, -1921, 0.705e-9, 105.1, 14.0)
    @classmethod
    def NH4(cls):
        return cls(Ion.NH4, 17.0, 1, None, None, -307, 1.98e-9, 74.4, 14.0)
    @classmethod
    def HCO3(cls):
        return cls(Ion.HCO3, 61.0, -1, None, None, None, 1.18e-9, 44.3, 3.9, pKa2 = 3.2)
    
    @classmethod
    def from_ion(cls, ion:Ion):
        return cls.from_name(ion.name)

    @classmethod
    def from_name(cls, name:str):
        if name == "H":
            return cls.H()
        elif name == "OH":
            return cls.OH()
        elif name == "K":
            return cls.K()
        elif name == "Na":
            return cls.Na()
        elif name == "Cl":
            return cls.Cl()
        elif name == "Ca":
            return cls.Ca()
        elif name == "Mg":
            return cls.Mg()
        elif name == "NH4":
            return cls.NH4()
        elif name == "HCO3":
            return cls.HCO3()

class Electrolyte:
    
    def __init__(self, **kwargs):
        self._concs:Dict[Ion, float] = {}
        
        ion_dict = Ion.to_dict()
        for ion_name, ion in ion_dict.items():
            self._concs[ion] = kwargs.get("c_" + ion_name, 0.0)
            
        self.pH = kwargs.get("pH", 7.4)
        self.T:float = kwargs.get("T", 310) # K
        
        self.calculate_HOH()
    
    def get_conc(self, ion):
        if isinstance(ion, str):
            ion = Ion.from_name(ion)
        return self._concs.get(ion, 0.0)
    
    def calculate_HOH(self):
        
        if self.pH < 7.0:
            c_H = np.exp(-np.log(10)*self.pH)
            self._concs[Ion.H] = c_H
        else:
            pOH =  WaterConsts.pKw - self.pH
            c_OH = np.exp(-np.log(10)*pOH)
            
            self._concs[Ion.OH] = c_OH
            
    @classmethod
    def intra_rest(cls, **kwargs):
        return cls(c_Na = 0.01, c_K = 0.14, c_Cl = 0.015, c_Ca = 1e-4, **kwargs)
    
    @classmethod
    def extra_rest(cls, **kwargs):
        return cls(c_Na = 0.145, c_K = 0.005, c_Cl = 0.110, c_Ca = 0.001, **kwargs)

    def update(self, T_new):
        self.T = T_new

    @property
    def debye_length(self):
        return 1.e-9 # m = 1nm
    
    
    def calculate_debye_length(self):
        """
        ~0.4-0.6 nm for physiological, close but not quite
        """
        
        qnsum = 0
        
        for sp, c in self._concs.items():
            
            iodat = IonData.from_ion(sp)
            
            n = c* WaterConsts.CONCENTRATION * Consts.NA  # ct / cm3
            q = iodat.z * Consts.e # C
            
            qnsum += n*(q)**2
        
        num = Consts.eps * Consts.kB * self.T
        r = num / qnsum
        dl = np.sqrt(r)
        return dl

class MembraneConsts:
    
    t:float = 8e-9
    """
    Membrane thickness, nm
    """
    t_range = [6e-9, 10e-9]
    t_samples = [6e-9, (6e-9,8e-9),(8e-9,10e-9)]
    
    Tinv = 1e-9 # reciprocal debye length, m = 1 nm
    
    d_axon = 5
    """
    diameter of the axon
    """
    d_axon_range = [1, 20]
    
    l_node = 2
    """
    length of node of ranvier, um
    """
    
    l_internode = 600
    """
    length of myelinated internodes, um
    """
    l_internode_range = [300, 2000]
    
    Rl = 1.0 # should be equal to salty walder
    
    Rm = 88.3 # MOhm
    
    Cm = 0.90
    Cm_range = [0.85, 1.0] # uF / cm2
    
    Cm_ests = [
        # all uF / cm2 
        0.7, # asolectin lipid bilayers
        0.94, # egg lecithin lipid bilayers
        1.0, # squid axon min
        1.3, # squid axon max
        0.75, # hippocampal pyramidal, measurements fit to model
        0.9, # hippocampal interneurons, measurements fit to model
        2.4, # spinal cord ventral horn, measurements fit to model
        # Gentet, 2000, Biophysics
        0.92, # cortical pyramidal
        0.85, # spinal cord
        0.92, # cultured hippocampal
        1.06, # cultural glial
    ]
    
    # eps_mem = 8.132, similar to natural rubber..
    
    qS = -0.15 # surface charge density, C/m2
    qS_range = [-0.002, -0.3]


@dataclass
class Membrane:
    intra: Electrolyte
    extra: Electrolyte
    channels: Dict[Ion, List['IonChannel']] = field(default_factory = dict)
    myelinated: bool = False

    T: float = 310 # K
    t: float = MembraneConsts.t # thickness: m = 4nm
    Tinv: float = MembraneConsts.Tinv
    d: float = 1e-6 # diameter: m = 1um
    
    Vm_rest:float = -0.070 # idk its just a constant
    
    @classmethod
    def rest(cls, channels = []):
        rest_mem = cls(Electrolyte.intra_rest(), Electrolyte.extra_rest())
        for chan in channels:
            rest_mem.add_channel(chan)
        return rest_mem
        
    @property
    def p(self, ion, Vm, c_ion):
        p = 0.0
        for i, chan in self.channels.items():
            if i == ion:
                p += chan.p(Vm, c_ion)
        
        if p == 0.0:
            return 1.0
        else:
            return p
    
    def get_p(self, ion, active = True):
        all_ps = [chan.get_p(active = active) for chan in self.channels.get(ion,[])]# is it a sum or other ? like resistances in parallel idk
        if not all_ps:
            return 1.0
        else:
            return sum(all_ps)
    
    def add_channel(self, channel:'IonChannel'):
        if not channel.ion in self.channels:
            self.channels[channel.ion] = []
            
        self.channels[channel.ion].append(channel)
    
    def set_channels(self, channels: List['IonChannel']):
        self.channels = {}
        for c in channels:
            self.add_channel(c)
    
    # capacitance, resistance (1/p?)
    
    def Em(self, *species:List[Ion]):
        
        if not species:
            species = Ion.all_ions()
        
        num = 0
        den = 0
        
        for sp in species:
            ion_p = self.get_p(sp, active = True)
            ion_c_in = self.intra.get_conc(sp)
            ion_c_out = self.extra.get_conc(sp)
            
            if sp.is_anion:
                num += ion_p*ion_c_in
                den += ion_p*ion_c_out
            elif sp.is_cation:
                num += ion_p*ion_c_out
                den += ion_p*ion_c_in
        
        r = num / den
        
        logr = np.log(r)
        f = Consts.Ex(self.T)
        
        return f * logr
        
    def reciprocal_debye_length(self, ion = None):
        """
        T^-1 = sqrt( eps eps_w R T / F2 )
        """
        
        Tinv_fact = Consts.eps * Consts.R * self.T *  WaterConsts.DIELECTRIC / (2 * Consts.F * Consts.F)
        
        c_cat_in = self.intra.get_conc(Ion.Na) + self.intra.get_conc(Ion.K)
        c_cat_out = self.extra.get_conc(Ion.Na) + self.extra.get_conc(Ion.K)
        
        Tinv_in = np.sqrt(Tinv_fact / c_cat_in)
        Tinv_out = np.sqrt(Tinv_fact / c_cat_out)
        return Tinv_in, Tinv_out
    
    @classmethod
    def longitudinal_electrodiffusion_ratio(cls, mem1, mem2, ion, dist):
        
        c_in_m1 = mem1.intra.get_conc(ion)
        c_out_m1 = mem1.extra.get_conc(ion)
        c_in_m2 = mem2.intra.get_conc(ion)
        c_out_m2 = mem1.extra.get_conc(ion)
        
        c_in = (c_in_m1 + c_in_m2)/2
        c_out = (c_out_m1 + c_out_m2)/2
        
        dc_in = c_in_m2 - c_in_m1
        dc_out = c_out_m2 - c_out_m1
        
        e_fact = Consts.F * IonData.from_ion(ion).z
        d_fact = Consts.R * mem1.T
        
        dv = mem2.Em() - mem1.Em()
        
        e_cont_in = e_fact * c_in * dv / dist
        
        pass
    
    def electrodiffusion_ratio(self, ion, Vm = None):
        """
        J = -u c dmu/dx = -u R T dc/dx - u c z F dV/dx
                            diffusion   electromigration
        J: molar flux of species due to diffusion and electromigration
        c: concentration of species, f'n of space (and sometimes time)
        V: electric potential, f'n of space (time)
        u: mobility
        
        """
        
        c_in = self.intra.get_conc(ion)
        c_out = self.extra.get_conc(ion)
        # c_in_tot = self.intra.get_conc(Ion.Na) + self.intra.get_conc(Ion.K)
        # c_out_tot = self.extra.get_conc(Ion.Na) + self.extra.get_conc(Ion.K)
        dc = c_in - c_out
        
        if Vm is None:
            Vm = self.Em()
        dv = Vm
        
        e_fact = Consts.F * IonData.from_ion(ion).z
        d_fact = Consts.R * self.T
        
        e_cont_in = e_fact * (c_in) * dv
        e_cont_out = e_fact * (c_out) * dv
        d_cont = d_fact * dc
        
        Jh_in = e_cont_in + d_cont
        Jh_out = e_cont_out + d_cont
        
        # r_in = e_cont_in / (Jh_in)
        # r_out = e_cont_out / (Jh_out)
        r_in = e_cont_in / d_cont
        r_out = e_cont_out / d_cont
        
        return r_in, r_out


class NeuronLocation(Enum):
    DENDRITIC_SPIKE = -2
    DENDRITE = -1
    SOMA = 0
    IAS = 1
    INTERNODE = 2
    NODE = 3
    SYNAPSE = 4


class NeuronData:
    
    def __init__(self, locations, **kwargs):
        
        self.times = [0]
        self.locations = {loc:[Membrane.rest()] for loc in locations}
        self.T: float = kwargs.get("T", 310.0)
    
    
"""
Just some useful tidbits

most biological ion channels of conductance 1-150 pS

the quantity of potassium moved across the membrane to generate 100mV is small:

let p_open = 0.5, g_ch = 20e-12 S/um2
g_m = p_open * g_ch = 1 mS/cm2
R_m = 1/g_m = 1000 Ohm/cm2
C_m = 1 uF/cm2
(so t_m = R_m * C_m 1 ms)
if E = 100mV:
Q = E*C_m = 10e-7 C/cm2
Q/F = 10e-12 mol K / cm2 membrane. per CENTIMETER SQUARED!!













"""