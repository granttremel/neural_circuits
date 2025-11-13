

from dataclasses import dataclass, field
from typing import Dict, List, Any, Union
from enum import Enum, StrEnum
import string

import numpy as np

class BaseUnit(StrEnum):
    MOL = "mol"
    MOLAR = "M" # molar
    J = "J"
    K = "K"
    C = "C"
    F = "F"
    V = "V"
    A = "A"
    OHM = "OHM"
    
    G = "g"
    M = "m"
    S = "s"
    L = "L"
    
    SIEMEN = "S"
    
@dataclass
class MagUnit:
    unit:BaseUnit
    prefix: str
    
    _prefixes={
        "f":1e-15,
        "p":1e-12,
        "n":1e-9,
        "u":1e-6,
        "m":1e-3,
        "c":1e-2,
        "d":1e-1,
        "":1,
        "k":1e3,
        "M":1e6,
        "G":1e9
    }
    
    def to_str(self):
        return self.prefix + self.unit.value
    
    def as_base(self):
        return MagUnit(self.unit, ""), self._prefixes.get(self.prefix, 1.0)
    
    def as_prefixed(self, prefix = None, prefixed_unit = ""):
        
        unit = self.unit.value
        if prefix is None:
            unit, prefix = self.separate_prefix(prefixed_unit)
        
        if prefix == "":
            return self.as_base()
        
        fact = self._prefixes.get(prefix, 1.0)/self._prefixes.get(self.prefix, 1.0)
        
        bu = BaseUnit(unit)
        return MagUnit(bu, prefix), fact
    
    @classmethod
    def from_str(cls, unit_str:str):
        unit, prefix = cls.separate_prefix(unit_str)
        return cls(BaseUnit(unit), prefix)
        
    @classmethod
    def separate_prefix(cls, unit_str:str):        
        
        for unit in BaseUnit._member_map_.values():
            unit_name = unit.value
            if unit_str.endswith(unit_name):
                prefix = unit_str.removesuffix(unit_name)
                # print(unit_str, unit, unit_name, prefix)
                return unit_name, prefix
        # print(unit_str, "flomp")
        return unit_str, ""
    
    @classmethod
    def is_valid(cls, unit_str:str):
        return cls.from_str(unit_str) is not None
    
    def __hash__(self):
        return hash((str(self.unit), self.prefix))
    
class Units:
    def __init__(self, value, **kwargs):
        self.value = value
        self.units:Dict[MagUnit, int] = {}
        
        for k in kwargs:
            if MagUnit.is_valid(k):
                mu = MagUnit.from_str(k)
                self.units[mu] = kwargs.get(k)

    def add_unit(self, unit, exp=1):
        if isinstance(unit, str):
            unit = BaseUnit(unit)
        if unit in self.units:
            exp = exp + self.units[unit]
        self.units[unit] = exp
        return self

    def add_num(self, unit, exp = 1):
        return self.add_unit(unit, exp=exp)

    def add_den(self, unit, exp = 1):
        return self.add_unit(unit, exp=-exp)
    
    def convert_value(self, factor):
        self.value *= factor
    
    def convert_to_other(self, other:'Units'):
        conv = 1.0
        new_units = Units(self.value)
        for mu, exp in self.units.items():
            for muu, expp in other.units.items():
                if mu.unit == muu.unit:
                    new_mu, fact = mu.as_prefixed(prefix = muu.prefix)
                    conv *= fact**exp
                    new_units.add_unit(new_mu, exp)
                    
        new_units.convert_value(conv)
        return new_units  
    
    def as_base(self):
        
        new_units = Units(self.value)
        conv = 1.0
        
        for mu, exp in self.units.items():
            
            new_m, fact = mu.as_base()
            conv *= fact**exp
            
            new_units.add_unit(new_m, exp)
        
        new_units.convert_value(conv)
        return new_units
    
    def convert(self, *unit_strs):
        
        conv = 1.0
        new_units = Units(self.value)
        mag_units = [MagUnit.from_str(ustr) for ustr in unit_strs]
        mag_unit_map = {mu.unit:mu for mu in mag_units}
        
        for mu, exp in self.units.items():
            
            if mu.unit in mag_unit_map:
                
                new_unit = mag_unit_map[mu.unit]
                new_m, fact = mu.as_prefixed(prefix = new_unit.prefix)
                
                conv *= fact**exp
                new_units.add_unit(new_m, exp)
            else:
                new_units.add_unit(new_m, exp)
        new_units.convert_value(1/conv)
        return new_units
    
    @classmethod
    def quick_convert(cls, value, from_units, *to_units, to_base = False):
        units = cls.from_unit_str(value, from_units)
        if to_base:
            conv_units = units.as_base()
        else:
            conv_units = units.convert(*to_units)
        return conv_units.value
        
    def to_inverse(self):
        inv_val = 1/self.value
        return Units(inv_val, **{k_unit.to_str():-v_exp for k_unit, v_exp in self.units.items()})
    
    def multiply(self, other):
        return Units._multiply(self, other)
    
    @classmethod
    def _multiply(cls, unit_a:'Units', unit_b:'Units'):
        
        a_base = unit_a.as_base()
        b_base = unit_b.as_base()
        
        newv = a_base.value * b_base.value
        new_units = cls(newv)
        
        new_exps = {u:0 for u in BaseUnit._member_map_.values()}
        
        for mu, a_exp in a_base.units.items():
            new_exps[mu.unit] += a_exp
        for mu, b_exp in b_base.units.items():
            new_exps[mu.unit] += b_exp
        
        for bu, exp in new_exps.items():
            if exp != 0:
                new_units.add_unit(MagUnit(bu, ""), exp)
        
        return new_units
    
    def divide(self, other):
        return Units._divide(self, other)
    
    @classmethod
    def _divide(cls, unit_a, unit_b):
        b_inv = unit_b.to_inverse()
        return cls._multiply(unit_a, b_inv)
    
    def unit_str(self):
        
        num_strs = []
        den_strs = []
        
        for u in BaseUnit._member_map_.values():
            for mu, exp in self.units.items():
                if mu.unit == u:
                    
                    if abs(exp) == 1:
                        unit_str = f"{mu.to_str()}"
                    else:
                        unit_str = f"{mu.to_str()}{abs(exp)}"
                        
                    if exp > 0:
                        num_strs.append(unit_str)
                    elif exp < 0:
                        den_strs.append(unit_str)
        if den_strs:
            return f"{" ".join(num_strs)} / {" ".join(den_strs)}"
        else:
            return " ".join(num_strs)
    
    def full_str(self, value_fmt = "0.3f"):
        
        value_str = format(self.value, value_fmt)
        unit_str = self.unit_str()
        
        return f"{value_str} {unit_str}"
    
    @classmethod
    def from_unit_str(cls, value, unit_str):
        
        if "/" in unit_str:
            num, den = unit_str.split('/')
            num_items = num.split(" ")
            den_items = den.split(" ")
        else:
            num_items = unit_str.split(" ")
            den_items = []
        
        units = cls(value)
        
        for i, items in enumerate([num_items, den_items]):
            for mu_str in items:
                if mu_str == "":
                    continue
                
                ns_unit = mu_str.rstrip(string.digits)
                if len(ns_unit) < len(mu_str):
                    exp = int(mu_str.removeprefix(ns_unit))
                else:
                    exp = 1
                if i==1:
                    exp *= -1
                
                mu = MagUnit.from_str(ns_unit)
                units.add_unit(mu, exp)
        
        return units
    
    
    @classmethod
    def Kb_Unit(cls):
        return cls().add_num("J", 1).add_den("K", 1)

    @classmethod
    def R_Unit(cls):
        return cls().add_num("J", 1).add_den("mol", 1).add_den("K", 1)
    
    @classmethod
    def F_Unit(cls):
        return cls().add_num("C", 1).add_den("mol", 1)
    
    @classmethod
    def e_Unit(cls):
        return cls().add_num("C", 1)
    
    @classmethod
    def D_Unit(cls):
        return cls().add_num("m", 2).add_den("s", 1)
    
    @classmethod
    def Lam_Unit(cls):
        return cls().add_num("Siemen",1).add_num("cm", 2).add_den("mol", 1)
    
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

class WaterData:
    
    MM = 18 # g/mol
    DENSITY = 0.99403 # g/cm3, 
    CONCENTRATION = 55.6 # moles/L
    DIELECTRIC = 73.9 # at 37C, unitless
    HEAT_CAPACITY = 75.26 # at 35-40C, J/mol K
    VISCOSITY = 0.7225 # 35C, mPa s
    DIFFUSION = 2.907e-9 # self-diffusion, m2 / s
    pKw= 14 # [H]*[OH]
    
    # standard molar entropy: 69.95 J/mol K
    # gibbs free energy: -237.24 kJ/mol
    # heat conductivity: 623.3 mW/mK
    
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
    m: float # molar mass # g/mol
    z: int # valence
    D:float # diffusivity # m2/s
    Lam: float # molar conductivity, S*cm2/mol
    pKa: float
    pKa2: float = None
    pKa3: float = None
    
    @property
    def Lam_eq(self):
        return self.Lam / self.z
    
    @classmethod
    def H(cls):
        return cls(Ion.H, 1, 1,9.31e-9, 349.6, 14.0)
    @classmethod
    def OH(cls):
        return cls(Ion.OH, 17, -1, 5.27e-9, 197.9, 0.0)
    @classmethod
    def Na(cls):
        return cls(Ion.Na, 23, 1, 1.33e-9, 50, 14.0)
    @classmethod
    def K(cls):
        return cls(Ion.K, 39.1, 1, 1.96e-9, 73.6, 14.0)
    @classmethod
    def Cl(cls):
        return cls(Ion.Cl, 35.4, -1, 2.03e-9, 76.2, 0.0)
    @classmethod
    def Ca(cls):
        return cls(Ion.Ca, 40.1, 2, 0.793e-9, 119.1, 14.0)
    @classmethod
    def Mg(cls):
        return cls(Ion.Mg, 24.3, 2, 0.705e-9, 105.1, 14.0)
    @classmethod
    def NH4(cls):
        return cls(Ion.NH4, 17.0, 1, 1.98e-9, 74.4, 14.0)
    @classmethod
    def HCO3(cls):
        return cls(Ion.HCO3, 61.0, -1, 1.18e-9, 44.3, 3.9, pKa2 = 3.2)
    
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
            pOH = WaterData.pKw - self.pH
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
            
            n = c*WaterData.CONCENTRATION * Consts.NA  # ct / cm3
            q = iodat.z * Consts.e # C
            
            qnsum += n*(q)**2
        
        num = Consts.eps * Consts.kB * self.T
        r = num / qnsum
        dl = np.sqrt(r)
        return dl

@dataclass
class IonChannel:
    ion:Ion
    
    p_active: float # m/s ?
    p_inactive: float # m/s ?
    t_activate: float
    t_inactivate: float
    
    v_thresh:float = None
    c_thresh: float = None
    
    surface_density:float = 1.0 # relative i guess

    def is_active(self, Vm, c_ion):
        res = False
        if self.v_thresh and Vm > self.v_thresh:
            res = True
        if self.c_thresh and c_ion > self.c_thresh:
            res = True
        return res

    @property
    def p(self, Vm, c_ion):
        if self.is_active(Vm, c_ion):
            return self.p_active
        else:
            return self.p_inactive

    def get_p(self, active = True):
        if active:
            return self.p_active
        else:
            return self.p_inactive

class MembraneConsts:
    
    t:float = 8e-9 #nm
    t_range = [6e-9, 10e-9]
    t_samples = [6e-9, (6e-9,8e-9),(8e-9,10e-9)]
    
    Tinv = 1e-9 # reciprocal debye length, m = 1 nm
    
    Rl = 1.0
    
    Rm = 1.0
    
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
class MembraneData:
    intra: Electrolyte
    extra: Electrolyte
    channels: Dict[Ion, List[IonChannel]] = field(default_factory = dict)
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
    
    def add_channel(self, channel:IonChannel):
        if not channel.ion in self.channels:
            self.channels[channel.ion] = []
            
        self.channels[channel.ion].append(channel)
    
    def set_channels(self, channels: List[IonChannel]):
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
        
        Tinv_fact = Consts.eps * Consts.R * self.T * WaterData.DIELECTRIC / (2 * Consts.F * Consts.F)
        
        c_cat_in = self.intra.get_conc(Ion.Na) + self.intra.get_conc(Ion.K)
        c_cat_out = self.extra.get_conc(Ion.Na) + self.extra.get_conc(Ion.K)
        
        Tinv_in = np.sqrt(Tinv_fact / c_cat_in)
        Tinv_out = np.sqrt(Tinv_fact / c_cat_out)
        return Tinv_in, Tinv_out
    
    def electrodiffusion_ratio(self, ion, Vm = -0.07):
        """
        J = -u c dmu/dx = -u R T dc/dx - u c z F dV/dx
                            diffusion   electromigration
        
        
        """
        
        
        c_in = self.intra.get_conc(ion)
        c_out = self.extra.get_conc(ion)
        # c_in_tot = self.intra.get_conc(Ion.Na) + self.intra.get_conc(Ion.K)
        # c_out_tot = self.extra.get_conc(Ion.Na) + self.extra.get_conc(Ion.K)
        dc = c_in - c_out
        
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
        self.locations = {loc:[MembraneData.rest()] for loc in locations}
        self.T: float = kwargs.get("T", 310.0)
    
    
