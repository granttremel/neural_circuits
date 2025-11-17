
from dataclasses import dataclass, field
from typing import Dict, List, Any, Union
from enum import Enum, StrEnum
import string

class BaseUnit(StrEnum):
    CT = "Ct" # count or number
    MOL = "mol"
    MOLAR = "M" # molar
    J = "J"
    K = "K"
    C = "C"
    F = "F" # farad
    V = "V" # volt
    A = "A" # ampere, C/s
    OHM = "Ohm"
    
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
    
class Units(float):
    def __init__(self, value, **kwargs):
        super().__init__()
        
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

    def has_unit(self, base_unit):
        
        for mu in self.units:
            if mu.unit == base_unit:
                return True
        return False

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
    
    def with_value(self, value):
        unit_kwargs = {k.to_str():v for k,v in self.units.items()}
        return Units(value, **unit_kwargs)
    
    def __call__(self, value):
        return self.with_value(float(value))
    
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
        new_units.convert_value(conv)
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
    def kB_Unit(cls):
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
