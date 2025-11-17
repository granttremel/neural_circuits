
from typing import Dict, List, Tuple, Union, Any
from tabulate import tabulate

import numpy as np

from neural_circuits.modeling.units import BaseUnit, MagUnit, Units

class EmpiricalData:

    _fields_idp = []
    _types_idp = []
    _fields_dp = []
    _types_dp = []
    _fields_meta = []
    _types_meta = []

    def __init__(self, **kwargs):
        
        self.idp:Dict[str, Any] = {k:kwargs.get(k, -1) for k in kwargs if k in self._fields_idp}
        self.dp:Dict[str, Any] = {k:kwargs.get(k) for k in kwargs if k in self._fields_dp}
        self.meta:Dict[str, Any] = {k:kwargs.get(k, "") for k in self._fields_meta}
    
    def __getattr__(self, att):
        if att in self._fields_dp:
            return self.dp.get(att)
        elif att in self._fields_idp:
            return self.idp.get(att)
        elif att in self._fields_meta:
            return self.meta.get(att)
    
    @property
    def fields(self):
        return self.get_fields()
    
    @classmethod
    def get_fields(cls):
        return cls._fields_idp + cls._fields_dp + cls._fields_meta
    
    @property
    def types(self):
        return self._types_idp + self._types_dp + self._types_meta
    
    @classmethod
    def get_type(cls, field, as_unit = False):
        
        tp = None
        if field in cls._fields_idp:
            tp = cls._types_idp[cls._fields_idp.index(field)]
        elif field in cls._fields_dp:
            tp = cls._types_dp[cls._fields_dp.index(field)]
        elif field in cls._fields_meta:
            tp = cls._types_meta[cls._fields_meta.index(field)]
        
        if isinstance(tp, str):
            if as_unit:
                return Units.from_unit_str(0.0,tp)
            else:
                return float
        else:
            return tp
    
    @classmethod
    def get_unit(cls, field):
        unit = cls.get_type(field, as_unit = True)
        if not isinstance(unit, Units):
            return None
        else:
            return unit
    
    def format_value(self, field, with_unit = False, float_fmt = "0.1f", sci_fmt = "0.1e"):
        
        v = getattr(self, field)
        tp = self.get_type(field, as_unit = True)
        
        if v is None:
            vstr = "None"
        elif isinstance(tp, Units):
            
            if (0 < abs(v) and abs(v) < 1e-4) or abs(v) > 1e4:
                fstr = sci_fmt
            else:
                fstr = float_fmt
            
            if with_unit:
                vstr = tp.with_value(v).full_str(value_fmt = fstr)
            else:
                vstr = format(v, fstr)
        else:
            vstr = str(v)
            
        return vstr
    
    @classmethod
    def get_header(cls):
        
        fields = cls.get_fields()
        hdr = []
        
        for f in fields:
            
            ftp = cls.get_type(f, as_unit = True)
            
            if isinstance(ftp, Units):
                unitstr = ftp.unit_str()
                hdr.append(f"{f} ({unitstr})")
            else:
                hdr.append(f)
        return hdr
    
    def __repr__(self):
        
        all_fields = self._fields_idp + self._fields_dp + self._fields_meta
        
        parts = []
        
        for f in all_fields:
            vstr = self.format_value(f, with_unit = True)
            if vstr == '':
                vstr = 'None'
            parts.append(f"{f}={vstr}")
            
        return f"{self.__class__.__name__}({",".join(parts)})"
    
class EmpiricalDataset:
    
    _data_type = EmpiricalData
    
    def __init__(self, source, **kwargs):
        
        self.source = source
        
        self.samples = []

    @property
    def fields(self):
        return self._data_type.get_fields()
    
    @property
    def types(self):
        return self._data_type.types
    
    @classmethod
    def get_type(cls, field, as_unit = False):
        return cls._data_type.get_type(field, as_unit=as_unit)
    
    def add_sample(self, sample:EmpiricalData):
        self.samples.append(sample)
    
    def interpolate(self, tgt):
        
        pass
    
    def get_stats(self, **conds):
        
        n = 0
        stat_fields = self._data_type._fields_idp + self._data_type._fields_dp
        data = {f:[] for f in stat_fields if self.get_type(f) != str}
        
        stat_samps = []
        
        for samp in self.samples:
            
            skip = False
            for f, (_min, _max) in conds.items():
                vtest = getattr(samp, f)
                if vtest and vtest < _min or vtest > _max:
                    skip = True
                
            if skip:
                continue
            
            n += 1
            stat_samps.append(samp)
            
            for f in data:
                v = getattr(samp, f)
                if v is not None:
                    data[f].append(v)
        
        means = {f:np.mean(fdat) for f, fdat in data.items()}
        sds = {f:np.std(fdat) for f, fdat in data.items()}
        mins = {f:min(fdat) for f, fdat in data.items()}
        maxs = {f:max(fdat) for f, fdat in data.items()}
        
        mean_obj = self._data_type(**means)
        sds_obj = self._data_type(**sds)
        mins_obj = self._data_type(**mins)
        maxs_obj = self._data_type(**maxs)
        
        return mean_obj, sds_obj, mins_obj, maxs_obj, n
        
    
    def __getitem__(self, i):
        if i < len(self):
            return self.samples[i]
    
    def __len__(self):
        return len(self.samples)
    
    def print_table(self):
        
        hdr = self._data_type.get_header()
        
        rows = []
        
        for samp in self.samples:
            row = []
            fields = self.fields
            for f in fields:
                vstr = samp.format_value(f)
                row.append(vstr)
            rows.append(row)
        
        print(tabulate(rows, headers = hdr))
        
    @classmethod
    def from_table(cls, src, table, **kwargs):
        pass
    
    @classmethod
    def from_raw(cls, src, raw_str, **kwargs):
        
        ds = cls(src)
        delim = kwargs.pop("delimit", " ")
        
        lines = raw_str.strip().split("\n")
        
        hdr = lines[0].split(delim)
        
        for line in lines[1:]:
            row = line.split(delim)
            row = cls.clean_row(row, hdr)
            row.update(kwargs)
            
            new_samp = cls._data_type(**row)
            ds.add_sample(new_samp)
        
        return ds
            
    @classmethod
    def clean_row(cls, row, hdrs):
        
        out_row = {}
        
        for hdr, item in zip(hdrs,row):
            
            item_type = cls._data_type.get_type(hdr, as_unit = True)
            
            try:
                val = item_type(item)
            except:
                val = None
            
            out_row[hdr] = val
        
        return out_row
            

class MembraneData(EmpiricalData):
    _fields_idp = [
        "sp", # species
        "diam" # axon diameter
    ]
    _types_idp = [str, "um"]
    _fields_dp = [
        "R_BC", # "seal resistance"
        "R_CD", # internodal resistance
        "R_ED", # internodal resistance
        "R_M", # resting membrane resistance
        "C", # initial, apparent nodal capacitance
        "E_Na", # absolute sodium current reversal potential
        "I_p", # peak sodium current during depol to -5mV
        "gamma", # single channel conductance near 0mV
        "N" # number sodium channels
    ]
    _types_dp = ["MOhm", "MOhm", "MOhm", "MOhm", "pF", "mV", "nA", "pS", int]
    _fields_meta = [
        "Node" # index of node
    ]
    _types_meta = [int]
    
class MembraneDataset(EmpiricalDataset):
    
    _data_type = MembraneData

    @classmethod
    def sigworth_1980(cls):
        
        src = "Sigworth 1980, J. Physiol."
        
        ds = cls.from_raw(src, sigworth_raw, sp = "Rana")
        
        return ds

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



sigworth_raw = """
Node R_BC R_CD R_ED R_M C diam E_Na I_p gamma N
12 24 52 32 100 - - 46 6.3 6.4 30000
13 19 73 47 49 - 12 45 6.6 7.6 40000
14 20 49 31 96 1.9 14 65 4.8 5.6 -
19 20 54 38 110 2.7 - 54 7.7 5.8 43000
21 11 54 20 35 3.0 - 55 8.0 5.8 -
22 18 90 60 90 3.0 13 63 8.8 8.0 31000
24 20 70 25 81 4.0 14 45 5.1 6.1 24000
29 20 50 29 68 3.4 14 58 5.6 5.7 20000
30 22 63 33 126 1.8 14.5 66 9.4 6.9 32000
32 18 48 28 110 3.2 15.5 75 8.7 4.9 46000
36 14 52 26 100 2.3 - 68 5.6 6.4 24000
38 27 76 37 126 2.2 12 64 6.8 7.3 22000
39 21 24 10.5 45 3.8 17 60 10.9 6.5 41000
43 13 40 21 100 33 16 58 6.4 6.9 21000
"""





