
from dataclasses import dataclass, field
from typing import Dict, List, Any, Union, Optional
from enum import Enum, StrEnum
import string

import numpy as np

from neural_circuits.modeling.bio import Ion, IonData
from neural_circuits.modeling.physics import Consts,  WaterConsts, Conversions, Physics
from neural_circuits.modeling.units import BaseUnit, MagUnit, Units


conv = Conversions()
phys = Physics()

@dataclass
class IonChannel:
    ion:Ion
    
    g: float # pS, single-channel conductance
    v_act: Optional[float] = None
    t_act: Optional[float] = None # ms, time constant of activation
    
    v_inact: Optional[float] = None
    t_inact: Optional[float] = None # ms, time constant of inactivation
    
    c_act: Optional[float] = None
    c_dpd: Optional[Ion] = None # ion that c_act refers to
    
    tsien_type: string = ""
    struct_name: string = ""
    
    v_dpd: Optional[float] = None
    
    N:int = 1
    N_open: int = 0
    N_inact: int = 0
    
    t_open: float = -1.0
    
    z_g: int = 12
    
    p_active: Optional[float] = None # m/s ?
    p_inactive: Optional[float] = None # m/s ?

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
        
    def set_count(self, count):
        self.N = count
        
    def get_open(self, t, Vm, concs = {}):
        # yeah its harder than this
        
        
        # closed -> open
        r_OOC = phys.boltzmann(self.z_g, Vm, self.v_act)
        N_rem = self.N - self.N_inact - self.N_open
        N_open_new = int(N_rem * r_OOC) # idk
        
        # open -> inactive
        N_inact_new = 0
        if self.t_open > 0:
            dt = t - self.t_open
            N_inact_new = int(self.N_open * np.exp(-dt / self.t_inact))
            self.N_inact += N_inact_new
            
        self.N_open += N_open_new
        self.N_open -= N_inact_new
        
        if r_OOC > 0.01 and self.t_open < 0:
            self.t_open = t
        elif self.t_open > 0 and self.v_inact < Vm:
            # hmmmm
            
            pass
        
        
        return self.N_open
            
        
class NaChannel(IonChannel):
    
    def __init__(self, g, v_act, t_act, v_inact, t_inact, **kwargs):
        super().__init__(Ion.Na, g, v_act, t_act, v_inact, t_inact, **kwargs)

    @classmethod
    def Na1(cls):
        return cls(g = 1.0, v_act = -50, t_act = 1.0, v_inact = -30, t_inact = 0.25)
    
    @classmethod
    def Na2(cls):
        return cls(g = 1.0, t_act = 1.0, v_act = -65, t_inact = 0.25, v_inact = -30)
    
class KChannel(IonChannel):
    
    def __init__(self, g, v_act, t_act, v_inact, t_inact,  **kwargs):
        super().__init__(Ion.K, g, v_act, t_act, v_inact, t_inact,  **kwargs)
    
    @classmethod
    def K1_delayedrect(cls):
        return cls(g = 15, v_act = None, t_act = 1.5, v_inact = None, t_inact = 100)

    @classmethod
    def K_Atype(cls):
        return cls(g = 15,v_act = None,  t_act = 1.5, v_inact = None, t_inact = 25)
    
    @classmethod
    def K_BK(cls):
        """
        Hille 3ed, Table 5.1
        [Ca2+]_act = 1-1uM
        Voltage dpd = e-fold / 9 to 15 mV ( meaning, 10x change per given change in voltage)
        Single channel conductance = (100, 250) pS
        """
        return cls(g = 150, c_act = 0.005, c_dpd = Ion.Ca, v_dpd = 12)
    
    @classmethod
    def K_IK(cls):
        """
        Hille 3ed, Table 5.1
        [Ca2+]_act = 50-900 nM
        Voltage dpd = None
        Single channel conductance = (20, 80) pS
        """
        return cls(g = 50, c_act = 0.5e-3, c_dpd = Ion.Ca, v_dpd = 12)
    
    @classmethod
    def K_SK(cls):
        """
        Hille 3ed, Table 5.1
        [Ca2+]_act = 50-900 nM
        Voltage dpd = e-fold / 9 to 15 mV ( meaning, 10x change per given change in voltage)
        Single channel conductance = (4, 20) pS
        """
        return cls(g = 12, c_act = 0.5e-3, c_dpd = Ion.Ca, v_dpd = 12)
    
class CaChannel(IonChannel):
    
    def __init__(self, g, v_act, t_act, v_inact, t_inact,  **kwargs):
        super().__init__(Ion.Ca, g, v_act, t_act, v_inact, t_inact,  **kwargs)

    @classmethod
    def Cav1x(cls):
        """
        Hille 3ed, Table 4.1
        HVA
        Activation range: (-30, 0)
        Inactivation range: (-60, -10)
        Inactivation: "Very slow", > 500ms
        """
        return cls(g = 25, v_act = -15, t_act = None, v_inact = -40, t_inact = 1000, tsien_type = "L", struct_name = "Cav1.x")

    @classmethod
    def Cav2x(cls):
        """
        Hille 3ed, Table 4.1
        HVA
        Activation range: (-20, 0)
        Inactivation range: (-120, -30)
        Inactivation: "Partial", 50-80 ms
        """
        return cls(g = 13, v_act = -10, t_act = None, v_inact = -75, t_inact = 65, tsien_type = "P/Q, N, R", struct_name = "Cav2.x")

    @classmethod
    def Cav3x(cls):
        """
        Hille 3ed, Table 4.1
        LVA
        Activation range: (-70, 0)
        Inactivation range: (-100, -60)
        Inactivation: "Complete", 20-50 ms
        """
        return cls(g = 8, v_act = -35, t_act = None, v_inact = -80, t_inact = 35, tsien_type = "G, H, I", struct_name = "Cav3.x")

"""





Hille 3ed, Table 6.1
Some Fast Chemical Synapses
Postsynaptic Cell       Response    E_rev (mV)  Receptor
Frog sk. musc.          E           -5          nACh
Cat motoneuron          E           0           Glu
Crayfish leg musc.      E           6           Glu
Crayfish leg musc.      I           -72         GABA_A
Aplysia ganglion        I           -60         ACh
Cat motoneuron          I           -78         Gly
Hipp. pyr. cell         I           -70         GABA_A

Hille 3ed, Table 12.2
Na Channel Gating-Charge Densities of Nerve and Muscle
Tissue              Na channel-density (12 qe / um2)
Squid giant axon    125-160
Myxicol giant axon  53
Crayfish giant axon 183
Frog node of Ran.   1467
Rad node of Ran.    1058
Frog twitch musc.   325
Rat ventricle       22
Dog Purkinje fib.   100

Hille 3ed, Table 12.3
Conductances and Densities of Na Channels
Prep                g (pS)  T (deg C)   Channel density (1/um3)
Squid giant axon    4       9           330
Frog node           7.9     13          1900
Frog node           6.4     2-5         400-920
Rat node            14.5    20          700
Mouse sk. musc.     23.5    15          65
Bovine chromaffin   17      21          1.5-10


Hille 3ed, Table 12.4
Conductances and Densities of K Channels
Prep                g (pS)  T (deg C)   Channel density (1/um3)
Delayed-rectifier K channels
Squid giant axon    12      9           30
Squid giant axon    20      13          18
Snail neuron        2.4     14          7
Frog node           2.7-4.6 17          570-960
Frog node           10-50   15          110
Frog sk. musc.      15      22          30


Inward-rectifier channels
Tunicate egg        5       14          0.04
Frog sk musc.       26      21          1.3

K(Ca) channels
Snail neuron        19      20          --
Mammalian BK type   130-240 22          --

Transient A channels
Insect, snail,      5-23    22          --
  mammal


Hille 3ed, Table 12.6
Single-Channel Conductances of Neurotransmitter-Activated Channels
Prep                    Agonist     g (pS)      T (deg C)
Amphib. rep. bird. ..   ACh         20-40       8-27
Rat myotubes            ACh         49          22
Bovine chromaffin       ACh         44          21
Aplysia ganglion        ACh         8           27
Rat cerbellar granule   Glu         8,17,41,50  21
Locust muscle           Glu         130         21

Lamprey brain stem      Gly         12,20,30,46 22
Cultured mouse spinal   GABA        12,19,39,44 33
Crayfish muscle         GABA        9           23

"""