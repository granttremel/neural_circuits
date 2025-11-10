

from typing import Dict, List, Tuple, Any, Union, Callable, TYPE_CHECKING
from dataclasses import dataclass, field

import random

from .neuron_o0 import NeuralState0, Synapse0, Neuron0, InputNeuron0, OutputNeuron0, Circuit0

class NeuralState2(NeuralState0):
    pass

class Synapse2(Synapse0):
    
    def get_activation(self, t):
        return self.weight_sign * self.weight * self.pre.activation(t - 1 - self.delay) * self.pre.is_spiking(t - 1 - self.delay)

@dataclass
class DendriticSegment:
    v:float
    l:float
    
@dataclass
class DendriticNode:
    ind:int
    activation:float
    synapses:List[Any]

class Dendrite2:
    
    def __init__(self):
        
        self.nodes = []
        self.tree = {}
        
    @classmethod
    def initialize_random(cls, num_nodes, mean_length, sd_length, mean_branches):
        
        
        
        
        pass
    
@dataclass
class AxonicSegment:
    v:float
    l:float
    
@dataclass
class AxonicNode:
    ind:int
    activation:float
    synapses:List[Any]

class Axon2:
    
    def __init__(self):
        
        self.nodes = []
        self.tree = {}
        
    
    @classmethod
    def initialize_random(cls, num_nodes, mean_length, sd_length):
        
        pass
        
        
        

class Neuron2(Neuron0):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        
        
    
    def _calc_state(self, t):
        
        raw_act = 0
        
        for syn in self.synapses["pre"]:
            raw_act += syn.get_activation(t)
        
        act = self.act(raw_act)
        return NeuralState0(t, raw_act, act, self.bias)
    
    pass

class InputNeuron2(InputNeuron0):
    pass
class OutputNeuron2(OutputNeuron0):
    pass

class Circuit2(Circuit0):
    pass    



