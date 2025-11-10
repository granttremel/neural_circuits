

from typing import Dict, List, Tuple, Any, Union, Callable, TYPE_CHECKING
from dataclasses import dataclass, field

from .neuron_o0 import NeuralState0, Synapse0, Neuron0, InputNeuron0, Circuit0

class NeuralState15(NeuralState0):
    pass

class Synapse15(Synapse0):
    
    base_freq = 110 # Hz
    
    # interference is hard! .. wip
    
    def get_activation(self, t):
        return self.weight_sign * self.weight * self.pre.activation(t - 1 - self.delay) * self.pre.is_spiking(t - 1 - self.delay)

class Neuron15(Neuron0):
    
    def _calc_state(self, t):
        
        raw_act = 0
        
        for syn in self.synapses["pre"]:
            raw_act += syn.get_activation(t)
        
        act = self.act(raw_act)
        return NeuralState0(t, raw_act, act, self.bias)
    
    pass

class InputNeuron15(InputNeuron0):
    pass

class Circuit15(Circuit0):
    pass    



