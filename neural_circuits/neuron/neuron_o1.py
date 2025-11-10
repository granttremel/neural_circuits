

from typing import Dict, List, Tuple, Any, Union, Callable, TYPE_CHECKING
from dataclasses import dataclass, field

from .neuron_o0 import NeuralState0, Synapse0, Neuron0, InputNeuron0, OutputNeuron0, Circuit0

class NeuralState1(NeuralState0):
    pass

class Synapse1(Synapse0):
    def get_activation(self, t):
        return self.weight_sign * self.weight * self.pre.activation(t - 1 - self.delay) * self.pre.is_spiking(t - 1 - self.delay)

class Neuron1(Neuron0):
    pass

class InputNeuron1(InputNeuron0):
    pass

class OutputNeuron1(OutputNeuron0):
    pass

class Circuit1(Circuit0):
    pass    



