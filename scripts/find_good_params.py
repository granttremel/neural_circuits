

import sys
import tty
import os
import termios
import time

import numpy as np

from neural_circuits import draw
from neural_circuits.draw import Colors
from neural_circuits.neuron import neuron_o0
from neural_circuits.neuron import neuron_o1
from neural_circuits.neuron.neuron_o2 import (
    NeuralState2 as NeuralState, 
    Synapse2 as Synapse, 
    Neuron2 as Neuron, 
    InputNeuron2 as InputNeuron, 
    OutputNeuron2 as OutputNeuron, 
    Circuit2 as Circuit
)
from neural_circuits.neuron import build
from neural_circuits.neuron.tune import HandCircuitTuner

def get_input_neurons():
    
    in1 = InputNeuron.beat(1, 1, 0)
    in2 = InputNeuron.beat(2, 3, 1)
    in3 = InputNeuron.beat(3, 5, 2)
    in4 = InputNeuron.beat(5, 8, 3)
    in4 = InputNeuron.beat(8, 13, 4)
    # in4 = InputNeuron.random(0.2)
    # print(f"in4 initialized with seed {in4.signal_params.get("seed")}")
    
    # for t in range(10):
    #     s=in4.get_state(t)
    #     print(s.raw_activation, s.activation, s.is_spiking)
    
    return [in1, in2, in3, in4]


def initialize_linear(**params):
    
    inps = get_input_neurons()
    
    bd = build.CircuitBuilder()
    
    circ = bd.build_linear(input_neurons = inps, allow_random = True, **params)
    return circ, bd

def generate_systems(num_systems=1):
    
    for ns in range(num_systems):
        circ, bd = initialize_linear(
            param_name = "test"
        )
        run_circuit(circ)
        # input("enter to keep going")


def run_circuit(circ, max_t = 128):
    
    circ.initialize()
    circ.step_to(max_t)
    circ.show_activations()
    for i in range(16):
        print()

def look_at_arrows():
    
    for arr in draw.all_arrows:
        print(" "*64 + f" this is just an arrow in context {Colors.SPIKE}{arr}{Colors.RESET} as usual! just normal stuff")
        
    
    pass

def start_tuner(circuit, builder):
    
    tnr = HandCircuitTuner(circuit, builder)
    
    circuit = tnr.start()
    

    
def main():
    generate_systems()
    
    
    # look_at_arrows()
    
    # circ, bd = initialize_linear(
    #     param_name = "test"
    # )
    
    # start_tuner(circ, bd)
    

    
if __name__=="__main__":
    main()

