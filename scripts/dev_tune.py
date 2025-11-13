
import random
import numpy as np

from tabulate import tabulate

from neural_circuits import draw
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
from neural_circuits.neuron import tune


def get_input_neurons():
    
    in1 = InputNeuron.beat(1, 1, 0)
    in2 = InputNeuron.beat(2, 3, 1)
    in3 = InputNeuron.beat(3, 5, 2)
    in4 = InputNeuron.beat(5, 8, 3)
    # in5 = InputNeuron.sinusoid(0.5, 0, 0.0)
    # in6 = InputNeuron.sinusoid(0.2, 1, 0.0)
    # in7 = InputNeuron.sinusoid(0.1, 2, 0.0)
    # in8 = InputNeuron.random(0.2)
    in9 = InputNeuron.hill(0.2, -1e9)
    
    # return [in1, in2, in3, in4, in5, in6, in7, in8]
    # return [in1, in4, in5, in8, in9]

def get_beats():
    
    in1 = InputNeuron.beat(1, 1, 0)
    in2 = InputNeuron.beat(2, 3, 1)
    in3 = InputNeuron.beat(3, 5, 2)
    # in4 = InputNeuron.beat(5, 8, 3)
    in4 = InputNeuron.random(0.5)
    in5 = InputNeuron.random(0.15)
    in6 = InputNeuron.random(0.15)
    
    # return [in1, in2, in3, in4, in5, in6, in7, in8]
    return [in1, in2, in3, in4, in5, in6]

def initialize_linear(**params):

    inputs=  get_beats()

    bd = build.CircuitBuilder()
    
    circ = bd.build_linear(input_neurons = inputs, allow_random = True, **params)
    return circ, bd

def run_circuit(circ, max_t = 128):
    
    circ.initialize()
    circ.step_to(max_t)
    circ.show_activations(maxval = None, minval = None, show_stats = True)
    # for i in range(8):
    #     print()

def generate_systems(num_systems=1):
    
    for ns in range(num_systems):
        circ, bd = initialize_linear(
            param_name = "test"
        )
        run_circuit(circ)

def main():

    inputs=  get_beats()

    done = False
    while not done:
        circ, bd = initialize_linear(
            param_name = "test"
        )
        run_circuit(circ)
        
        circ.show_mean_dist()
        circ.show_cov_dist()
        
        res = input("done?")
        if 'y' in res.lower():
            break
    
    t_max = 128
    # tnr = tune.CircuitTuner(bd, t_max, param_name = "test", input_neurons = inputs)
    
    # tnr.circuit.show_mean_dist()
    # tnr.circuit.show_cov_dist()
    
    # rmeans = lambda x:0.05
    # rcovs = lambda x:0.1
    
    # all_states = tnr.iter_tune_both(num_means = 20, num_covs = 20, rmeans = rmeans, rcovs= rcovs)
    
    # circ = tnr.circuit
    
    circ.walk(t_max = t_max)
    

if __name__=="__main__":
    main()

