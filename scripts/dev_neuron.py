

import random

from neural_circuits.neuron import neuron_o0
from neural_circuits.neuron import neuron_o1
# from neural_circuits.neuron.neuron_o0 import (
#     NeuralState0 as NeuralState, 
#     Synapse0 as Synapse, 
#     Neuron0 as Neuron, 
#     InputNeuron0 as InputNeuron, 
#     Circuit0 as Circuit
# )
from neural_circuits.neuron.neuron_o1 import (
    NeuralState1 as NeuralState, 
    Synapse1 as Synapse, 
    Neuron1 as Neuron, 
    InputNeuron1 as InputNeuron, 
    OutputNeuron1 as OutputNeuron, 
    Circuit1 as Circuit
)

from neural_circuits.neuron import build

from neural_circuits import draw

def get_input_neurons():
    
    in1 = InputNeuron.sinusoid(0.5, 0, 0.25)
    in2 = InputNeuron.beat(1, 3, 0)
    in3 = InputNeuron.step(0, 5)
    return [in1, in2, in3]

def initialize_linear(**params):
    
    inps = get_input_neurons()
    
    bd = build.CircuitBuilder()
    
    circ = bd.build_linear(input_neurons = inps, **params)
    return circ

def initialize_linear_small():
    
    inps = get_input_neurons()
    
    bd = build.CircuitBuilder()
    
    # circ = bd.build_linear(param_name = "small", input_neurons = inps)
    circ = bd.build_linear(param_name = "small")
    return circ

def run_circuit(circ:Circuit, max_t = 20):
    
    circ.initialize()
    final_state = circ.step_to_completion(max_t)
    return final_state

def test_json(circ):
    
    print("before storing")
    circ.print_system()
    run_circuit(circ)
    
    circ.show_activations(raw=True)
    print()
    
    fname = "test_circuit.json"
    circ.to_json(fname)
    
    print("after reconstitution")
    same_circ = Circuit.from_json(fname)
    same_circ.print_system()
    run_circuit(same_circ)
    same_circ.show_activations(raw = True)
    print()



def main():
    
    # testc = initialize_linear(num_inputs =3, num_neurons = 10, num_outputs = 3)
    testc = initialize_linear_small()
    
    fwd, rev= testc.get_connectivity()
    
    print(fwd)
    print(rev)
    
    testc.print_system(show_synapsing = True)
    
    testc.initialize()
    
    test_json(testc)
    
    # res = run_circuit(testc,max_t = 50)
    
    # testc.show_activations(raw =True)
    

if __name__=="__main__":
    main()
