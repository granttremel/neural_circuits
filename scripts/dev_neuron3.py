
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
    Circuit2 as Circuit,
    Dendrite2 as Dendrite
)

from neural_circuits.neuron import build

def hookup_inputs(neuron, mean_wgt = 0.3, sd_wgt = 0.15):
    
    for seg in neuron.dendrite.tree:
        
        wgt = random.normalvariate(mean_wgt, sd_wgt)
        
        new_neuron = InputNeuron.beat(n_on = 3, n_off = 5, phase= 0)
        
        new_syn = Synapse(pre = new_neuron, post = neuron, weight = wgt)
        new_syn.add_to_neurons(dend_seg = seg.ind)
    
    return neuron

def test_dend_seg(num_tests = 5):
    
    
    for n in range(num_tests):
        dend = Dendrite.initialize_random(20, 10, 2.5)
        # act=dend.integrate(0)
        neuron = Neuron(dendrite = dend, bias = 0.5)
        neuron = hookup_inputs(neuron)
        for ds in dend.tree:
            pres = [str(d.ind)for d in ds.pre]
            
            if ds.is_base:
                print(f"ind = {ds.ind}, pre={",".join(pres)}, length={ds.l:0.3f}, length to soma={ds.length_to_soma():0.3f}, num_syns = {len(ds.synapses)}")
            elif ds.is_terminal:
                print(f"ind = {ds.ind}, post={ds.post.ind}, length={ds.l:0.3f} nm, length to soma={ds.length_to_soma():0.3f} nm, diameter = {ds.d:0.3f} um, num_syns = {len(ds.synapses)}")
            else:
                print(f"ind = {ds.ind}, post={ds.post.ind}, pre={",".join(pres)}, length={ds.l:0.3f} nm, length to soma={ds.length_to_soma():0.3f} nm, diameter = {ds.d:0.3f} um, num_syns = {len(ds.synapses)}")
            
        print("no integrating until now..")
        act = dend.integrate(0)
        print(f"tree activation at t=0: {act}")

        
        
        for t in range(1,5):
            
            s = neuron.get_state(t)
            # print(s.raw_activation, s.activation)
            print(f"neuron activation at t={t}: {s.activation:0.3f} ({s.raw_activation:0.3f})")
            
            # for ds in neuron.dendrite.tree:
            #     act = ds.integrate(t)
            #     print(f"segment {ds.ind} activation at time t={t}: {act:0.3f}")

def main():
    
    test_dend_seg(num_tests =1)
    
    pass    
    

if __name__=="__main__":
    main()


