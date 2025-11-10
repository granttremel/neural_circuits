
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


def get_input_neurons():
    
    in1 = InputNeuron.beat(1, 1, 0)
    in2 = InputNeuron.beat(2, 3, 1)
    in3 = InputNeuron.beat(3, 5, 2)
    return [in1, in2, in3]

def initialize_linear(**params):
    
    inps = get_input_neurons()
    
    bd = build.CircuitBuilder()
    
    circ = bd.build_linear(input_neurons = inps, allow_random = True, **params)
    return circ

def try_deactivate(circ):
    
    max_t = 50
    
    inps = circ.layers.get("Input")
    
    print("all inputs active")
    circ.initialize()
    circ.step_to(max_t)
    circ.show_activations()
    
    for inp in inps:
        
        print(f"deactivating input neuron {inp.ind}")
        circ.initialize()
        inp.toggle(active = False)
        
        circ.step_to(max_t)
        circ.show_activations()
        
        inp.toggle(active = True)

def quiet(neuron, v):
    for syn in neuron.synapses.get("pre"):
        syn.weight += -syn.weight_sign * v

def loud(neuron, v):
    for syn in neuron.synapses.get("pre"):
        syn.weight += syn.weight_sign * v

def differ(neuron1, neuron2, v):
    
    n1ps = {syn.pre.ind:syn.weight for syn in neuron1.synapses.get("pre")}
    n2ps = {syn.pre.ind:syn.weight for syn in neuron2.synapses.get("pre")}
    
    perts = {}
    
    common = list(set(n1ps).intersection(n2ps))
    
    for syn in neuron1.synapses.get("pre"):
        if syn.pre.ind in common:
            delta = n1ps[syn.pre.ind] - n2ps[syn.pre.ind]
            sgn = -1 if delta < 0 else 1
            pert = sgn*v / 2
            perts[syn.pre.ind] = pert
            syn.weight += pert
            # print(f"differ: perturbing weight of neurons {neuron1.ind}, {neuron2.ind} from neuron {syn.pre.ind} by {pert} ({syn.weight})")
    
    for syn in neuron2.synapses.get("pre"):
        if syn.pre in perts:
            pert = -perts.get(syn.pre.ind)
            syn.weight += pert
            
def closen(neuron1, neuron2, v):
    
    n1ps = {syn.pre.ind:syn.weight for syn in neuron1.synapses.get("pre")}
    n2ps = {syn.pre.ind:syn.weight for syn in neuron2.synapses.get("pre")}
    
    perts = {}
    
    common = list(set(n1ps).intersection(n2ps))
    diff1 = list(set(n1ps).difference(n2ps))
    diff2 = list(set(n2ps).difference(n1ps))
    
    for syn in neuron1.synapses.get("pre"):
        if syn.pre.ind in common:
            delta = n1ps[syn.pre.ind] - n2ps[syn.pre.ind]
            sgn = 1 if delta < 0 else -1
            pert = sgn*v / 2
            perts[syn.pre.ind] = pert
            syn.weight += pert
            # print(f"closen: perturbing weight of neurons {neuron1.ind}, {neuron2.ind} from neuron {syn.pre.ind} by {pert} ({syn.weight})")
        elif syn.pre.ind in diff1:
            syn.weight += v / 2
        elif syn.pre.ind in diff2:
            syn.weight += v / 2
    
    for syn in neuron2.synapses.get("pre"):
        if syn.pre in perts:
            pert = -perts.get(syn.pre.ind)
            syn.weight += pert

def tune_means(circ, tgt_mean, r = 0.05):
    for nrn in circ:
        if nrn.type == "Input":
            continue
        
        states = [nrn.get_state(t) for t in range(circ.tstep)]
        # acts_i = [s.raw_activation for s in states]
        acts_i = [s.raw_activation*s.is_spiking for s in states]
        mean = np.mean(acts_i)
        
        dmean = r*abs(tgt_mean - mean)
        if mean < tgt_mean:
            loud(nrn, dmean)
        else:
            quiet(nrn, dmean)
    return circ

def tune_covariances(circ, r=0.05):
    
    for layer in circ.layers:
        if layer == "Input":
            continue
        
        tgt_cov = 1/np.sqrt(len(circ.layers[layer]))
        
        for ni in range(len(circ.layers[layer])):
            nrni = circ.layers[layer][ni]
            for nj in range(ni + 1, len(circ.layers[layer])):
                nrnj = circ.layers[layer][nj]
                states_i = [nrni.get_state(t) for t in range(circ.tstep)]
                states_j = [nrnj.get_state(t) for t in range(circ.tstep)]
                
                # acts_i = [nrni.get_state(t).raw_activation for t in range(circ.tstep)]
                # acts_j = [nrnj.get_state(t).raw_activation for t in range(circ.tstep)]
                
                acts_i = [s.raw_activation * s.is_spiking for s in states_i]
                acts_j = [s.raw_activation * s.is_spiking for s in states_j]
                
                cv_mat = np.cov(acts_i, acts_j)
                if cv_mat[0,0] != 0:
                    cv = cv_mat[0,1]/cv_mat[0,0]
                else:
                    cv = 1.0
                
                # dcv = r*abs(abs(cv) - tgt_cov)
                dcv = r*1
                if cv < tgt_cov:
                    closen(nrni, nrnj, dcv)
                elif cv > tgt_cov:
                    differ(nrni, nrnj, dcv)

def iter_tune_means(circ, num_rounds = 5):
    t_max = 50
    
    all_states = []
    
    circ.print_system()
    
    circ.initialize()
    circ.step_to(t_max)
    circ.print_stats_table()
    circ.show_activations()
    all_states.append(circ.get_all_neuron_states())
    
    tgt_mean = 0.5
    
    for n in range(num_rounds):
        
        tune_means(circ, tgt_mean)
        
        circ.initialize()
        circ.step_to(t_max)
        all_states.append(circ.get_all_neuron_states())
        
        if n%5 == 4:
            print(f"result at n={n}")
            circ.show_activations()
    
    print("after mean tuning")
    circ.show_activations()
    
def iter_tune_covs(circ, num_rounds = 5):
    t_max = 50
    
    all_states = []
    
    circ.print_system()
    
    circ.initialize()
    circ.step_to(t_max)
    circ.print_stats_table()
    circ.show_activations()
    all_states.append(circ.get_all_neuron_states())
    
    for n in range(num_rounds):
        tune_covariances(circ)
        circ.initialize()
        circ.step_to(t_max)
        all_states.append(circ.get_all_neuron_states())
    
    print("after cov tuning")
    circ.show_activations()
    return all_states

def iter_tune_both(circ, num_rounds = 5, rmeans = None, rcovs = []):
    t_max = 50
    
    if not rmeans:
        rmeans = lambda n: 0.05
    
    if not rcovs:
        rcovs = lambda n: 0.05
    
    all_states = []
    
    circ.print_system()
    
    circ.initialize()
    circ.step_to(t_max)
    circ.print_stats_table()
    circ.show_activations()
    all_states.append(circ.get_all_neuron_states())
    
    tgt_mean = 0.5
    
    for n in range(num_rounds):
        tune_means(circ, tgt_mean, rmeans(n))
        circ.initialize()
        circ.step_to(t_max)
        tune_covariances(circ, rcovs(n))
        circ.initialize()
        circ.step_to(t_max)
        all_states.append(circ.get_all_neuron_states())
    
    print("after both tuning")
    circ.show_activations()
    return all_states

def trend_stat_changes(all_states, show_means = True, show_covs = True, layerdict = {}):
    
    means = {}
    sds = {}
    covs = {}
    
    for nt in all_states:
        for ni in nt:
            if not ni in means:
                means[ni] = []
                sds[ni]= []
            acti = [s.raw_activation for s in nt[ni]]
            means[ni].append(np.mean([s.raw_activation*s.is_spiking for s in nt[ni]]))
            # sds[ni].append(np.std([s.raw_activation for s in nt[ni]]))
            
            # l = layerdict[ni]
            
            for nj in nt:
                
                if nj < ni:
                    continue
                if layerdict[nj] != layerdict[ni]:
                    continue
                
                actj = [s.raw_activation*s.is_spiking for s in nt[ni]]
                if not (ni, nj) in covs:
                    covs[(ni, nj)] = []
                cov_mat = np.cov(acti, actj)
                
                if ni == nj:
                    cv = cov_mat[0,0]
                else:
                    cv = cov_mat[1,0]/cov_mat[0,0]
                
                covs[(ni, nj)].append(cv)
    
    if show_means:
        print("means")
        for ni in means:
            sctxt = draw.scalar_to_text_nb(means[ni], minval = -1, maxval = 1, bit_depth=16)
            for r in sctxt:
                print(ni, r)
                print()
    
    # print("sds")
    # for ni in sds:
    #     sctxt = draw.scalar_to_text_nb(sds[ni], minval = -1, maxval = 1, bit_depth=16)
    #     for r in sctxt:
    #         print(ni, r)
    #     print()
    
    if show_covs:
        print("covariances")
        for (ni, nj) in covs:
            # if np.std(covs[(ni,nj)])<0.02:
            if ni == nj or ni < 3 or nj < 3:
                continue
            sctxt = draw.scalar_to_text_nb(covs[(ni, nj)], bit_depth = 16,add_range = True)
            for r in sctxt:
                print(ni, nj, r)
            print()
    # print(covs[(ni, nj)])
    

def main():

    circ = initialize_linear(
        num_layers=3,
        num_feedforward = 3,
        num_feedback = 0,
        inter_synapse_density = 0.75,
        inter_bias = 0.3,
        inter_weight = 0.6,
        sd_inter_weight = 0.1,
        sd_ff_weight= 0.1,
    )
    
    layerdict = {n.ind:n.layer for n in circ}
    
    nmax = 20
    rcinit = 0.3
    rcfinal = 0
    
    rmeans = lambda n:0.02
    rcovs = lambda n:rcinit*(1-n/nmax)
    
    # all_states = iter_tune_means(circ, num_rounds = 20, num_cov_rounds = 0)
    # all_states = iter_tune_covs(circ, num_rounds = 20)
    all_states = iter_tune_both(circ, num_rounds = nmax, rmeans = rmeans, rcovs = rcovs)
    input("enter to keep going")
    trend_stat_changes(all_states, show_means = False, layerdict = layerdict)
    
    

if __name__=="__main__":
    main()


