

from typing import Dict, List, Tuple, Any, Union, Callable, TYPE_CHECKING
from dataclasses import dataclass, field

import random

from .neuron_o0 import NeuralState0, Synapse0, Neuron0, InputNeuron0, OutputNeuron0, Circuit0

class NeuralState2(NeuralState0):
    pass

class Synapse2(Synapse0):
    
    def add_to_neurons(self, dend_seg = 0):
        self.pre.add_synapse(synapse = self, is_post = False)
        self.post.add_synapse(synapse = self, is_post = True, dend_seg = dend_seg)
    
    def get_activation(self, t):
        return self.weight_sign * self.weight * self.pre.activation(t - 1 - self.delay) * self.pre.is_spiking(t - 1 - self.delay)

@dataclass
class DendriticSegment:
    ind:int
    l:float
    l_to_soma:float = -1.0
    d:float = -1.0
    # for node at tip .-----(o)---axon---
    activation:float = 0.0
    pre:List['DendriticSegment'] = field(default_factory = list)
    post:'DendriticSegment' = None
    synapses:List['Synapse2'] = field(default_factory = list)

    _act_cache = {}

    @property
    def is_terminal(self):
        return len(self.pre) == 0

    @property
    def is_base(self):
        return self.ind == 0

    def attach(self, other:'DendriticSegment'):
        self.pre.append(other)
        other.post = self
        
    def attach_to(self, other:'DendriticSegment'):
        other.attach(self)

    def length_to_soma(self):
        if self.is_base:
            return self.l
        elif self.l_to_soma > 0.0:
            return self.l_to_soma
        else:
            lts = self.l + self.post.length_to_soma()
            self.l_to_soma = lts
            return lts

    def _initialize_segment(self):
        lts = self.length_to_soma()
        if lts>0:
            self.d = self.get_diameter(lts)
        elif self.is_base:
            pass
        else:
            print(f"dendritic segment {self.ind} could not intialize, zero length to soma")
    
    def integrate(self, t):
        act, nin = self._integrate(t)
        return act
    
    def _integrate(self, t):
        if t in self._act_cache:
            # print(f"returning from cache for segment {self.ind}")
            return self._act_cache[t]
        
        act = 0
        num_inputs = 0
        
        for syn in self.synapses:
            act += syn.get_activation(t)
            num_inputs += 1
        
        # print(f"integrating for dendritic segment {self.ind}. after synapses, num_inputs = {num_inputs}")
        for pre in self.pre:
            new_act, pre_inputs = pre._integrate(t)
            act += new_act
            num_inputs += 1
            
        # print(f"integrating for dendritic segment {self.ind}. after all, num_inputs = {num_inputs}")
        
        if num_inputs == 0:
            return 0
        mean_act = act / num_inputs
        self._act_cache[t] = (mean_act, num_inputs)
        return mean_act, num_inputs
    
    @property
    def is_attached(self):
        return self.post is not None or self.ind == 0
    
    @classmethod
    def get_diameter(cls, length_to_soma, factor = 0.1):
        return max(1.1, 5.0 - factor * length_to_soma)
        return factor / (length_to_soma + 1.0)

class Dendrite2:
    
    def __init__(self, segments):
        if not segments[0].is_base:
            segments[0].ind = 0
        self.tree:List['DendriticSegment'] = segments
        self._initialize()
    
    def _initialize(self):
        for seg in self.tree:
            seg._initialize_segment()
    
    def integrate(self, t):
        return self.base_segment.integrate(t)
        
    def add_synapse(self, synapse, seg_ind):
        self.tree[seg_ind].synapses.append(synapse)
        
    @property
    def base_segment(self):
        return self.tree[0]
    
    @classmethod
    def initialize_random(cls, num_segments, mean_length, sd_length):
        
        base_seg = DendriticSegment(0, 0.0)
        segments = [base_seg]
        while len(segments) < num_segments:
            
            p_branch = 1/len(segments)
            seg_len = random.normalvariate(mean_length, sd_length)
            new_seg = DendriticSegment(len(segments), seg_len)
            
            for seg in segments:
                if random.random() < p_branch:
                    seg.attach(new_seg)
                    break
            
            if not new_seg.post:
                new_seg.attach_to(random.choice(segments))
            if new_seg.is_attached:
                segments.append(new_seg)
    
        return cls(segments)
        
        
        
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
        
        self.dendrite:Dendrite2 = kwargs.get("dendrite", Dendrite2.initialize_random(20, 15, 2.5))
        
        
    def _calc_state(self, t):
        
        raw_act = self.dendrite.integrate(t)
        
        act = self.act(raw_act)
        return NeuralState0(t, raw_act, act, self.bias)
    
    def add_synapse(self, synapse=None, is_post = False, dend_seg = 0):
        
        if is_post:
            self.dendrite.add_synapse(synapse, dend_seg)
        else:
            self.synapses["pre"].append(synapse)
        
    
    pass

class InputNeuron2(InputNeuron0):
    pass
class OutputNeuron2(OutputNeuron0):
    pass

class Circuit2(Circuit0):
    pass    



