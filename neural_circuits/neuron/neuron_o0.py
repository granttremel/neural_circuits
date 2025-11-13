
from typing import Dict, List, Tuple, Any, Union, Callable, TYPE_CHECKING
from dataclasses import dataclass, field

import json
from pathlib import Path
import random
from tabulate import tabulate
import os

import numpy as np

from .utils import Activation
from .. import draw

@dataclass
class NeuralState0:
    t:int
    raw_activation:float
    activation:float
    bias:float
    
    @property
    def is_spiking(self):
        return self.activation != 0.0
    
class Synapse0:
    
    def __init__(self, **kwargs):
        
        self.pre:'Neuron0' = kwargs.get("pre")
        self.post:'Neuron0' = kwargs.get("post")
        self.weight = abs(kwargs.get("weight", 0.0))
        self.weight_sign = -1 if self.pre.inhib else 1
        self.delay = kwargs.get("delay", 0)

    def add_to_neurons(self):
        self.pre.add_synapse(synapse=self)
        self.post.add_synapse(synapse=self)
        
    def get_activation(self, t):
        return self.weight_sign * self.weight * self.pre.is_spiking(t - 1 - self.delay)

    def __repr__(self):
        parts = [f"Pre: {self.pre.string_id()}"]
        parts.append(f"Post: {self.post.string_id()}")
        # parts.append(f"Weight: {self.weight}")
        parts.append(f"Weight: {self.weight_sign * self.weight:0.3f}")
        return ", ".join(parts)

    def to_dict(self):
        return {
            "_type":type(self).__name__,
            "pre":self.pre.ind,
            "post":self.post.ind,
            "weight":self.weight,
            "weight_sign":self.weight_sign,
            "delay":self.delay,
        }

class Neuron0:
    """
    basic neuron with boolean valued state, synaptic connections, non-linear activation, bias    
    """
    
    def __init__(self, **kwargs):
        
        self.ind = -1
        self.synapses:Dict[str,List[Synapse0]] = {
            "pre":[],
            "post":[]
        }
        
        self.bias = kwargs.get("bias", 0)
        self.act = Activation(kwargs.get("act_fn","ReLU"))
        self.inhib = kwargs.get("inhib", False)
        
        self.type = kwargs.get("type", "")
        self.layer = ""
        self.active = True
        
        self._state_cache:Dict[int, NeuralState0] = {}

    @property
    def num_presynaptic(self):
        return len(self.synapses["pre"])

    @property
    def num_postsynaptic(self):
        return len(self.synapses["post"])

    @property
    def num_synapsing(self):
        return self.num_presynaptic + self.num_postsynaptic

    @property
    def is_input(self):
        return "Input" in type(self).__name__
    
    def toggle(self, active = False):
        self.active = active
    
    def initialize(self, shallow = False):
        self._state_cache.clear()
        if shallow:
            state = NeuralState0(0, 0.0, 0.0, self.bias)
            self._state_cache[0] = state
        return self.get_state(0)

    def get_state(self, t)->NeuralState0:
        if t < 0:
            t = 0
        
        if not self.active:
            return NeuralState0(t, -1, -1, self.bias)
            
        if not t in self._state_cache:
            state = self._calc_state(t)
            self._state_cache[t] = state
        return self._state_cache[t]

    def _calc_state(self, t):
        
        raw_act = self.bias
        
        for syn in self.synapses["pre"]:
            raw_act += syn.get_activation(t)
        
        act = self.act(raw_act)
        return NeuralState0(t, raw_act, act, self.bias)
    
    def is_spiking(self, t):
        return self.get_state(t).is_spiking
    
    def activation(self, t):
        return self.get_state(t).activation
    
    def add_synapse(self, **kwargs):
        """
        adds new synapse, pre or
        """
        
        if kwargs.get("synapse"):
            synapse = kwargs.get("synapse")
            if self is synapse.post:
                if not synapse in self.synapses["pre"]:
                    self.synapses["pre"].append(synapse)
            else:
                if not synapse in self.synapses["post"]:
                    self.synapses["post"].append(synapse)
        else:
            pre = kwargs.get("pre")
            wgt = kwargs.get("weight")
            delay = kwargs.get("delay")
            newsyn = Synapse0(pre=pre, post=self, weight=wgt, delay=delay)
            newsyn.add_to_neurons()
        
    def get_presynaptic_neurons(self):
        return [syn.pre for syn in self.synapses["pre"]]
        
    def get_postsynaptic_neurons(self):
        return [syn.post for syn in self.synapses["post"]]
        
    def is_connected(self, other_neuron:'Neuron0'):
        return self.is_presynaptic_of(other_neuron) or other_neuron.is_presynaptic_of(self)
    
    def is_presynaptic_of(self, other_neuron:'Neuron0'):
        return self in other_neuron.get_presynaptic_neurons()
    
    def is_postsynaptic_of(self, other_neuron:'Neuron0'):
        return self in other_neuron.get_postsynaptic_neurons()

    def get_stats(self, t_max):
        
        states = [self.get_state(t) for t in range(t_max)]
        
        raw_acts =[s.raw_activation for s in states]
        
        return np.mean(raw_acts), np.std(raw_acts), np.min(raw_acts), np.max(raw_acts), len(raw_acts)

    def __repr__(self):
        parts = [f"Neuron {self.ind}"]
        parts.append(f"layer {self.layer} ({self.string_id()})")
        parts.append("Inhibitory" if self.inhib else "Excitatory")
        if self.type:
            parts.append(self.type)
        parts.append(f"bias = {self.bias:0.3f}")
        parts.append(f"{self.num_presynaptic} presynaptic")
        parts.append(f"{self.num_postsynaptic} postsynaptic")
        return ", ".join(parts)
        # return f"Neuron {self.ind}, layer {self.layer}, {inh_str}, {self.type}, bias = {self.bias:0.3f}, {self.num_presynaptic} presynaptic, {self.num_postsynaptic} postsynaptic"
    
    def string_id(self):
        return f"L{self.layer}-N{self.ind}"

    def to_dict(self):
        return {
            "_type":type(self).__name__,
            "ind":self.ind,
            "synapses": {k: [s.to_dict() for s in syns] for k, syns in self.synapses.items()},
            "bias":self.bias,
            "act": self.act.to_dict(),
            "inhib":self.inhib,
            "type":self.type,
            "layer":self.layer
        }
    
    @classmethod
    def from_dict(cls, ndict):
        
        nrn = cls(
            bias = ndict.get("bias"), 
            act_fn = ndict.get("act").get("act_fn"), 
            inhib = ndict.get("inhib"),
            type = ndict.get("type")
        )
        nrn.ind = ndict.get("ind")
        return nrn

class InputNeuron0(Neuron0):
    """
    just kind of a signal generator
    """
    
    def __init__(self, **kwargs):
        
        super().__init__(**kwargs)
        
        self.signal_fn = kwargs.get("signal_fn","")
        self.signal_params = kwargs.get("signal_params",{})
        self.type = "Input"
    
    def _calc_state(self, t):
        raw_act = float(self.__call__(max(t,0)))
        act_val = self.act(raw_act)
        return NeuralState0(t, raw_act, act_val, self.bias)
    
    def add_synapse(self, **kwargs):
        """
        adds new synapse
        """
        if kwargs.get("synapse"):
            synapse = kwargs.get("synapse")
            if self is synapse.pre:
                self.synapses["post"].append(synapse)
    
    def __call__(self, t):
        if self.signal_fn == "sinusoid":
            return self._sinusoid(t)
        if self.signal_fn == "step":
            return self._step(t)
        if self.signal_fn == "beat":
            return self._beat(t)
        if self.signal_fn == "hill":
            return self._hill(t)
        if self.signal_fn == "random":
            return self._random(t)
    
    def _sinusoid(self, t):
        freq = self.signal_params.get("freq")
        phase = self.signal_params.get("phase")
        return float(np.sin(freq*t - phase))
    
    def _step(self, t):
        t_on = self.signal_params.get("t_on")
        t_off = self.signal_params.get("t_off")
        return int(t >= t_on and t < t_off)
    
    def _beat(self, t):
        n_on = self.signal_params.get("n_on")
        n_off = self.signal_params.get("n_off")
        phase = self.signal_params.get("phase")
        ntot = n_on + n_off
        return int((t-phase)%ntot < n_on)
    
    def _hill(self, t):
        var= self.signal_params.get("var")
        newval = (2*random.random()-1)*var + self.signal_params.get("last", 0)
        ema = self.signal_params.get("ema", 0)
        ema += newval * 0.2
        self.signal_params["ema"] = ema
        newval = newval - 0.1*ema
        self.signal_params['last'] = newval
        return newval
    
    def _random(self, t):
        seed = self.signal_params.get("seed")
        return int(self.signal_params.get("p_on") > random.random())
    
    @classmethod
    def sinusoid(cls, freq, phase, bias, act_fn = "I"):
        signal_params = {"freq":freq,"phase":phase}
        return cls(signal_fn = "sinusoid", bias = bias, act_fn = act_fn, signal_params = signal_params)
    
    @classmethod
    def step(cls, t_on, t_off):
        signal_params = {"t_on":t_on,"t_off":t_off}
        return cls(signal_fn = "step", bias = 0.5, act_fn = "I", signal_params = signal_params)
    
    @classmethod
    def beat(cls, n_on, n_off, phase):
        ntot = n_on+n_off
        phase = phase % ntot
        signal_params = {"n_on":n_on,"n_off":n_off, "phase":phase}
        return cls(signal_fn = "beat", bias = 0.0, act_fn = "I", signal_params = signal_params)
        
    @classmethod
    def hill(cls, var, bias):
        return cls(signal_fn = "hill",bias = bias, act_fn = "I", signal_params = {"var":var})
        
    @classmethod
    def random(cls, p_on, seed = 129):
        seed = random.random()*1e14
        signal_params = {"p_on":p_on,"seed":seed}
        return cls(signal_fn = "random", bias = 0, act_fn = "ReLU", signal_params = signal_params)

    def to_dict(self):
        ndict = super().to_dict()
        ndict["signal_fn"] = self.signal_fn
        ndict["signal_params"] = self.signal_params
        return ndict
        
    @classmethod
    def from_dict(cls, ndict):
        nrn = cls(
            bias = ndict.get("bias"), 
            act_fn = ndict.get("act").get("act_fn"), 
            type = ndict.get("type"),
            signal_fn = ndict.get("signal_fn"),
            signal_params = ndict.get("signal_params"),
        )
        nrn.ind = ndict.get("ind")
        return nrn
        

class OutputNeuron0(Neuron0):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.type = "Output"
    
    def add_synapse(self, **kwargs):
        """
        adds new synapse
        """
        if kwargs.get("synapse"):
            synapse = kwargs.get("synapse")
            if self is synapse.post:
                self.synapses["pre"].append(synapse)

class Circuit0:
    
    def __init__(self):
        self.neurons:List[Neuron0] = []
        self.layers:Dict[str, List[Neuron0]] = {"Default":[]}
        self.tstep:int = 0
        self._initialized = False
    
    @property
    def num_neurons(self):
        return len(self.neurons)
    
    @property
    def state(self):
        return [n.is_spiking(self.tstep) for n in self.neurons]
    
    def add_neuron(self, neur:Neuron0, layer = "Default"):
        neur.ind = len(self.neurons)
        self.neurons.append(neur)
        layer = str(layer)
        if not layer in self.layers:
            self.layers[layer] = []
        self.layers[layer].append(neur)
        neur.layer = layer
    
    def initialize(self, layer = None):
        inits = []
        if layer is not None:
            neurons = self.layers.get(str(layer))
        else:
            neurons = self.neurons
        
        if "Special" in self.layers:
            for nrn in self.layers.get("Special"):
                nrn.initialize(shallow=True)
        
        for n in neurons:
            initstate = n.initialize()
            inits.append(initstate)
        
        self._initialized = True
        return inits

    def print_system(self, show_synapsing =False, **kwargs):
        draw.print_system(self, show_synapsing = show_synapsing, **kwargs)
    
    def print_states(self, min_t = 0, max_t = -1, **kwargs):
        draw.print_states(self, min_t = min_t, max_t = max_t, **kwargs)
    
    def show_states(self, max_t = -1, **kwargs):
        draw.show_states(self, max_t = max_t, **kwargs)
    
    def show_activations(self, max_t = -1, bit_depth = 8, **kwargs):
        if bit_depth ==8:
            draw.show_activations_8b(self, max_t = max_t, **kwargs)
        else:
            draw.show_activations(self, max_t = max_t, **kwargs)
    
    def show_mean_dist(self):
        draw.show_mean_dist(self)
    
    def show_cov_dist(self):
        draw.show_cov_dist(self)
    
    def plot_net_activity(self, max_t = -1, **kwargs):
        draw.plot_net_activity(self, max_t=max_t, **kwargs)
    
    def get_neuron_states(self, t=None):
        return [nr.get_state(t or self.tstep) for nr in self.neurons]
    
    def get_all_neuron_states(self):
        return {nr.ind:[nr.get_state(t) for t in range(self.tstep)] for nr in self}
        
    def get_layer(self, layer):
        return self.layers.get(str(layer))
    
    def step(self):
        self.tstep += 1
        self.update(self.tstep)
        return self.state
        
    def update(self, t):
        for nr in self.neurons:
            nr.get_state(t)
    
    def step_to(self, t):
        while self.tstep < t:
            self.step()
    
    def step_to_completion(self, max_t):
        
        act_state = [n.activation(self.tstep) for n in self.neurons]
        for i in range(max_t):
            
            newstate = self.step()
            new_act_state = [n.activation(self.tstep) for n in self.neurons if not n.is_input]
            
            if all([sn == so for sn, so in zip(new_act_state, act_state)]):
                break
            
            act_state = new_act_state
        return new_act_state
    
    def walk(self, t_max = 128):
        while True:
            for inp in self.layers.get("Input"):
                if inp.signal_fn == "beat":
                    inp.signal_params["phase"] = inp.signal_params["phase"] + inp.signal_params["n_on"]
                elif inp.signal_fn == "sinusoid":
                    inp.signal_params["phase"] = inp.signal_params["phase"] + 1/inp.signal_params["freq"]
                    
            self.initialize()
            self.step_to(t_max)
            
            os.system('clear' if os.name == 'posix' else 'cls')
            
            self.show_activations()
            input()
            
            
    def __getitem__(self, ind):
        return self.neurons[ind]
    
    def scale_weights(self, factor, exclude = [], p = None):
        
        for nrn in self:
            if nrn.layer in exclude:
                continue
            if p and random.random() > p:
                continue
            
            for syn in nrn.synapses.get("pre"):
                syn.weight *= factor
        
    def scale_biases(self, factor, exclude = [], p = None):
        
        for nrn in self:
            if nrn.layer in exclude:
                continue
            if p and random.random() > p:
                continue
            
            nrn.bias *= factor
        
    def balance_ei(self, factor, exclude = [], p = None):
        """
        factor > 1 means more inhibitory, less excitatory
        factor < 1 means more excitatory, less inhibitory
        """
        
        for nrn in self:
            if nrn.layer in exclude:
                continue
            if p and random.random() > p:
                continue
            for syn in nrn.synapses["post"]:
                if nrn.inhib:
                    syn.weight *= factor
                else:
                    syn.weight /= factor
    
    def add_noise(self, sigma, exclude = [], p = None):
        
        for nrn in self:
            if nrn.layer in exclude:
                continue
            if p and random.random() > p:
                continue
            nrn.bias += random.normalvariate(0, sigma)
            for syn in nrn.synapses["post"]:
                syn.weight += random.normalvariate(0, sigma)
        
    def get_connectivity(self):
        
        fwd = {}
        rev = {}
        
        for nn in self.neurons:
            rev[nn.ind] = []
            for nb in nn.synapses.get("pre"):
                preind = nb.pre.ind
                if nn.ind == preind:
                    print(f"neuron {nn.ind} connected to self!")
                rev[nn.ind].append(preind)
            fwd[nn.ind] = []
            for nb in nn.synapses.get("post"):
                postind = nb.post.ind
                if nn.ind == postind:
                    print(f"neuron {nn.ind} connected to self!")
                fwd[nn.ind].append(postind)
        
        return fwd, rev
    
    def print_stats_table(self):
        
        headers = ["Neuron", "Type","Layer","Bias","Mean","SD","Min","Max", "N"]
        
        rows = []
        for nrn in self:
            row = [nrn.ind, nrn.type, nrn.layer, nrn.bias, nrn.layer, *nrn.get_stats(self.tstep)]
            rows.append(row)
        
        print(tabulate(rows, headers = headers))
    
    def print_covariance_table(self):
        
        headers = []
        rows =[]
        
        acts = {nrn.ind:[nrn.get_state(t).raw_activation for t in range(self.tstep)] for nrn in self}
        
        for ni in range(self.num_neurons):
            row = [ni]
            headers.append(ni)
            for nj in range(self.num_neurons):
                if nj < ni:
                    row.append(" ")
                    continue
                
                cov_mat = np.cov(acts[ni], acts[nj])
                if ni != nj:
                    cv = cov_mat[0,1]/cov_mat[0,0]
                else:
                    cv = cov_mat[0,0]
                row.append(format(cv, "0.3f"))
            rows.append(row)
        print(tabulate(rows, headers = headers, floatfmt = "0.3f"))
    
    def to_dict(self):
        
        syns = []
        for nr in self.neurons:
            for syn in nr.synapses["pre"]:
                syns.append(syn.to_dict())
        
        return {
            "_type":type(self).__name__,
            "neurons":[n.to_dict() for n in self.neurons],
            "layers":{layer:[n.ind for n in ns] for layer, ns in self.layers.items()},
            "synapses":syns,
        }
    
    def to_json(self, fname):
        
        fdir = Path("./data/circuits")
        
        fpath = fdir / fname
        fpath = fpath.with_suffix(".json")
        
        outdict = self.to_dict()
        
        with open(fpath, "w+") as f:
            json.dump(outdict, f, indent = 3)
    
    @classmethod
    def from_dict(cls, cdict):
        
        circ = cls()
        
        index_map = {}
        
        for n in cdict.get("neurons"):
            if not n:
                continue
            if n.get("type") == "Input":
                nrn = InputNeuron0.from_dict(n)
            else:
                nrn = Neuron0.from_dict(n)
            index_map[nrn.ind] = nrn
            print(f"added {nrn.ind} to index map")
        
        for layer, ns in cdict.get("layers").items():
            for nind in ns:
                nrn = index_map.get(nind)
                if not nrn:
                    continue
                circ.add_neuron(nrn, layer=layer)
        
        for syndict in cdict.get("synapses"):
            preind = syndict.get("pre")
            postind = syndict.get("post")
            
            npre = index_map.get(syndict.get("pre"))
            npost = index_map.get(syndict.get("post"))
            if not npre or not npost:
                print(preind, postind)
                continue
            
            syn = Synapse0(pre = npre, post = npost, weight = syndict.get("weight"), delay = syndict.get("delay"))
            print(f"creating synapse between neurons {npre}, {npost}")
            
            syn.add_to_neurons()
        
        return circ
        
    @classmethod
    def from_json(cls, fname):
        fdir = Path("./data/circuits")
        
        fpath = fdir / fname
        fpath = fpath.with_suffix(".json")
        
        if not fpath.exists():
            return None
        
        with open(fpath, "r") as f:
            obj = json.load(f)
        
        return cls.from_dict(obj)
        
        
        
    