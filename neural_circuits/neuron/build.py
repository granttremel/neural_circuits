
import random

from pathlib import Path
import os
import yaml

from neural_circuits.neuron.neuron_o1 import (
    NeuralState1 as NeuralState, 
    Synapse1 as Synapse, 
    Neuron1 as Neuron, 
    InputNeuron1 as InputNeuron, 
    OutputNeuron1 as OutputNeuron,
    Circuit1 as Circuit
)

class CircuitParams:
    
    def __init__(self, name, defaults):
        self.name = name
        self.defaults = defaults
        self.params = {}
        self.allow_random = True
    
    def set_params(self, **params):
        allow_random = params.pop("allow_random", True)
        self.set_allow_random(allow_random)
        self.params.update(params)
    
    def __getattr__(self, att, default=None):
        if att.startswith("sd") and not self.allow_random:
            return 0.0
        if att in self.params:
            return self.params.get(att, default)
        elif att in self.defaults:
            return self.defaults.get(att, default)
        else:
            return default
        
    def get(self, att, default=None):
        if att.startswith("sd") and not self.allow_random:
            return 0.0
        if att in self.params:
            return self.params.get(att, default)
        elif att in self.defaults:
            return self.defaults.get(att, default)
        else:
            return default
        
    def set_allow_random(self, val):
        self.allow_random = val
        
class CircuitBuilder:
    
    input_types =["sinusoid","step","beat","random"]
    act_types =["ReLU","LeakyReLU","Sigmoid","Tanh","I"]
    
    def __init__(self):
        
        self._defaults = {}
        self._params = {}
        self._last_method = ""
        self.load_defaults()
    
    def load_defaults(self):
        
        params_dir = Path("./data/params")
        
        all_defs = os.listdir(params_dir)
        
        for eachdef in all_defs:
            fpath = params_dir / eachdef
            if not fpath.suffix == ".yaml":
                continue
            
            with open(fpath, "r") as f:
                obj = yaml.safe_load(f)
            
            def_key = eachdef.removesuffix("_defaults.yaml")
            cparams = CircuitParams(def_key, obj)
            
            self._defaults[def_key] = cparams
    
    def get_defaults(self, name):
        return self._defaults.get(name, {})
    
    def get_params(self, name, **params):
        ps = self.get_defaults(name)
        ps.set_params(**params)
        return ps        
    
    def load_params(self, key, name):
        
        file_name = f"{key}_{name}"
        full_name = file_name + ".yaml"
        
        params_dir = Path("./data/params")
        fpath = params_dir / full_name
        if not fpath.exists():
            return {}
        
        with open(fpath, "r") as f:
            obj = yaml.safe_load(f)
        
        cparams = CircuitParams(file_name, obj)
        return cparams
    
    def cache_params(self, calling_method, params):
        self._last_method = calling_method
        self._params = params
    
    def rebuild_last(self):
        # if self._params.
        if self._last_method == "linear":
            return self.build_linear("", **self._params.__dict__)
        else:
            return None
        
    def make_input_layer(self, circuit, params):
        
        num_inputs = params.num_inputs
        
        input_neurons = params.input_neurons
        input_type = params.input_type
    
        if input_neurons:
            for nin in input_neurons:
                circuit.add_neuron(nin, layer = "Input")
        else:
            for ni in range(num_inputs):
                inp = self.make_input_neuron(input_type, params)
                circuit.add_neuron(inp, layer = "Input")
        
        return circuit

    def make_input_neuron(self, input_type, params):
        
        if input_type=="sinusoid":
            return self._make_sine_input(params)
        
        elif input_type=="step":
            return self._make_step_input(params)
        
        elif input_type=="beat":
            return self._make_step_input(params)
        
        elif input_type=="random":
            return self._make_random_input(params)
        else:
            return None
    
    def _make_sine_input(self, params):
        bias = params.input_sine_bias
        sd_bias = params.sd_input_sine_bias
        if sd_bias:
            bias = random.normalvariate(bias, sd_bias)
        
        f = params.input_sine_f
        sd_f = params.sd_input_sine_f
        if sd_f:
            f = random.normalvariate(f, sd_f)
        
        phase = params.input_sine_phase
        sd_phase = params.sd_input_sine_phase
        if sd_phase:
            phase = random.random()*sd_phase + phase
            
        return InputNeuron.sinusoid(f, phase, bias)
    
    def _make_step_input(self, params):
        
        t_on = params.input_step_t_on
        sd_t_on = params.sd_input_step_t_on
        if sd_t_on:
            t_on = random.randint(t_on-sd_t_on//2, t_on+sd_t_on//2)
        t_off = params.input_step_t_off
        sd_t_off = params.sd_input_step_t_off
        if sd_t_off:
            t_off = random.randint(t_off-sd_t_off//2, t_off+sd_t_off//2)
        
        return InputNeuron.step(t_on, t_off)
    
    def _make_beat_input(self, params):
        
        n_on = params.input_beat_n_on
        sd_n_on = params.sd_input_beat_n_on
        if sd_n_on:
            n_on = random.randint(n_on - sd_n_on//2, n_on + sd_n_on//2)
        
        n_off = params.input_beat_n_off
        sd_n_off = params.sd_input_beat_n_off
        if sd_n_off:
            n_off = random.randint(n_off - sd_n_off//2, n_off + sd_n_off//2)
            
        phase = params.input_beat_phase
        sd_phase = params.sd_input_beat_phase
        if sd_phase:
            phase = random.randint(phase - sd_phase//2, phase + sd_phase//2)
            
        return InputNeuron.beat(n_on, n_off, phase)
    
    def _make_random_input(self, params):
        p_on = params.p_on
        sd_p_on = params.sd_p_on
        if sd_p_on:
            p_on = random.normalvariate(p_on, sd_p_on)
        seed = params.get("seed", random.randint(0, 500))
        return InputNeuron.random(p_on, seed)
    
    def add_interneuron_layer(self, circuit, num_layer, synapse_density, presynaptic, params):
        
        num_neurons = params.num_neurons
        
        inter_bias = params.inter_bias
        sd_inter_bias = params.sd_inter_bias
        inter_act = params.inter_act_fn
        inter_p_inhib = params.inter_p_inhib
        inter_wgt = params.inter_weight
        sd_inter_wgt = params.sd_inter_weight
        inter_delay = params.inter_delay
        sd_inter_delay = params.sd_inter_delay
        
        inter_synapse_density = params.get("inter_synapse_density", synapse_density)
            
        currns = []
        for nnr in range(num_neurons):
            bias = random.normalvariate(inter_bias,sd_inter_bias)
            inh = bool(random.random() < inter_p_inhib)
            newn = Neuron(bias = bias, act_fn = inter_act, inhib = inh)
            currns.append(newn)
            circuit.add_neuron(newn, layer = num_layer)
        
        for ln in presynaptic:
            for currn in currns:
                roll = random.random()
                if roll > inter_synapse_density:
                    continue
                
                wgt = random.normalvariate(inter_wgt,sd_inter_wgt)
                dl = random.randint(inter_delay-sd_inter_delay//2, inter_delay + sd_inter_delay//2)
                syn = Synapse(pre = ln, post = currn, weight = wgt)
                syn.add_to_neurons()
                
        return circuit
    
    def add_feedforward(self, circuit, num_layers, synapse_density,  params):
        
        num_feedforward = params.num_feedforward
        ff_synapse_density = params.ff_synapse_density
        
        ff_bias = params.ff_bias
        sd_ff_bias = params.sd_ff_bias
        ff_p_inhib = params.ff_p_inhib
        ff_act = params.ff_act_fn
        ff_wgt = params.ff_weight
        sd_ff_wgt = params.sd_ff_weight
        
        for nf in range(num_feedforward):
            
            from_layer = random.randint(0, num_layers - 2)
            to_layer = random.randint(from_layer, num_layers-1)
            
            from_nr = random.choice(circuit.get_layer(from_layer))
            to_nrs = [nr for nr in circuit.get_layer(to_layer) if random.random() < ff_synapse_density]
            
            bias = random.normalvariate(ff_bias,sd_ff_bias)
            inh = bool(random.random() < ff_p_inhib)
            fbn = Neuron(bias = bias, act_fn= ff_act,inhib = inh, type= "Feedforward")
            circuit.add_neuron(fbn, layer = "Special")
            
            wgt0 = random.normalvariate(ff_wgt,sd_ff_wgt)
            syn = Synapse(pre = from_nr, post = fbn, weight = wgt0)
            syn.add_to_neurons()
            
            for to_nr in to_nrs:
                wgt1 = random.normalvariate(ff_wgt,sd_ff_wgt)
                syn = Synapse(pre = fbn, post = to_nr, weight = wgt1)
                syn.add_to_neurons()
        
        return circuit

    def add_feedback(self, circuit, num_layers, synapse_density, params):
        
        num_feedback = params.num_feedback
        fb_synapse_density = params.fb_synapse_density
        
        fb_bias = params.fb_bias
        sd_fb_bias = params.sd_fb_bias
        fb_p_inhib = params.fb_p_inhib
        fb_act = params.fb_act_fn
        fb_wgt = params.fb_weight
        sd_fb_wgt = params.sd_fb_weight
        
        for nf in range(num_feedback):
        
            to_layer = random.randint(0, num_layers - 2)
            from_layer = random.randint(to_layer, num_layers-1)
            
            from_nr = random.choice(circuit.get_layer(from_layer))
            to_nrs = [nr for nr in circuit.get_layer(to_layer) if random.random() < fb_synapse_density]
            
            bias = random.normalvariate(fb_bias,sd_fb_bias)
            inh = bool(random.random() < fb_p_inhib)
            fbn = Neuron(bias = bias, act_fn= fb_act,inhib = inh, type = "Feedback")
            circuit.add_neuron(fbn,layer = "Special")
            
            wgt0 = random.normalvariate(fb_wgt,sd_fb_wgt)
            syn = Synapse(pre = from_nr, post = fbn, weight = wgt0)
            syn.add_to_neurons()
            
            for to_nr in to_nrs:
                wgt1 = random.normalvariate(fb_wgt,sd_fb_wgt)
                syn = Synapse(pre = fbn, post = to_nr, weight = wgt1)
                syn.add_to_neurons()
        
        return circuit
    
    def add_output(self, circuit, presynaptic,  params):
        
        num_outputs = params.num_outputs
        
        out_bias = params.output_bias
        sd_out_bias = params.sd_output_bias
        out_act = params.output_act_fn
        out_wgt = params.output_weight
        sd_out_wgt = params.sd_output_weight
        
        currns = []
        for nnr in range(num_outputs):
            bias = random.normalvariate(out_bias, sd_out_bias)
            newn = OutputNeuron(bias = bias, act_fn = out_act)
            currns.append(newn)
            circuit.add_neuron(newn, layer = "Output")
        
        for ln in presynaptic:
            for currn in currns:
                wgt = random.normalvariate(out_wgt, sd_out_wgt)
                syn = Synapse(pre = ln, post = currn, weight = wgt)
                syn.add_to_neurons()
                
        return circuit
    
    def build_linear(self, param_name = "", **params):
        
        cust_params = {}
        if param_name:
            cust_params = self.load_params("linear", param_name)
            if cust_params:
                cust_params.set_params(**params)
                params = cust_params
        if not cust_params:
            params = self.get_params("linear", **params)
        
        self.cache_params("linear", params)
        
        num_layers = params.num_layers
        synapse_density = params.synapse_density

        circuit = Circuit()
        circuit = self.make_input_layer(circuit, params)
        
        presynaptic = circuit.get_layer("Input")
        for nl in range(num_layers):
            circuit = self.add_interneuron_layer(circuit, nl, synapse_density, presynaptic, params)
            presynaptic = circuit.get_layer(nl)
        
        circuit = self.add_feedforward(circuit, num_layers, synapse_density, params)
        circuit = self.add_feedback(circuit, num_layers, synapse_density, params)
        
        circuit = self.add_output(circuit, presynaptic, params)
        
        return circuit
    
    def build_random(self, num_neurons, num_synapses, mean_wgt = 0, sd_wgt = 1, mean_bias = 0.5, sd_bias = 0.1, act_fn = "ReLU"):
        
        ncirc = Circuit()
        for n in range(num_neurons):
            bias = random.normalvariate(mean_bias, sd_bias)
            neur = Neuron(bias = bias, act_fn = act_fn)
            ncirc.add_neuron(neur)
        
        init_cncs = range(num_neurons)
        random.shuffle(init_cncs)
        
        num_synapses = max(num_synapses, num_neurons+1)
        
        for n in range(num_synapses):
            srcind = init_cncs[n]
            tgtind = random.randrange(0, n)
            if tgtind == srcind:
                tgtind = (srcind+1)%num_neurons
            src = ncirc.neurons[srcind]
            tgt = ncirc.neurons[tgtind]
            wgt = random.normalvariate(mean_wgt, sd_wgt)
            tgt.add_synapse(src, wgt)
    
    def build_idk(self):
        
        circ = Circuit()
        
        inp = InputNeuron.beat(2, 3)
        
        
        
        
        
        pass
    
    # def connect_cliques(self, mean_wgt = 0, sd_wgt = 1, num_new_per_clique = 1):
        
    #     clcs = self.get_cliques()
        
    #     clc_keys = clcs.keys()
    #     num_clcs = len(clcs)
        
    #     for i in range(num_clcs - 1):
    #         clc1=  clc_keys[i]
    #         clc2 = clc_keys[i+1]
            
    #         c1n = random.sample(clcs[clc1])
    #         c2n = random.sample(clcs[clc2])
            
    #         self.neurons[c1n].add_synapse(c2n, random.normalvariate(mean_wgt, sd_wgt))
    
    # def get_cliques(self):
        
    #     fwd, rev = self.get_connectivity()
                
    #     clcs = {}
        
    #     known_neurs = self.neurons.copy()
    #     while len(known_neurs) > 0:
    #         to_check = []
    #         checked = set()
            
    #         nn = known_neurs.pop(0)
    #         clcs[nn.ind] = set()
    #         to_check.append(nn)
            
    #         while len(to_check) > 0:
    #             curr = to_check.pop(0)
    #             pres = rev.get(curr.ind)
    #             posts = fwd.get(curr.ind)
    #             for npre in pres:
    #                 nb = self.neurons[npre]
    #                 if nb.ind not in checked:
    #                     to_check.append(nb)
    #             for npost in posts:
    #                 nb = self.neurons[npost]
    #                 if nb.ind not in checked:
    #                     to_check.append(nb)
                
    #             checked.add(curr.ind)
    #             to_check.remove(curr)
    #             known_neurs.remove(curr)
            
    #         clcs[nn.ind] = checked
        
    #     return clcs

    