

import numpy as np
import sys
import tty
import os
import termios
import time

from tabulate import tabulate

import random

from neural_circuits import draw
from neural_circuits.draw import Colors

from neural_circuits.neuron.neuron_o1 import (
    NeuralState1 as NeuralState, 
    Synapse1 as Synapse, 
    Neuron1 as Neuron, 
    InputNeuron1 as InputNeuron, 
    OutputNeuron1 as OutputNeuron,
    Circuit1 as Circuit
)


class CircuitTuner:
    
    def __init__(self, builder, t_max, input_params = {}, **kwargs):
        self.builder = builder
        self.circuit = builder.build_linear(**kwargs)
        self.t_max = t_max
        self.input_params = input_params
        self.run_circuit()
        
    def set_input_params(self, params):
        self.input_params = params
        
    def run_circuit(self):
    
        self.circuit.initialize()
        self.circuit.step_to(self.t_max)
    
    def quiet(self, neuron, v):
        for syn in neuron.synapses.get("pre"):
            syn.weight += -syn.weight_sign * v
    
    def loud(self, neuron, v):
        for syn in neuron.synapses.get("pre"):
            syn.weight += syn.weight_sign * v

    def differ(self, neuron1, neuron2, v):
        
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
    
    def closen(self, neuron1, neuron2, v):
    
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

    # def show_mean_dist(self):
        
    #     means = []
    #     for nr in self.circuit:
    #         states = [nr.get_state(t) for t in range(self.circuit.tstep)]
    #         acts = [s.activation * s.is_spiking for s in states]
    #         means.append(np.mean(acts))
        
    #     means_srt = list(sorted(means, reverse = True))
        
    #     sctxt = draw.scalar_to_text_nb(means_srt, fg_color = 28, add_range = True)
    #     print("Mean distribution")
    #     for r in sctxt:
    #         print(r)
    #     print()
        
    # def show_cov_dist(self):
    
    #     covs = []
    #     for ni in range(len(self.circuit.neurons)):
    #         nri = self.circuit[ni]
    #         istates = [nri.get_state(t) for t in range(self.circuit.tstep)]
    #         iacts = [s.activation * s.is_spiking for s in istates]
    #         for nj in range(ni+1, len(self.circuit.neurons)):
    #             nrj = self.circuit[nj]
    #             if not nri.layer == nrj.layer:
    #                 continue
                
    #             jstates = [nrj.get_state(t) for t in range(self.circuit.tstep)]
    #             jacts = [s.activation * s.is_spiking for s in jstates]
    #             cov_mat = np.cov(iacts, jacts)
    #             if cov_mat[0,0] != 0:
    #                 cv = cov_mat[0,1] / cov_mat[0,0]
    #             else:
    #                 cv=  0.0
    #             covs.append(cv)
        
    #     covs_srt = list(sorted(covs, reverse = True))
        
    #     sctxt = draw.scalar_to_text_nb(covs_srt, fg_color = 125, add_range = True)
    #     print("Covariance distribution")
    #     for r in sctxt:
    #         print(r)
    #     print()

    def tune_means(self, tgt_mean, r = 0.05):
        for layer in self.circuit.layers:
            for nrn in self.circuit.layers[layer]:
                if nrn.type == "Input":
                    continue
                
                states = [nrn.get_state(t) for t in range(self.circuit.tstep)]
                acts_i = [s.raw_activation*s.is_spiking for s in states]
                mean = np.mean(acts_i)
                
                dmean = r*abs(tgt_mean - mean)
                if mean < tgt_mean:
                    self.loud(nrn, dmean)
                else:
                    self.quiet(nrn, dmean)
        return self.circuit

    def tune_covariances(self, r=0.05):
        
        for layer in self.circuit.layers:
            if layer == "Input":
                continue
            
            tgt_cov = 1/np.sqrt(len(self.circuit.layers[layer])) if len(self.circuit.layers[layer]) > 0 else 0.5
            
            for ni in range(len(self.circuit.layers[layer])):
                nrni = self.circuit.layers[layer][ni]
                for nj in range(ni + 1, len(self.circuit.layers[layer])):
                    nrnj = self.circuit.layers[layer][nj]
                    states_i = [nrni.get_state(t) for t in range(self.circuit.tstep)]
                    states_j = [nrnj.get_state(t) for t in range(self.circuit.tstep)]
                    
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
                        self.closen(nrni, nrnj, dcv)
                    elif cv > tgt_cov:
                        self.differ(nrni, nrnj, dcv)
        
    def iter_tune_both(self, num_means = 5, num_covs = 5, rmeans = None, rcovs = []):
        
        if not rmeans:
            rmeans = lambda n: 0.05
        
        if not rcovs:
            rcovs = lambda n: 0.05
        
        all_states = []
        
        self.circuit.print_system()
        
        self.circuit.initialize()
        self.circuit.step_to(self.t_max)
        self.circuit.print_stats_table()
        self.circuit.show_activations(show_stats = True)
        self.circuit.show_mean_dist()
        self.circuit.show_cov_dist()
        input()
        all_states.append(self.circuit.get_all_neuron_states())
        
        tgt_mean = 0.5
        
        nms = 0
        ncovs = 0
        
        while nms < num_means or ncovs < num_covs:
            if nms < num_means:
                print(f"Mean tuning round {nms+1}")
                self.tune_means(tgt_mean, rmeans(nms))
                self.circuit.initialize()
                self.circuit.step_to(self.t_max)
                
                # self.circuit.show_activations(show_stats = True)
                # self.circuit.show_mean_dist()
                # self.circuit.show_cov_dist()
                # input()
                nms += 1
            
            if ncovs < num_covs:
                print(f"Covariance tuning round {ncovs+1}")
                self.tune_covariances(rcovs(ncovs))
                self.circuit.initialize()
                self.circuit.step_to(self.t_max)
                all_states.append(self.circuit.get_all_neuron_states())
            
                # self.circuit.show_activations(show_stats = True)
                # self.circuit.show_mean_dist()
                # self.circuit.show_cov_dist()
                # input()
                ncovs +=1
            
        self.circuit.show_activations(show_stats = True)
        self.circuit.show_mean_dist()
        self.circuit.show_cov_dist()
        return all_states  
    
    def trend_stat_changes(self, all_states, show_means = True, show_sds = False, show_covs = True, layerdict = {}):
        
        means = {}
        sds = {}
        covs = {}
        
        for nt in all_states:
            for ni in nt:
                if not ni in means:
                    means[ni] = []
                    sds[ni]= []
                acti = [s.raw_activation*s.is_spiking for s in nt[ni]]
                means[ni].append(np.mean(acti))
                
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
        
        if show_sds:
            print("sds")
            for ni in sds:
                sctxt = draw.scalar_to_text_nb(sds[ni], minval = -1, maxval = 1, bit_depth=16)
                for r in sctxt:
                    print(ni, r)
                print()
        
        if show_covs:
            print("covariances")
            for (ni, nj) in covs:
                if ni == nj or ni < 3 or nj < 3:
                    continue
                sctxt = draw.scalar_to_text_nb(covs[(ni, nj)], bit_depth = 16,add_range = True)
                for r in sctxt:
                    print(ni, nj, r)
                print()                 

class HandCircuitTuner:
    
    subcursor_effect = Colors.TUNE + "▶{data:0.3f}◀" + Colors.RESET
    
    def __init__(self, circuit, builder):
        
        self.circuit = circuit
        self.builder = builder
        
        self.msgs = []
        self.res = 0.1
        self.pmod = 0.5
        
        self.num_input= len(self.circuit.layers["Input"])
        self.num_neurons = len(self.circuit.neurons)
        self.cursor = None
        self.cursor_neuron = None
        self.subcursor = None
        
        self.diffs = [1.0,1.0,1.0,1.0]
        
    def start(self):
        
        self.run_circuit()
        
        self.cursor = len(self.circuit.layers["Input"])
        self.subcursor = 0
        try:
            self._run_tuner()
        except KeyboardInterrupt:
            self._finish()
        
        
    def _run_tuner(self):
        
        while True:
            
            os.system('clear' if os.name == 'posix' else 'cls')
            
            time.sleep(0.01)
            print()
            self.show_activations()
            time.sleep(0.01)
            
            print()
            print(", ".join(self.msgs))
            print()
            
            self.msgs = []
            self.cursor_neuron = self.circuit[self.cursor]
            self.display_cursor_info()
            
            combo, key = self._get_key()
            
            if 'q' in key.lower():
                break
            
            elif key == '\x1b[A':  # Up arrow or 'k'
                if not self.cursor_neuron.synapses["pre"]:
                    continue
                self.modify_data(1)
                
            elif key == '\x1b[B':  # Down arrow or 'j'
                if not self.cursor_neuron.synapses["pre"]:
                    continue
                self.modify_data(-1)
                    
            elif key == '\x1b[C':  # Right arrow or 'l'
                self.move_cursor(1)
                
            elif key == '\x1b[D':  # Left arrow or 'h'
                self.move_cursor(-1)
            
            elif key == "g":
                self.add_noise()
                
            elif key == "a":
                self.move_subcursor(-1)
                
            elif key == "d":
                self.move_subcursor(1)
            
            elif key == "n":
                self.msgs.append("built new circuit")
                self.rebuild_circuit()
                
            else:
                continue
            
            self.run_circuit()
    
    def move_cursor(self, direction):
        self.cursor += direction
        self.cursor = (self.cursor - self.num_input) % (self.num_neurons - self.num_input) + self.num_input
    
    def move_subcursor(self, direction):
        num_items = 1 + self.cursor_neuron.num_synapsing 
        self.subcursor += direction
        self.subcursor = self.subcursor % num_items
    
    def modify_data(self, sign):
        
        sgnstr = "Increased" if sign>0 else "Decreased" 
        
        if self.subcursor == 0:
            data_label = "bias of neuron"
            old_data = self.cursor_neuron.bias
            self.cursor_neuron.bias += sign*self.res
            new_data = self.cursor_neuron.bias
        else:
            data_label = "weight of synapse"
            syns = self.cursor_neuron.synapses["pre"] + self.cursor_neuron.synapses["post"]
            syn = syns[self.subcursor - 1]
            old_data = syn.weight
            syn.weight += sign*self.res
            new_data = syn.weight
        self.msgs.append(f"{sgnstr} {data_label} from {old_data:0.3f} to {new_data:0.3f}")
    
    def rebuild_circuit(self):
        self.circuit = self.builder.rebuild_last()
        self.run_circuit()
    
    def add_noise(self):
        print("idk")
        pass
    
    def display_cursor_info(self):
        isb = 1
        print(f"Currently selected:")
        ninfo =self.format_neuron_info(self.cursor_neuron)
        print(ninfo)
        print(self.format_neuron_stats(self.cursor_neuron))
        headers = ["I/O", "Pre", "Post", "Weight"]
        rows = []
        for syn in self.cursor_neuron.synapses.get("pre",[]):
            uline = isb == self.subcursor
            synrow = self.format_synapse_row(syn, emphasize_weight= uline)
            rows.append(synrow)
            isb += 1
        for syn in self.cursor_neuron.synapses.get("post",[]):
            uline = isb == self.subcursor
            synstr = self.format_synapse_row(syn, emphasize_weight = uline)
            rows.append(synstr)
            isb += 1
        print(tabulate(rows, ))
        
    def format_neuron_info(self, nrn):    
        parts = [f"Neuron {nrn.ind}"]
        parts.append(f"layer {nrn.layer} ({nrn.string_id()})")
        parts.append("Inhibitory" if nrn.inhib else "Excitatory")
        if nrn.type:
            parts.append(nrn.type)
        if self.subcursor == 0:
            emph_str = self.subcursor_effect.format(data=nrn.bias)
            parts.append(f"bias = {emph_str}")
        else:
            parts.append(f"bias = {nrn.bias:0.3f}")
        parts.append(f"{nrn.num_presynaptic} presynaptic")
        parts.append(f"{nrn.num_postsynaptic} postsynaptic")
        return ", ".join(parts)
    
    def format_neuron_stats(self, neuron):
        mean, sd, minn, maxx, _ = neuron.get_stats(self.circuit.tstep)
        return f"Mean = {mean:0.3f}, SD = {sd:0.3f}, min = {minn:0.3f}, max = {maxx:0.3f}"
    
    def format_synapse_info(self, syn, emphasize_weight = False):
        iostr = "I" if syn.post is self.cursor_neuron else "O"
        rows = [iostr]
        if syn.pre.inhib:
            parts = [f"Pre: {Colors.NEG_TEXT}{syn.pre.string_id()}{Colors.RESET}"]
        else:
            parts = [f"Pre: {syn.pre.string_id()}"]
        if syn.post.inhib:
            parts.append(f"Post: {Colors.NEG_TEXT}{syn.post.string_id()}{Colors.RESET}")
        else:
            parts.append(f"Post: {syn.post.string_id()}")
        if emphasize_weight:
            emph_str = self.subcursor_effect.format(data = syn.weight_sign * syn.weight)
            parts.append(f"Weight: {emph_str}")
        else:
            parts.append(f"Weight: {syn.weight_sign * syn.weight:0.3f}")
        return ", ".join(parts)
        
    def format_synapse_row(self, syn, emphasize_weight = False):
        iostr = "I" if syn.post is self.cursor_neuron else "O"
        row = [iostr]
        if syn.pre.inhib:
            row.append(f"{Colors.NEG_TEXT}{syn.pre.string_id()}{Colors.RESET}")
        else:
            parts = [f"{syn.pre.string_id()}"]
        if syn.post.inhib:
            row.append(f"{Colors.NEG_TEXT}{syn.post.string_id()}{Colors.RESET}")
        else:
            row.append(f"{syn.post.string_id()}")
        if emphasize_weight:
            emph_str = self.subcursor_effect.format(data = syn.weight_sign * syn.weight)
            row.append(f"{emph_str}")
        else:
            row.append(f"{syn.weight_sign * syn.weight:0.3f}")
        return row
    
    def show_activations(self, max_t = -1, minval = -1, maxval = 1,  raw = False):
        
        bit_depth = 8
        
        marg = "L{layer}-N{ind} "
        marg_frm = ">16"
        
        nnrs = self.circuit.num_neurons
        
        if max_t < 1:
            max_t = self.circuit.tstep
        
        header = [format(" ", marg_frm)] + ["." for n in range(max_t)]
        all_rows = []
        curr_layer = "Input"
        for nn in range(nnrs):
            nr = self.circuit[nn]
            
            if nr.layer != curr_layer:
                curr_layer = nr.layer
                all_rows.append(format("", marg_frm) + "-"*max_t)
            
            nnstates = [nr.get_state(t) for t in range(0, max_t)]        
            if raw:
                acts = [nst.raw_activation for nst in nnstates]
            else:
                acts = [nst.activation for nst in nnstates]
            acts_qt = draw.quantize(acts, bit_depth, maxval=maxval, minval=minval)
            spks = [nst.is_spiking for nst in nnstates]
            
            rmarg = marg.format(layer = nr.layer, ind = nr.ind)
            pre = post = ""
            if nr.inhib:
                pre += Colors.NEG_TEXT
                post += Colors.RESET
            row=[pre+format(rmarg,marg_frm)+post]
                
            for actq, spk in zip(acts_qt, spks):
                _, fgc = draw.get_act_color(1, spk, nr.inhib)
                
                sv = min(bit_depth-1, max(actq, 0))
                sym = draw.SCALE[sv]
                row.append(f"{fgc}{sym}{Colors.RESET}")
            suffix = ""
            if nn == self.cursor:
                cursor_sym = "◀"
                suffix = f" {Colors.BOLD}{Colors.SPIKE}{cursor_sym}{Colors.RESET}"
            all_rows.append(row + [suffix])
            
        print("".join(header))
        for row in all_rows:
            print("".join(row))
        print()

    def _get_key(self):
        """Get a single keypress from the user."""
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            key = sys.stdin.read(1)
            # self.msgs.append(f"key={repr(key)}")
            
            # Check for escape sequences (arrow keys and ctrl combinations)
            if key == '\x1b':
                next_chars = sys.stdin.read(2)
                # self.msgs[-1] += f" next_chars={repr(next_chars)}"
                key += next_chars
                
                prefix = suffix = ""
                # Check for Ctrl+Arrow combinations
                if next_chars == '[1':
                    # Read the next character to identify Ctrl+Arrow
                    extra = sys.stdin.read(2)
                    # self.msgs[-1] += f" extra={repr(extra)}"
                    if extra == ';5':
                        prefix = "ctrl"
                    elif extra == ';2':
                        prefix = "shift"
                    
                    final = sys.stdin.read(1)
                    # self.msgs[-1] += f" final={repr(final)}"
                    
                    if final == 'C':  # Ctrl+Right
                        suffix = 'right'
                        return prefix, suffix
                    elif final == 'D':  # Ctrl+Right
                        suffix = 'left'
                        return prefix, suffix
                    elif final == 'A':  # Ctrl+Right
                        suffix = 'up'
                        return prefix, suffix
                    elif final == 'B':  # Ctrl+Right
                        suffix = 'down'
                        return prefix, suffix
                    
                    key += extra
            
            return "", key
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    

    def run_circuit(self, max_t = 50, suppress = True):
        
        self.circuit.initialize()
        self.circuit.step_to(max_t)
        if not suppress:
            self.circuit.print_system()
            self.circuit.show_activations()
    
    
    def _finish(self):
        
        print("completed tuning. here are the results:")
        
        self.circ.print_stats_table()
        
        return self.circuit
    
    