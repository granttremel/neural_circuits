
from typing import TYPE_CHECKING

import numpy as np

from tabulate import tabulate
# from neural_circuits.neuron.neuron_o1 import InputNeuron

if TYPE_CHECKING:
    from neural_circuits.neuron.neuron_o0 import Circuit0

SCALE = " ▁▂▃▄▅▆▇█"

SCALE_H = " ▏▎▍▌▋▊▉█"

RESET = "\x1b[0m"

OTHER={
    "upper_half":"▀",
    "upper_eighth":"▔",
    "lower_eighth":"▁",
    "right_half":"▐",
    "left_eighth":"▏",
    "right_eighth":"▕",
    "light":"░",
    "medium":"▒",
    "dark":"▓",
    "misc":"▖▗▘▙▚▛▜▝▞▟◐◑◒◓◔◕"
}

# https://www.calculators.org/math/html-arrows.php so many arrows.....
_arrows = "".join(chr(i) for i in range(0x2933, 0x2941 + 1))
_arrows2 = "".join(chr(i) for i in range(0x2962, 0x2965+1)) + str(chr(0x2970))
_arrows3 = "".join(chr(i) for i in range(0x2794, 0x27B2+1)) + str(chr(0x27BE)) # these symbols are cool in the right font
_arrows4 = "⤏⤎⤍⤌⟿⟾⟽⟹⟶⟵"

all_arrows = "".join(chr(i) for i in range(0x2190, 0x21FF + 1)) + "".join(chr(i) for i in range(0x27F0, 0x27FF+1)) + "".join(chr(i) for i in range(0x2900, 0x2974+1)) + "".join(chr(i) for i in range(0x2798, 0x27BE+1))

class Colors:
    
    RESET = "\x1b[0m"
    SPIKE = "\x1b[38;5;208m"
    NEG_SPIKE = "\x1b[38;5;56m"
    NEG_TEXT = "\x1b[38;5;93m"
    DARK = "\x1b[38;5;8m"
    HIDE = "\x1b[38;5;234"
    
    BG_SPIKE = "\x1b[48;5;214m"
    BG_NEG_SPIKE = "\x1b[48;5;56m"
    BG_DARK = "\x1b[48;5;8m"
    BG_HIDE = "\x1b[48;5;234"
    
    TUNE = "\x1b[38;5;160m"
    
    BOLD = "\x1b[1m"
    FAINT = "\x1b[2m"
    IDK = "\x1b[3m"
    UNDERLINE = "\x1b[4m"


def scalar_to_text_8b(scalars, minval = None, maxval = None, fg_color = 53, bg_color = 234, flip = False):
    return scalar_to_text_nb(scalars, minval = minval, maxval = maxval, fg_color = fg_color, bg_color = bg_color, bit_depth = 8, flip = flip)

def scalar_to_text_16b(scalars, minval = None, maxval = None, fg_color = 53, bg_color = 234, flip = False):
    return scalar_to_text_nb(scalars, minval = minval, maxval = maxval, fg_color = fg_color, bg_color = bg_color, bit_depth = 16, flip = flip)

def scalar_to_text_nb(scalars, minval = None, maxval = None, fg_color = 53, bg_color = 234, bit_depth = 24, flip = False, effect = None, add_range = False, **kwargs):
    
    if flip:
        bg, fg = get_fgbg(bg_color, fg_color)
    else:
        fg, bg = get_fgbg(fg_color, bg_color)
    
    add_border = False
    eff = ""
    if effect:
        if effect == "border":
            add_border = True
        else:
            eff += str(effect)
    
    
    base_bit_depth = len(SCALE) - 1
    if not bit_depth % base_bit_depth == 0:
        return ["no"]
    
    nrows = bit_depth // base_bit_depth
    ncols = len(scalars)
    nvals = base_bit_depth * nrows
    
    rows = [[fg+bg+eff] for r in range(nrows)]
    
    bit_ranges = [base_bit_depth*i for i in range(nrows)]
    
    if minval is None:
        minval = min(scalars)
    if maxval is None:
        maxval = max(scalars)
    rng = (maxval - minval)/1
    c = (minval+ maxval)/2
    if rng == 0:
        return [fg+bg+eff + SCALE[-1] + RESET]
    
    for s in scalars:
        sv = int(nvals*((s - c)/rng)) + bit_depth // 2
            
        for row, bit_range in zip(rows, bit_ranges):
            if sv < bit_range:
                sym = SCALE[0]
            elif sv >= bit_range + base_bit_depth:
                sym = SCALE[-1]
            else:
                ssv = sv % base_bit_depth
                sym = SCALE[ssv]
            row.append(sym)
    
    brdc = "\x1b[38;5;6m"
    outstrs= []
    for row in rows[::-1]:
        if add_border:
            row.insert(0, brdc + OTHER.get("right_eighth",""))
            row.append(brdc + SCALE_H[1])
        row.append(RESET)
        outstrs.append("".join(row))
    
    if add_border:
        outstrs.insert(0, " " + SCALE[1]*ncols + " ")
        outstrs.append(f"{brdc} " + OTHER.get("upper_eighth","")*ncols + f" {RESET}")
    
    if add_range:
        ran_fstr = kwargs.get("range_fstr", "0.2f")
        hilo = "⌝⌟"
        hi, lo = list(hilo)
        hi, lo = (lo, hi) if flip else (hi, lo)
        minstr = format(minval, ran_fstr)
        maxstr = format(maxval, ran_fstr)
        outstrs[0] += hi + maxstr
        if bit_depth > 8:
            outstrs[-1] += lo + minstr
    
    return outstrs

def scalar_to_text_mid(scalars, center = None, rng = None, fg_color = 53, bg_color = 234,  effect = None, same_color = False):
    
    bit_depth = 16
    
    bg, fg = get_fgbg(bg_color, fg_color)
    if same_color:
        ibg, ifg = get_fgbg(fg_color, bg_color)
    else:
        ibg, ifg = get_fgbg(bg_color, fg_color - 6)
    
    eff = ""
    if effect:
        eff += str(effect)
    
    base_bit_depth = len(SCALE) - 1
    if not bit_depth % base_bit_depth == 0:
        return ["no"]
    
    nrows = bit_depth // base_bit_depth
    ncols = len(scalars)
    nvals = base_bit_depth * nrows
    
    rows = [[fg+bg+eff],[ifg+ibg+eff]]
    
    bit_ranges = [base_bit_depth*i - bit_depth/2 for i in range(nrows)]
    
    if not center:
        c = 0
    else:
        c = center
    
    if not rng:
        minval = min(scalars)
        maxval  = max(scalars)
        rng = 2*max(abs(minval), abs(maxval))
    minval, maxval = c-rng/2, c+rng/2
    
    neg = False
    for s in scalars:
        sv = int(nvals*((s - c)/rng))
        if sv < 0 and not neg:
            neg = True
        elif sv >= 0 and neg:
            neg = False
        
        for row, bit_range in zip(rows, bit_ranges):
            if sv < bit_range:
                sym = SCALE[0]
            elif sv >= bit_range + base_bit_depth:
                sym = SCALE[-1]
            else:
                ssv = sv % base_bit_depth
                sym = SCALE[ssv]
            row.append(sym)
    
    outstrs= []
    for row in rows[::-1]:
        row.append(RESET)
        outstrs.append("".join(row))
        
    return outstrs


def quantize(data, bit_depth, maxval=None, minval=None, mid = False):
    if not maxval:
        maxval = max(data)
    if not minval:
        minval = min(data)
    rng = maxval-minval
    c = (minval+maxval)/2
    off = 0 if mid else 0.5
    if rng == 0.0:
        return 0.0
    return [int(bit_depth * (((d-c)/rng) + off)) for d in data]


def print_system(circ:'Circuit0', show_synapsing = False):
    
    headers = ["Ind","Type","Layer", "Bias", "Act", "Pre/Post Wgts L2", "Number Synapsing"]
    if show_synapsing:
        headers += ["Presynaptic", "Postsynaptic"]
    
    rows = []
    for n in circ.neurons:
        nrtype = ""
        if n.type:
            nrtype = n.type
        elif n.inhib:
            nrtype = "Inh."
        else:
            nrtype = "Exc."
        preweights = [syn.weight for syn in n.synapses.get("pre")]
        prewgts_l2 = np.sqrt(np.sum(np.power(preweights, 2)))
        postweights = [syn.weight for syn in n.synapses.get("post")]
        postwgts_l2 = np.sqrt(np.sum(np.power(postweights, 2)))
        rows.append([n.ind, nrtype, n.layer, n.bias, n.act.act_fn,f"{prewgts_l2:0.3f}/{postwgts_l2:0.3f}", f"{n.num_presynaptic}/{n.num_postsynaptic}"])
        if show_synapsing:
            rows[-1].append(",".join([str(nn.ind) for nn in n.get_presynaptic_neurons()]))
            rows[-1].append(",".join([str(nn.ind) for nn in n.get_postsynaptic_neurons()]))
    
    print(tabulate(rows, headers = headers, floatfmt = "0.3f", tablefmt = "rounded_grid"))
    print()

def print_states(circ:'Circuit0', min_t = 0, max_t = -1):
    
    max_t = min(circ.tstep, max_t)
    if max_t < 0:
        max_t = circ.tstep
    
    headers = ["t"]
    rows = []
    for nr in circ.neurons:
        neur_str = ""
        if type(nr).__name__=="InputNeuron":
            neur_str = "Input"
        elif nr.inhib:
            neur_str = "Inh."
        else:
            neur_str = "Exc."
        
        headers.append(f"N{nr.ind} ({neur_str})")
    
    for t in range(min_t, max_t):
        row = [t]
        states = circ.get_neuron_states(t=t)
        for ni in range(len(states)):
            col = Colors.SPIKE if states[ni].is_spiking else Colors.DARK
            row.append(f"{col}{states[ni].activation}{Colors.RESET}")
        rows.append(row)
    print(tabulate(rows, headers = headers, floatfmt = "0.3f"))
    print()

def show_states(circ:'Circuit0', max_t = -1):
    
    nnrs = circ.num_neurons
    
    header = ["t"] + [n for n in range(nnrs)]
    rows = []
    if max_t < 1:
        max_t = circ.tstep
        
    for t in range(0, max_t):
        row = []
        row.append(t)
        states = circ.get_neuron_states(t)
        
        for nst in states:
            sym = SCALE[-1] if nst.is_spiking else SCALE[0]
            row.append(f"{sym}{sym} ")
        rows.append(row)
    
    print(tabulate(rows, headers = header, tablefmt = "plain", maxcolwidths = 2))

def show_activations_8b(circ:'Circuit0', max_t = -1, minval = -1, maxval = 1,  raw = False, show_stats = False):
    
    bit_depth = 8
    
    marg = "L{layer}-N{ind} "
    marg_frm = ">16"
    stat_frm = ("{:>8.3f}"*4)
    nnrs = circ.num_neurons
    
    if max_t < 1:
        max_t = circ.tstep
    
    header = [format(" ", marg_frm)] + ["." for n in range(max_t)]
    if show_stats:
        header += ("{:>8}"*4).format("Mean", "SD","Min","Max")
    all_rows = []
    curr_layer = "Input"
    for nn in range(nnrs):
        nr = circ[nn]
        
        if nr.layer != curr_layer:
            curr_layer = nr.layer
            all_rows.append(format("", marg_frm) + "-"*max_t)
        
        nnstates = [nr.get_state(t) for t in range(0, max_t)]        
        if raw:
            acts = [nst.raw_activation for nst in nnstates]
        else:
            acts = [nst.activation for nst in nnstates]
            
        rng = max(acts) - min(acts)
        if rng == 0.0:
            maxval = acts[0] + 1
            minval = acts[0] - 1
            
        acts_qt = quantize(acts, bit_depth, maxval=maxval, minval=minval)
        spks = [nst.is_spiking for nst in nnstates]
        
        rmarg = marg.format(layer = nr.layer, ind = nr.ind)
        pre = post = ""
        if nr.inhib:
            pre= Colors.NEG_TEXT
            post = Colors.RESET
        row=[pre+format(rmarg,marg_frm)+post]
            
        for actq, spk in zip(acts_qt, spks):
            _, fgc = get_act_color(1, spk, nr.inhib)
            
            sv = min(bit_depth-1, max(actq, 0))
            sym = SCALE[sv]
            row.append(f"{fgc}{sym}{RESET}")
        mean, sd, minn, maxx, n = nr.get_stats(circ.tstep)
        suffix = ""
        if show_stats:
            suffix = stat_frm.format(mean, sd, minn, maxx)
        all_rows.append(row + [suffix])
        
    print("".join(header))
    for row in all_rows:
        print("".join(row))
    print()

def show_activations(circ:'Circuit0', max_t = -1, minval = -1, maxval = 1, raw = False):
    
    base_bit_depth = 8
    bit_depth = 16
    
    nnrs = circ.num_neurons
    
    if max_t < 1:
        max_t = circ.tstep
    
    header = ["n "] + [str(n) if n %5==0 else "." for n in range(max_t)]
    all_rows = []
    for nn in range(nnrs):
        nr = circ[nn]
        
        nnstates = [nr.get_state(t) for t in range(0, max_t)]
        if raw:
            acts = [nst.raw_activation for nst in nnstates]
        else:
            acts = [nst.activation for nst in nnstates]
        acts_qt = quantize(acts, bit_depth, maxval=maxval, minval=minval, mid = True)
        spks = [nst.is_spiking for nst in nnstates]
        
        rows = []
        for nb in range(bit_depth//base_bit_depth):
            row = []
            if nb==0:
                row.append("  ")
            else:
                row.append(format(nn,"<2"))
                
            bit_range = base_bit_depth * nb - bit_depth/2
            for actq, spk in zip(acts_qt, spks):
                bgc, fgc = get_act_color(nb, spk)
                
                if actq < bit_range:
                    sym = SCALE[0]
                elif actq >= bit_range + base_bit_depth:
                    sym = SCALE[-1]
                else:
                    ssv = actq % base_bit_depth
                    sym = SCALE[ssv]
                
                row.append(f"{bgc}{fgc}{sym}{RESET}")
            rows.append("".join(row))
        all_rows.extend(reversed(rows))
        
    print("".join(header))
    for row in all_rows:
        print("".join(row))
    print()


def show_mean_dist(circuit):
    
    means = []
    for nr in circuit:
        states = [nr.get_state(t) for t in range(circuit.tstep)]
        acts = [s.activation * s.is_spiking for s in states]
        means.append(np.mean(acts))
    
    means_srt = list(sorted(means, reverse = True))
    
    sctxt = scalar_to_text_nb(means_srt, fg_color = 28, add_range = True)
    print("Mean distribution")
    for r in sctxt:
        print(r)
    print()
    
def show_cov_dist(circuit):

    covs = []
    for ni in range(len(circuit.neurons)):
        nri = circuit[ni]
        istates = [nri.get_state(t) for t in range(circuit.tstep)]
        iacts = [s.activation * s.is_spiking for s in istates]
        for nj in range(ni+1, len(circuit.neurons)):
            nrj = circuit[nj]
            if not nri.layer == nrj.layer:
                continue
            
            jstates = [nrj.get_state(t) for t in range(circuit.tstep)]
            jacts = [s.activation * s.is_spiking for s in jstates]
            cov_mat = np.cov(iacts, jacts)
            if cov_mat[0,0] != 0:
                cv = cov_mat[0,1] / cov_mat[0,0]
            else:
                cv=  0.0
            covs.append(cv)
    
    covs_srt = list(sorted(covs, reverse = True))
    
    sctxt = scalar_to_text_nb(covs_srt, fg_color = 125, add_range = True)
    print("Covariance distribution")
    for r in sctxt:
        print(r)
    print()


def plot_net_activity(circ:'Circuit0', max_t = -1):
        
    if max_t < 1:
        max_t = circ.tstep
    
    header = ["L "] + ["." for n in range(max_t)]
    
    max_depth = max([n.depth for n in circ.neurons])
    max_act = 0
    
    layerdata = []
    for i in range(max_depth):
        ns = circ.get_layer(i)
        
        row = []
        for t in range(max_t):
            act_sum = sum([abs(n.activation(t)) for n in ns])
            row.append(act_sum)
        if max(row) > max_act:
            max_act = max(row)
    
    print(f"max act: {max_act}")
    print("".join(header))
    for i in range(max_depth):
        maxval = None
        sctxt = scalar_to_text_8b(row, minval = 0, maxval = maxval, fg_color = Colors.SPIKE)
        layerdata.append(sctxt[0])
        print(str(i), layerdata[i])
    

def get_act_color(row_ind, spiking, inhib=False):
    if row_ind == 1:
        bgc = Colors.BG_HIDE
        if spiking:
            fgc = Colors.NEG_SPIKE if inhib else Colors.SPIKE
        else:
            fgc = Colors.DARK
    elif row_ind == 0:
        fgc = Colors.HIDE
        if spiking:
            bgc = Colors.BG_NEG_SPIKE if inhib else Colors.BG_SPIKE
        else:
            bgc = Colors.BG_DARK
    else:
        bgc = fgc = None
    return bgc, fgc

def show_colors(addl_str = ""):
    
    for c in range(0, 256):
        cstr = f"\x1b[38;5;{c}m"
        bgcstr = f"\x1b[48;5;{c}m" + Colors.HIDE
        print(f"{c} {cstr} this is color {c} with a bar | {addl_str} | {SCALE}{SCALE[::-1]}{RESET} {bgcstr}{SCALE}{SCALE[::-1]}{Colors.RESET}")
    
    
def get_fgbg(fg_color, bg_color):
    if isinstance(fg_color, int):
        fg = f"\x1b[38;5;{fg_color}m"
    else:
        fg = fg_color
    if isinstance(bg_color, int):
        bg = f"\x1b[48;5;{bg_color}m"
    else:
        bg = bg_color
    return fg, bg