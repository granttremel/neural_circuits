
import math

import numpy as np

from . import draw




def make_step_waveform(f, d_on, d_tot, ph, A=1.0, num_cycles = 1):
    
    ns = int(d_tot * num_cycles)
    dt = 1 / f / d_tot
    
    
    if ph == 0:
        ph_d = 0
    else:
        ph_d = dt / (f*ph)
    
    wf = np.zeros(ns)
    
    for nt in range(ns):
        if (nt - ph_d) % d_tot < d_on:
            wf[nt] = A
    
    return wf, dt

def make_step_waveform2(f, d, ph, dt, t_max,  A=1.0):
    
    ns = t_max / dt
    
    tp = 1/f
    num_cycles = t_max / tp
    d_tot = ns / num_cycles
    d_on = d*d_tot
    
    ns = round(ns)
    num_cycles = round(num_cycles)
    d_tot = round(d_tot)
    d_on = max(1, round(d_on))
    
    ph_d = ph / f / dt
    
    wf = np.zeros(ns)
    
    for nt in range(ns):
        if (nt - ph_d) % d_tot < d_on:
            wf[nt] = A
    
    return wf, dt

def compare_waveforms(w1, w2, dt = None, t_max = None):
    
    f1, d1, ph1, = w1
    f2, d2, ph2 = w2
    
    A1 = A2 = 1.0
    
    # dt = 1/(f1+f2)/2 # least time res for frequency
    ton1 = d1/f1
    toff1 = (1-d1) / f1
    
    ton2 = d2/f2
    toff2 = (1-d2) / f2
    
    if not dt or not t_max:
        ns = 256
        dt = min(ton1, ton2, toff1, toff2)
        t_max = ns * dt
    else:
        ns = int(t_max / dt)
    
    nc1 = t_max * f1
    nc2 = t_max * f2
    # print(f"num cycles: {nc1:0.1f} and {nc2:0.1f}")
    
    wf1, _ = make_step_waveform2(f1, d1, ph1, dt, t_max, A = A1)
    wf2, _ = make_step_waveform2(f2, d2, ph2, dt, t_max, A = A2)
    
    sc = 0
    for ww1, ww2 in zip(wf1, wf2):    
        sc += ww1 + ww2
    sc /= ns
    sc /= (A1 + A2)
    
    return sc, dt, t_max

def sum_waveforms(wf1, wf2, thresh = None):
    summ= [v1+v2 for v1, v2 in zip(wf1, wf2)]
    if thresh:
        summ = [min(v,thresh) for v in summ]
    return summ
    
def diff_waveforms(wf1, wf2, thresh = None, rect = False):
    diff= [v1-v2 for v1, v2 in zip(wf1, wf2)]
    if rect:
        diff = [max(v,0) for v in diff]
    return diff

def extract_waveform_info(wf, dt):
    
    ns = len(wf)
    t_max = dt*ns
    
    ons = []
    summ = 0
    npks = 0
    
    vlast = 0
    for i in range(ns):
        v = wf[i]
        summ += v
        if v > 0:
            npks += 1
            if vlast == 0.0:
                ons.append(i)
        vlast = v
    
    if not ons:
        return -1, -1, -1
    dons = [on1 - on0 for on1, on0 in zip(ons[1:], ons[:-1])]
    f_avg = 1/np.mean(dons)/dt
    
    d_avg = npks / ns
    
    
    phis = [n/f_avg - ons[n]*dt for n in range(len(ons))]
    phi_avg = -np.mean(phis)*f_avg
    
    return f_avg, d_avg, phi_avg
    
def validate_waveform(w, t_refr):
    
    f, d, ph = w
    
    tp = 1/f
    t_on = tp*d
    t_off = tp*(1-d)
    
    if t_off < t_refr:
        t_off = t_refr
        d = t_on / (t_on + t_off)
    
    return (f, d, ph)

def show_waveform(wf, bit_depth = 16, lbl = ""):
    
    add_range = False
    if bit_depth > 8:
        add_range = True
    
    marg = "{:<6}" if lbl else "{}"
    lbls = [lbl] + [""]*(bit_depth//8 - 1)
    
    sctxt = draw.scalar_to_text_nb(wf, bit_depth = bit_depth, add_range = add_range, fg_color = draw.Colors.SPIKE, bg_color = 232)
    
    for lbl,r in zip(lbls,sctxt):
        print(marg.format(lbl), r)
    print()
    
def fourier_transform(wf):
    ft = np.fft.fftshift(np.fft.fft(wf)) # f = 0 in center
    return np.real(ft), np.imag(ft)

def show_ft(wf, bit_depth = 16, polar = False, show_time = False, **kwargs):
    
    if show_time:
        show_waveform(wf, lbl = "Time")
    
    real, imag = fourier_transform(wf)
    if polar:
        show_ft_polar(real, imag, bit_depth = bit_depth, **kwargs)
    else:
        show_ft_reim(real, imag, bit_depth = bit_depth, **kwargs)

def show_ft_reim(real, imag, bit_depth = 16):
    num_rows = bit_depth // 8

    data1 = np.real(real)
    data2 = np.imag(1j*imag)
    minval1 = minval2 = min(min(real), min(imag))
    maxval1 = maxval2 = max(max(real[:len(real)//2]), max(real[len(real)//2+1:]), max(imag))
    c1 = 0
    r1 = 2*max(maxval1, abs(minval1))
    c2 = 0
    r2 = 2*max(maxval2, abs(minval2))
        
    # add_range = False
    # if bit_depth > 8:
    #     add_range = True
    
    marg = "{:<6}"
    
    # sctxt1 = draw.scalar_to_text_nb(data1, bit_depth = bit_depth, add_range = add_range, minval = minval1, maxval = maxval1)
    sctxt1 = draw.scalar_to_text_mid(data1, center = c1, rng = r1)
    lbl1s = ["Real"] + (num_rows - 1) * [" "]
    
    for lbl,r in zip(lbl1s,sctxt1):
        print(marg.format(lbl), r)
    
    
    # sctxt2 = draw.scalar_to_text_nb(data2, bit_depth = bit_depth, add_range = add_range, minval = minval2, maxval = maxval2)
    sctxt2 = draw.scalar_to_text_mid(data2, center = c2, rng = r2)
    lbl2s = ["Imag"] + (num_rows - 1) * [" "]
    
    for lbl,r in zip(lbl2s,sctxt2):
        print(marg.format(lbl), r)
    

def show_ft_polar(real, imag, bit_depth = 16):
    
    num_rows = bit_depth // 8
    lbl1 = "Mag"
    lbl2 = "Arg"
    
    pwr = np.sum(np.power(real,2)+np.power(imag,2))
    data1 = np.sqrt(np.power(real,2)+np.power(imag,2))
    # data1[len(data1)//2] = .01
    data2 = np.real(np.angle(real+1j*imag))
    minval1 = 0
    maxval1 = max(data1[:len(data1)//2])*1.1
    
    c2 = 0
    r2 = 2*np.pi
        
    add_range = False
    if bit_depth > 8:
        add_range = True
    
    marg = "{:<6}"
    
    sctxt1 = draw.scalar_to_text_nb(data1, bit_depth = bit_depth, add_range = add_range, minval = minval1, maxval = maxval1, fg_color = draw.Colors.SPIKE, bg_color = 232)
    lbl1s = [lbl1] + (num_rows - 1) * [" "]
    
    for lbl,r in zip(lbl1s,sctxt1):
        print(marg.format(lbl), r)
    print()
    
    sctxt2 = draw.scalar_to_text_mid(data2, center = c2, rng = r2, fg_color = 56, bg_color = 232, same_color = True)
    lbl2s = [lbl2] + (num_rows - 1) * [" "]
    
    for lbl,r in zip(lbl2s,sctxt2):
        print(marg.format(lbl), r)
    print()

def unwrap_phase(phase_data):
    
    n = len(phase_data)
    c = len(phase_data)//2
    deltas = np.zeros(n)
    
    jump_thresh = np.pi/2
    
    jump_ct = 0
    last_p = 0
    last_dp = 0
    for i in range(c, n):
        p = phase_data[i]
        dp = p - last_p
        
        if p*last_p < 0 and abs(dp) > jump_thresh and last_dp < jump_thresh:
            if p>0:
                jumpct += 1
            elif p < 0:
                jumpct -=1
        
        deltas[i] = jumpct*np.pi*2
        
        
        
        pass
    
    
    pass
