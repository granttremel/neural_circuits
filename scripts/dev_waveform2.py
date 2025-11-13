
import fractions
import math

import numpy as np
from tabulate import tabulate
from neural_circuits import waveform as wvf, draw
from neural_circuits.waveform import Waveform

import random


def make_waveform2(f, d, ph, dt, t_max, **kwargs):
    
    wf = wvf.Waveform(f, d, ph)
    wfd = wf.make_step_waveform2(f, d,  ph, dt, t_max, **kwargs)
    
    wf.show(dt, t_max)
    
    return wfd

def get_random_waveform(fmin = 10, fmax = 100, dmin = 0.1, dmax = 0.9, phmin = 0, phmax = 1, Amin = 1.0, Amax = 1.0):
    f1 = random.random() * (fmax - fmin) + fmin
    # d_tot1 = random.randint(5, 20)
    # d_on1 = random.randint(1, d_tot1-1)
    d1 = random.random()*(dmax - dmin) + dmin
    ph1 = random.random()*(phmax - phmin) + phmin
    A = random.random()*(Amax - Amin) + Amin
    wvf = Waveform(f1, d1, ph1, A = A)
    return wvf
    

def test_waveforms(num_tests = 20):
    
    for n in range(num_tests):
        
        wf = get_random_waveform()
        f, d, ph = wf
        
        # d_tot = random.randint(3, 20)
        # d_on = int(d*d_tot)
        # d = d_on / d_tot
        d_on, d_off, d_tot = wf.ds
        
        nc = random.randint(2,6)
        
        dt = 1/f/d_tot
        t_max = nc / f
        
        print(f"*** params: f: {f:0.3f}, d: {d:0.3f}, d_tot: {d_tot}, nc:{nc}, dt: {dt:0.3f}, t_max: {t_max:0.3f} ***")
        # print("v1")
        
        # wf, dt = make_waveform(f, d_on, d_tot, ph, num_cycles = nc)
        # print()
        
        print("v2")
        # wf2, dt2 = make_waveform2(f, d, ph, dt, t_max)
        wfd = wf.make_step(dt, t_max)
        print()
        
        print(f"dt1: {dt:0.3f}")
        # print(f"************ waveforms equal: {all(wf==wf2)} **************")
        print()
        

def test_compare_waveforms(num_tests =20):
    
    for n in range(num_tests):
        
        print(f"***** test {n} *****")
        
        f1 = random.random() * 99 + 1
        # d_tot1 = random.randint(5, 20)
        
        d_on1 = random.randint(1, Waveform._duty_res-1)
        d1 = d_on1 / Waveform._duty_res
        ph1 = random.random()
        wf1 = Waveform(f1, d1, ph1)
        print(f"*** w1: f: {f1:0.3f}, d: {d1:0.3f}, ph: {ph1:0.3f} ***")
        
        f2 = random.random() * 99 + 1
        d_on2 = random.randint(1, Waveform._duty_res-1)
        d2 = d_on2 / Waveform._duty_res
        ph2 = random.random()
        wf2 = Waveform(f2, d2, ph2)
        print(f"*** w2: f: {f2:0.3f}, d: {d2:0.3f}, ph: {ph2:0.3f} ***")
        print()
        
        sc, dt, t_max = wf1.compare(wf2)
        # sc, dt, t_max = wvf.compare_waveforms(w1, w2)
        ns = int(t_max / dt)
        
        print(f"ns: {ns}, dt: {dt:0.3f}, t_max: {t_max:0.3f}")
        print()
        
        # wf1, _ = wvf.make_step_waveform2(f1, d1, ph1, dt, t_max)
        # wf2, _ = wvf.make_step_waveform2(f2, d2, ph2, dt, t_max)
        # wfd1 = wf1.make_step(dt, t_max)
        # wfd2 = wf2.make_step(dt, t_max)
        
        print(len(wf1), len(wf2))
        print(sum(wf1), sum(wf2))
        
        print(f"Score = {sc:0.1%}")
        wfd1 = wf1.show(dt, t_max)
        wfd2 = wf2.show(dt, t_max)
        # wvf.show_waveform(wf1, bit_depth = 16)
        # wvf.show_waveform(wf2, bit_depth = 16)
        print()


def test_ips(num_tests = 5):
    
    wf1 = get_random_waveform()
    f1, d1, ph1 = wf1.to_tuple()
        
    
    ph1 = 0
    d1 = 0.2
    
    wf1 = Waveform(f1, d1, ph1)
    
    dt = min(d1, 1-d1)/f1/2
    t_max = 16/f1
    
    # wf1, _ = wvf.make_step_waveform2(f1, d1, ph1, dt, t_max)
    wfd1 = wf1.make_step(dt, t_max)
    fratio = np.exp(np.linspace(np.log(0.5), np.log(2), num_tests))
    
    rows = []
    
    for n in range(num_tests):
        
        v = fratio[n]
        print(f"***** test {n}, value {v:0.2f} *****")
        
        wf2 = get_random_waveform()
        # f2,d2,ph2 = w
        f2,d2,ph2 = wf2.to_tuple()
        
        # w2 = w1
        # f2 = v*f1
        # d2 = d1
        # ph2 = ph1
        
        wf2 = Waveform(f2, d2, ph2)
        
        print(f"*** w2: f: {f2:0.3f}, d: {d2:0.3f}, ph: {ph2:0.3f} ***")
        print()
        
        wfd1 = wf1.show(dt =dt, t_max = t_max, bit_depth = 16, lbl = "w1")
        # wf2, _ = wvf.make_step_waveform2(*wf2, dt, t_max)
        wfd2 = wf2.show(dt =dt, t_max = t_max, bit_depth = 16, lbl = "w2")
        
        wfd_sum = wf1.sum_data(wf2, thresh = 1.0)
        Waveform.show_data(wfd_sum, bit_depth = 16, lbl = "w1+w2")
        wfd_diff =  wf1.diff_data(wf2, rect = True)
        Waveform.show_data(wfd_diff, bit_depth = 16, lbl = "w1-w2")
        wfd_diff2 =  wf2.diff_data(wf1, rect = True)
        Waveform.show_data(wfd_diff2, bit_depth = 16, lbl = "w2-w1")
        
        sc, _, _ = wf1.compare(wf2, dt = dt, t_max = t_max)
        
        # sc = sum([v1*v2 for v1,v2 in zip(wf1, wf2)]) / len(wf1)
        
        sc_scaled = sc / max(d1, d2)
        print(f"Score = {sc:0.1%}")
        print()
        rows.append([f1, f2,d1, d2, ph1, ph2, sc, sc_scaled])
    
    print(tabulate(rows, headers = ["f1", "f2","d1","d2","phi1","phi2", "Score", "D scaled"]))
    p_on_naive = d1*d2
    print(f"Naive P(w1 and w2): {p_on_naive:0.3f}")
    
def test_alg(num_tests = 5, show_spectrum = False):
   
    ind_lbls = ["f","d","phi"]
    wf_headers = ["w1","w2","w1+w2","w1-w2","w2-w1", "direct sum", "direct diff"] 
    score_data = []
    
    for n in range(num_tests):
        
        print(f"***** test {n} *****")
        
        wf1 = get_random_waveform()
        f1, d1, ph1 = wf1.to_tuple()
        
        wf2 = get_random_waveform()
        
        f2,d2,ph2 = wf2.to_tuple()
        
        f2 = f1
        d1 = d2 = .1
        
        wf1 = Waveform(f1, d1, ph1)
        wf2 = Waveform(f2, d2, ph2)
        
        ns = 256
        dt = min(min(d1, 1-d1)/f1/2,min(d2, 1-d2)/f2/2)
        t_max = ns*dt
        
        print(f"*** w1: {repr(wf1)} ***")
        wfd1 = wf1.make_step(dt, t_max)
        Waveform.show_data(wfd1, bit_depth = 16, lbl = "w1")
        
        print(f"*** w2: {repr(wf2)} ***")
        wfd2 = wf2.make_step(dt, t_max)
        Waveform.show_data(wfd2, bit_depth = 16, lbl = "w2")
        
        wfsum = wf1.sum_data(wf2, dt, t_max, thresh = 1.0)
        Waveform.show_data(wfsum, bit_depth = 16, lbl = "w1+w2")
        wres1 = Waveform.extract(wfsum, dt)
        if wres1:
            wres1.show(dt, t_max, lbl = "w1+w2?")
        
        wfdiff = wf1.diff_data(wf2, dt, t_max, rect = True)
        Waveform.show_data(wfdiff, bit_depth = 16, lbl = "w1-w2")
        wres2 = Waveform.extract(wfdiff, dt)
        if wres2:
            wres2.show(dt, t_max, lbl = "w1-w2?")
        
        wfdiff2 = wf2.diff_data(wf1, dt, t_max, rect = True)
        Waveform.show_data(wfdiff2, bit_depth = 16, lbl = "w2-w1")
        wres3 = Waveform.extract(wfdiff2, dt)
        if wres3:
            wres3.show(dt, t_max, lbl = "w2-w1?")
        
        if show_spectrum:
            print("w1 spectrum")
            wf1.show_spectrum(dt, t_max, polar = True)
            print()
            
            print("w2 spectrum")
            wf2.show_spectrum(dt, t_max, polar = True)
            print()
            
            print("w1+w2 spectrum")
            wres1.show_spectrum(dt, t_max, polar = True)
            print()
            
            print("w1-w2 spectrum")
            wres2.show_spectrum(dt, t_max, polar = True)
            print()
            
            print("w2-w1 spectrum")
            wres3.show_spectrum(dt, t_max, polar = True)
            print()
            
        wf_direct_sum = Waveform(*[a+b for a, b in zip(wf1.to_tuple(), wf2.to_tuple())])
        wf_direct_diff = Waveform(*[abs(a-b) for a, b in zip(wf1.to_tuple(), wf2.to_tuple())])
        
        wf_data =[]
        for i in range(3):
            row = [ind_lbls[i]]
            for wff in [wf1, wf2, wres1, wres2, wres3, wf_direct_sum, wf_direct_diff]:
                row.append(wff.to_tuple()[i])
            wf_data.append(row)
        
        print(tabulate(wf_data, headers = wf_headers))
        print()
        
        sc, _, _ = wf1.compare(wf2, dt = dt, t_max = t_max)
        
        sc_scaled = sc / max(d1, d2)
        print(f"Score = {sc:0.1%}")
        print()
        score_data.append([f1, f2,d1, d2, ph1, ph2, sc, sc_scaled])
    
    print(tabulate(score_data, headers = ["f1", "f2","d1","d2","phi1","phi2", "Score", "D scaled"]))
    # p_on_naive = d1*d2
    # print(f"Naive P(w1 and w2): {p_on_naive:0.3f}")

def compare_phases(wf1, wf2, min_phase, max_phase, num_tests, dt, t_max):
    
    phases = np.linspace(min_phase, max_phase, num_tests)
    
    data= []
    
    wf1.show(dt, t_max)
    wf2.show(dt, t_max)
    
    for n in range(num_tests):
        
        wf2.phi = phases[n]
        
        s, _, _ = wf1.compare(wf2, dt = dt, t_max = t_max)
        
        data.append(s)
    
    print(f"phase responses for {repr(wf1)} and {repr(wf2)} with dt={dt:0.3f}, t_max={t_max:0.3f}")
    sctxt = draw.scalar_to_text_nb(data, minval = 0, add_range = True)
    sctxt, _ = draw.add_ruler(sctxt, xmin = phases[0], xmax = phases[-1], minor_ticks = 0, fstr = "0.2f")
    for r in sctxt:
        print(r)
        
def compare_freqs(wf1, wf2, min_freq, max_freq, num_tests, dt, t_max, show_every = 0):
    
    freqs = np.linspace(min_freq, max_freq, num_tests)
    
    data= []
    
    wf1.show(dt, t_max)
    
    
    for n in range(num_tests):
        
        wf2.f = freqs[n]
        if show_every and n%show_every == 0:
            wf2.show(dt, t_max)
        
        s, _, _ = wf1.compare(wf2, dt = dt, t_max = t_max)
        
        data.append(s)
    
    print(f"freq responses for {repr(wf1)} and {repr(wf2)} with dt={dt:0.3f}, t_max={t_max:0.3f}")
    sctxt = draw.scalar_to_text_nb(data, minval = 0, add_range = True)
    sctxt, _ = draw.add_ruler(sctxt, xmin = freqs[0], xmax = freqs[-1], minor_ticks = 0, fstr = "0.2f")
    for r in sctxt:
        print(r)

def compare_random(min_freq, max_freq, min_duty, max_duty, min_phase, max_phase, num_tests, num_cycles = 3, show_every = 0):
    
    ns = 128
    # wf_data = []
    all_data = [[], [],[],[]]
    
    for n in range(num_tests):
        wvs = []
        for i in range(2):
            f = random.random()*(max_freq - min_freq) + min_freq
            d = random.random()*(max_duty - min_duty) + min_duty
            p = random.random()*(max_phase - min_phase) + min_phase
            wvs.append(Waveform(f, d, p))
        
        t_max = max(wvs[0].tp, wvs[1].tp) * num_cycles
        dt = t_max / ns
        
        if show_every and n%show_every == 0:
            print(f"round {n} with {repr(wvs[0])}, {repr(wvs[1])}")
            wvs[0].show(dt, t_max)
            wvs[1].show(dt, t_max)
        
        wf1d = wvs[0].make_step(dt, t_max)
        wf2d = wvs[1].make_step(dt, t_max)
        
        s = Waveform._compare(wf1d, wf2d)
        
        sumres = wvs[0].sum_data(wvs[1], dt, t_max)
        sum_score = Waveform._compare(wf1d, sumres)
        
        diffres = wvs[0].diff_data(wvs[1], dt, t_max, rect = True)
        diff_score = Waveform._compare(wf1d, diffres)
        
        row = []
        row.extend(wvs[0].to_tuple())
        row.extend(wvs[1].to_tuple())
        all_data[0].append(row)
        
        all_data[1].append(s)
        all_data[2].append(sum_score)
        all_data[3].append(diff_score)
        
    # scores = [d[0] for d in score_data]
    # sdscores = [d[1] for d in score_data]

    return tuple(all_data)

def show_random_results(scores, num_bins, data_label = "Score"):

    minval = min(0.0, min(scores))
    maxval = max(1.0, max(scores))
    
    print(f"{data_label} stats: mean = {np.mean(scores):0.3f}, sd = {np.std(scores):0.3f}, min = {np.min(scores):0.3f}, max = {np.max(scores):0.3f}")
    draw.plot_scalar_hist(scores, num_bins, (minval, maxval), fg_color = draw.Colors.SPIKE)
    
def test_class():
    """
    phi: kind obvs, adds linear phase ramp to time shift spectrum
    d: causes broadening of spectrum (convolution w/ square = mult with sinc?)
    dt: finer dt means greater f_max 
    t_max: greater value causes more zeros filling in..?
    """
    wf =  get_random_waveform()
    
    for d in range(1, 8):
        # for ns in [32, 64, 128]:
        for ncycles in [1, 2, 3]:
            dd = d / 8
            wf._d = dd*wf._duty_res
            print(repr(wf))
            
            t_max = wf.tp * ncycles
            ns = 128
            dt = t_max / ns
            
            wf.show_spectrum(dt, t_max, show_time = True)
            

def test_val_time(vary_duty = False, monte_carlo = True, num_tests = 50):
    
    if vary_duty:
        f = 100
        phi = 0.2
        dt = 0.05*1/f
        t_max = dt*128
        
        for d in [0.05, 0.5, 0.95]:
            
            wf = Waveform(f, d, phi)
            wfd = wf.make_step(dt, t_max)
            
            print(f"d = {d:0.2f}")
            Waveform.show_data(wfd, lbl = "no val")
            
            wfdv, _ =Waveform.validate_min_time(wfd, dt)
            Waveform.show_data(wfdv, lbl = "val")
    
    if monte_carlo:
        for n in range(num_tests):
            
            wf1 = get_random_waveform(fmin = 1, fmax = 200, dmin = 0.05, dmax = 0.95)
            wf2 = get_random_waveform(fmin = 1, fmax = 200, dmin = 0.05, dmax = 0.95)
            
            tpmax = max(wf1.tp, wf2.tp)
            ns = 128
            nc = 1
            dt = nc*tpmax/ns
            t_max = dt*ns
            
            diff = Waveform.diff_data(wf1, wf2, dt, t_max, rect = True, thresh = 1.0)
            
            wfdv, res =Waveform.validate_min_time(diff, dt)
            if res:
                wfdv2, res2 = Waveform.validate_min_time(wfdv, dt)
                if res2:
                    print("result changed upon repeat validation!")
                    print(f"f1 = {wf1.f:0.2f}, f2 = {wf2.f:0.2f}")
                    Waveform.show_data(diff, lbl = "no val", add_ruler = True, t_max = t_max)
                    Waveform.show_data(wfdv, lbl = "val", add_ruler = True, t_max = t_max)
                    Waveform.show_data(wfdv2, lbl = "val2", add_ruler = True, t_max = t_max)
            if n%(num_tests//10) == 0:
                print(f"completed test {n}")

def run_compare_tests():
    f1 = 50
    f2 = 50
    d1 = 1/8
    d2 = 3/8
    phi1 = 0
    phi2 = .2
    
    wf1 = Waveform(f1, d1, phi1)
    wf2 = Waveform(f2, d2, phi2)
    
    ns = 128
    t_max = wf1.tp * 3
    dt = t_max / ns
    
    min_phi = -.5
    max_phi = 0.5
    
    min_freq = 25
    max_freq = 75
    
    num_tests = 5
    
    # compare_phases(wf1, wf2, min_phi, max_phi, num_tests, dt, t_max)
    # compare_freqs(wf1, wf2, min_freq, max_freq, num_tests, dt, t_max, show_every = 2)
    
    pass

def run_monte_carlo():
    
    min_freq = 10
    max_freq = 100
    
    min_duty = 0.1
    max_duty = 0.9
    
    min_phi = -0.5
    max_phi = 0.5
    
    ns = 128
    t_max = wf1.tp * 3
    dt = t_max / ns
    
    num_tests =10000
    # show_every = num_tests // 5
    show_every = 0
    
    wvdata, scores, sum_scores, diff_scores = compare_random(min_freq, max_freq, min_duty, max_duty, min_phi, max_phi, num_tests, num_cycles = 3, show_every = show_every)
    show_random_results(scores, 100)
    show_random_results(sum_scores, 100, data_label = "Sum scores")
    show_random_results(diff_scores, 100, data_label = "Diff scores")

    absdiffs = [(wv,sc, dsc, abs(sc-dsc)) for wv, sc,dsc in zip(wvdata, scores, diff_scores)]
    topk = 5
    absdiffs_topk = sorted(absdiffs, key = lambda tp:-tp[-1])[:topk]

    for wavedata, sc, dsc, absdiff in absdiffs_topk:
        # if abs(sc- dsc) > diff_thresh:
        wp1 = wavedata[:3]
        wp2 = wavedata[3:]
        wf1 = Waveform(*wp1)
        wf2 = Waveform(*wp2)
        print("{}sc={:0.2f}, dsc={:0.2f}, sc-dsc={:0.2f}{}".format(draw.Colors.BOLD,sc, dsc, sc-dsc, draw.Colors.RESET))
        print("f1={:0.2f}, d1={:0.2f}, phi1={:0.2f}".format(*wp1))
        wf1.show(dt, t_max, lbl = "wf1")
        print("f2={:0.2f}, d2={:0.2f}, phi2={:0.2f}".format(*wp2))
        wf2.show(dt, t_max, lbl = "wf2")

def test_multiplexing():
    
    
    
    
    
    
    pass

        
def main():
    
    
    # test_waveforms(num_tests = 5)
    # test_compare_waveforms(num_tests = 3)
    # test_ips()
    # test_alg(num_tests = 1, show_spectrum=True)
    # test_class()
    
    run_monte_carlo()
    
    # test_val_time(num_tests = 200)
    

        
    pass

if __name__=="__main__":
    main()
