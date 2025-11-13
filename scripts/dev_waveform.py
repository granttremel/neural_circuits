
import fractions
import math

import numpy as np
from tabulate import tabulate
from neural_circuits import waveform as wvf
import random


def make_waveform(f, d_on, d_tot, ph, **kwargs):
    
    wf, dt = wvf.make_step_waveform(f, d_on, d_tot, ph, **kwargs)
    
    wvf.show_waveform(wf)
    
    return wf, dt


def make_waveform2(f, d, ph, dt, t_max, **kwargs):
    
    wf, dt = wvf.make_step_waveform2(f, d,  ph, dt, t_max, **kwargs)
    
    wvf.show_waveform(wf)
    
    return wf, dt

def get_random_waveform():
    f1 = random.random() * 99 + 1
    d_tot1 = random.randint(5, 20)
    d_on1 = random.randint(1, d_tot1-1)
    d1 = d_on1 / d_tot1
    ph1 = random.random()
    w1 = (f1, d1, ph1)
    return w1
    

def test_waveforms(num_tests = 20):
    
    for n in range(num_tests):
        
        w = get_random_waveform()
        f, d, ph = w
        
        d_tot = random.randint(3, 20)
        d_on = int(d*d_tot)
        d = d_on / d_tot
        
        nc = random.randint(2,6)
        
        dt = 1/f/d_tot
        t_max = nc / f
        
        print(f"*** params: f: {f:0.3f}, d: {d:0.3f}, d_tot: {d_tot}, nc:{nc}, dt: {dt:0.3f}, t_max: {t_max:0.3f} ***")
        # print("v1")
        
        # wf, dt = make_waveform(f, d_on, d_tot, ph, num_cycles = nc)
        # print()
        
        print("v2")
        wf2, dt2 = make_waveform2(f, d, ph, dt, t_max)
        print()
        
        print(f"dt1: {dt:0.3f}, dt2: {dt2:0.3f}")
        # print(f"************ waveforms equal: {all(wf==wf2)} **************")
        print()


def test_numden(num_tests = 5):
    
    for n in range(num_tests):
        
        f = random.random() * 99 + 1
        d_r = random.random()
        d_tot = random.randint(3, 20)
        d_on = max(1, int(d_r*d_tot))
        d_r = d_on / d_tot
        
        print(f"d_on: {d_on}, d_tot: {d_tot}, d_r: {d_r:0.3f}")
        ph = random.random()
        nc = random.randint(2,6)
        
        dt = 1/f/d_tot
        t_max = nc / f
        
        num, den = wvf.get_duty(f, d_r) 
        d_r2 = num/den
        print(f"d_on2: {num}, d_tot2: {den}, d_r2: {d_r2:0.3f}")
        
        wf2, dt2 = make_waveform2(f, d_r, ph, dt, t_max)
        print()


def test_compare_waveforms(num_tests =20):
    
    for n in range(num_tests):
        
        print(f"***** test {n} *****")
        
        f1 = random.random() * 99 + 1
        d_tot1 = random.randint(5, 20)
        d_on1 = random.randint(1, d_tot1-1)
        d1 = d_on1 / d_tot1
        ph1 = random.random()
        w1 = (f1, d1, ph1)
        print(f"*** w1: f: {f1:0.3f}, d: {d1:0.3f}, ph: {ph1:0.3f} ***")
        
        f2 = random.random() * 99 + 1
        d_tot2 = random.randint(5, 20)
        d_on2 = random.randint(1, d_tot2-1)
        d2 = d_on2 / d_tot2
        ph2 = random.random()
        w2 = (f2, d2, ph2)
        print(f"*** w2: f: {f2:0.3f}, d: {d2:0.3f}, ph: {ph2:0.3f} ***")
        print()
        
        sc, dt, t_max = wvf.compare_waveforms(w1, w2)
        ns = int(t_max / dt)
        
        print(f"ns: {ns}, dt: {dt:0.3f}, t_max: {t_max:0.3f}")
        print()
        
        wf1, _ = wvf.make_step_waveform2(f1, d1, ph1, dt, t_max)
        wf2, _ = wvf.make_step_waveform2(f2, d2, ph2, dt, t_max)
        
        print(len(wf1), len(wf2))
        print(sum(wf1), sum(wf2))
        
        print(f"Score = {sc:0.1%}")
        wvf.show_waveform(wf1, bit_depth = 16)
        wvf.show_waveform(wf2, bit_depth = 16)
        print()
    
def test_fts(duty_tot = 4):
    
    ds = [di / duty_tot for di in range(1, duty_tot)]
    
    wr = get_random_waveform()
    f, _, ph = wr
    ph = 0/duty_tot
    
    dt = min(ds)/f
    t_max = 2/f
    ns = int(t_max/dt)
    if ns > 256:
        ns = 256
        t_max = ns*dt
    
    print("f={:0.3f} Hz, phase={:0.3f} ".format(f, ph))
    print("dt={:0.3f} Hz, t_max={:0.3f}, ns={:}".format(dt, t_max, int(t_max / dt)))
    for d in ds:
        print(f"duty={d:0.3f}")
        
        w = (f, d, ph)
        wf,_ = wvf.make_step_waveform2(*w, dt, t_max)
        
        wvf.show_ft(wf, polar=True, show_time = True)
        

def test_ips(num_tests = 5):
    
    w = get_random_waveform()
    f1, d1, ph1 = w
        
    # d_tot = random.randint(5, 20)
    # d_on = random.randint(1, d_tot-1)
    # d1 = d_on / d_tot
    
    ph1 = 0
    d1 = 0.2
    
    w = (f1, d1, ph1)
    
    dt = min(d1, 1-d1)/f1/2
    t_max = 16/f1
    
    wf1, _ = wvf.make_step_waveform2(f1, d1, ph1, dt, t_max)
    fratio = np.exp(np.linspace(np.log(0.5), np.log(2), num_tests))
    
    rows = []
    
    for n in range(num_tests):
        
        v = fratio[n]
        print(f"***** test {n}, value {v:0.2f} *****")
        
        w2 = get_random_waveform()
        # f2,d2,ph2 = w
        f2,d2,ph2 = w2
        
        # w2 = w1
        # f2 = v*f1
        # d2 = d1
        # ph2 = ph1
        
        w2 = (f2, d2, ph2)
        
        print(f"*** w2: f: {f2:0.3f}, d: {d2:0.3f}, ph: {ph2:0.3f} ***")
        print()
        
        wvf.show_waveform(wf1, bit_depth = 16, lbl = "w1")
        wf2, _ = wvf.make_step_waveform2(*w2, dt, t_max)
        wvf.show_waveform(wf2, bit_depth = 16, lbl = "w2")
        
        wfsum = wvf.sum_waveforms(wf1, wf2, thresh = 1.0)
        wvf.show_waveform(wfsum, bit_depth = 16, lbl = "w1+w2")
        wfdiff = wvf.diff_waveforms(wf1, wf2, rect = True)
        wvf.show_waveform(wfdiff, bit_depth = 16, lbl = "w1-w2")
        wfdiff2 = wvf.diff_waveforms(wf2, wf1, rect = True)
        wvf.show_waveform(wfdiff2, bit_depth = 16, lbl = "w2-w1")
        
        sc, _, _ = wvf.compare_waveforms(w, w2, dt = dt, t_max = t_max)
        
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
        
        w1 = get_random_waveform()
        f1, d1, ph1 = w1
        
        w2 = get_random_waveform()
        
        f2,d2,ph2 = w2
        
        f2 = f1
        d1 = d2 = .1
        
        w1 = f1, d1, ph1
        w2 = f2, d2, ph2
        
        ns = 256
        dt = min(min(d1, 1-d1)/f1/2,min(d2, 1-d2)/f2/2)
        t_max = ns*dt
        
        print(f"*** w1: f: {f1:0.3f}, d: {d1:0.3f}, ph: {ph1:0.3f} ***")
        wf1, _ = wvf.make_step_waveform2(f1, d1, ph1, dt, t_max)
        wvf.show_waveform(wf1, bit_depth = 16, lbl = "w1")
        
        print(f"*** w2: f: {f2:0.3f}, d: {d2:0.3f}, ph: {ph2:0.3f} ***")
        wf2, _ = wvf.make_step_waveform2(*w2, dt, t_max)
        wvf.show_waveform(wf2, bit_depth = 16, lbl = "w2")
        
        wfsum = wvf.sum_waveforms(wf1, wf2, thresh = 1.0)
        wvf.show_waveform(wfsum, bit_depth = 16, lbl = "w1+w2")
        wres1 = wvf.extract_waveform_info(wfsum, dt)
        
        wfres1, _ = wvf.make_step_waveform2(*wres1, dt, t_max)
        if any(wfres1):
            wvf.show_waveform(wfres1, bit_depth = 16, lbl="w1+w2?")
        else:
            print("no wfres1 :(")
            
        wfdiff = wvf.diff_waveforms(wf1, wf2, rect = True)
        wvf.show_waveform(wfdiff, bit_depth = 16, lbl = "w1-w2")
        wres2 = wvf.extract_waveform_info(wfdiff, dt)
        
        wfres2, _ = wvf.make_step_waveform2(*wres2, dt, t_max)
        if any(wfres2):
            wvf.show_waveform(wfres2, bit_depth = 16, lbl="w1-w2?")
        else:
            print("no wfres2 :(")
        
        wfdiff2 = wvf.diff_waveforms(wf2, wf1, rect = True)
        wvf.show_waveform(wfdiff2, bit_depth = 16, lbl = "w2-w1")
        wres3 = wvf.extract_waveform_info(wfdiff2, dt)
        
        wfres3, _ = wvf.make_step_waveform2(*wres3, dt, t_max)
        if any(wfres3):
            wvf.show_waveform(wfres3, bit_depth = 16, lbl="w2-w1?")
        else:
            print("no wfres1 :(")
        
        if show_spectrum:
            print("w1 spectrum")
            wvf.show_ft(wf1, polar = True)
            print()
            
            print("w2 spectrum")
            wvf.show_ft(wf2, polar = True)
            print()
            
            print("w1+w2 spectrum")
            wvf.show_ft(wfsum, polar = True)
            print()
            
            print("w1-w2 spectrum")
            wvf.show_ft(wfdiff, polar = True)
            print()
            
            print("w2-w1 spectrum")
            wvf.show_ft(wfdiff2, polar = True)
            print()
            
        wf_direct_sum = [a+b for a, b in zip(w1, w2)]
        wf_direct_diff = [abs(a-b) for a, b in zip(w1, w2)]
        
        wf_data =[]
        for i in range(3):
            row = [ind_lbls[i]]
            for wff in [w1, w2, wres1, wres2, wres3, wf_direct_sum, wf_direct_diff]:
                row.append(wff[i])
            wf_data.append(row)
        
        print(tabulate(wf_data, headers = wf_headers))
        print()
        
        sc, _, _ = wvf.compare_waveforms(w1, w2, dt = dt, t_max = t_max)
        
        sc_scaled = sc / max(d1, d2)
        print(f"Score = {sc:0.1%}")
        print()
        score_data.append([f1, f2,d1, d2, ph1, ph2, sc, sc_scaled])
    
    print(tabulate(score_data, headers = ["f1", "f2","d1","d2","phi1","phi2", "Score", "D scaled"]))
    # p_on_naive = d1*d2
    # print(f"Naive P(w1 and w2): {p_on_naive:0.3f}")

def test_time_res():
    w1 = get_random_waveform()
    w2 = get_random_waveform()
    
    f1, d1, ph1 = w1
    tp1 = 1/f1
    ton1 = d1*tp1
    toff1 = (1-d1)*tp1
    
    print(f"f1 {f1:0.3} Hz, d1 {d1:0.3}, tp1 {tp1:0.3f} s, ton1 {ton1:0.3f} s, toff1 {toff1:0.3f} s")
    
    f2, d2, ph2 = w2
    
    tp2 = 1/f2
    ton2 = d2*tp2
    toff2 = (1-d2)*tp2
    
    print(f"f2 {f2:0.3} Hz, d2 {d2:0.3}, tp2 {tp2:0.3f} s, ton2 {ton2:0.3f} s, toff2 {toff2:0.3f} s")
    
    tn = 1/(f1+f2)
    tp = 1/abs(f1 - f2)
    
    print(f"tn {tn:0.3f} s, tp {tp:0.3f} s")
    
    tmin = min(ton1, toff1, ton2, toff2)
    tmax = max(tp1, tp2)
    
    nc = 3
    dt = tmin
    t_max = tmax * nc
    
    wf1, _dt = wvf.make_step_waveform2(f1, d1, ph1, dt, t_max)
    wf2, _dt = wvf.make_step_waveform2(f2, d2, ph2, dt, t_max)
    
    wvf.show_waveform(wf1)
    wvf.show_waveform(wf2)

        
def main():
    
    
    # test_waveforms(num_tests = 5)
    # test_numden(num_tests = 1)
    # test_compare_waveforms(num_tests = 3)
    # test_fts(duty_tot = 7)
    # test_ips()
    test_alg(num_tests = 1, show_spectrum=True)
    
    
    
    
    
    
    pass


    

if __name__=="__main__":
    main()
