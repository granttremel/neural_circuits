

import numpy as np
from tabulate import tabulate
from neural_circuits import draw


def calculate_wave_product(f0, f1, t0, t1, nsamps = 100):
    
    if f0 == f1:
        tm = 2*np.pi/f0
    else:
        tm = 2*np.pi/np.abs(f0 - f1)
    
    dom = np.linspace(0, tm, nsamps)
    w0 = np.cos(f0*dom - t0)
    w1 = np.cos(f1*dom - t1)
    
    w0w1_norm = np.power(w0+w1,2)
    w0w1_int = (tm / nsamps)*np.sum(w0w1_norm)
    w0_norm = np.power(w0, 2)
    w1_norm = np.power(w1, 2)
    
    w0_int = (tm / nsamps)*np.sum(w0_norm)
    w1_int = (tm / nsamps)*np.sum(w1_norm)
    
    return float(w0w1_int), float(w0_int), float(w1_int)


def calculate_thresh_wave_product(f0, f1, t0, t1, thresh, nsamps, do_mean = False):
    if f0 == f1:
        tm = 2*np.pi/f0
    else:
        tm = 2*np.pi/np.abs(f0 - f1)
    
    dom = np.linspace(0, tm, nsamps)
    w0 = np.cos(f0*dom - t0)
    w1 = np.cos(f1*dom - t1)
    
    if do_mean:
        fact = 1/nsamps
    else:
        fact = tm/nsamps
    
    w0w1_norm = np.power(w0+w1,2)
    w0w1_int = fact*np.sum(w0w1_norm > thresh)
    w0_norm = np.power(w0, 2)
    w1_norm = np.power(w1, 2)
    
    w0_int = fact*np.sum(w0_norm > thresh)
    w1_int = fact*np.sum(w1_norm > thresh)
    
    return float(w0w1_int), float(w0_int), float(w1_int)


def calculate_thresh_pulse_product(fr0, f0, t0, fr1, f1, t1, tmax, thresh, nsamps, do_mean = False):
    if f0 == f1:
        tm = 2*np.pi/f0
    else:
        tm = 2*np.pi/np.abs(f0 - f1)
    
    tres = nsamps / tmax
    td, w0 = make_pulse_train(fr0, tres, f0, t0, tmax)
    td, w1 = make_pulse_train(fr1, tres, f1, t1, tmax)
    
    if do_mean:
        fact = 1/nsamps
    else:
        fact = tm/nsamps
    
    w0w1_norm = np.power(w0+w1,2)
    w0w1_int = fact*np.sum(w0w1_norm > thresh)
    w0_norm = np.power(w0, 2)
    w1_norm = np.power(w1, 2)
    
    w0_int = fact*np.sum(w0_norm > thresh)
    w1_int = fact*np.sum(w1_norm > thresh)
    
    return float(w0w1_int), float(w0_int), float(w1_int)

def test_wave_products():
    f0 = 1
    t0 = 0
    
    nps =101
    f1s = np.linspace(.5, 2, nps)
    t1s = np.linspace(0, 2*np.pi, nps)
    threshs = np.linspace(0, 3, nps)
    
    f1 = 1
    t1= 1
    thresh = 2.0
    
    rows = []
    data = []
    
    for f1 in f1s[::5]:
        data = []
        for t1 in t1s:
            wp, _, _ = calculate_thresh_wave_product(f0, f1, t0, t1, thresh, nsamps = 200, do_mean = True)
            data.append(wp)
            
        maxval = None
        sctxt = draw.scalar_to_text_nb(data, minval = 0, maxval = maxval, add_range = True)
        for r in sctxt:
            print(r)
    
    print(tabulate(rows))
    
def test_pulse_products():
    f0 = 1
    t0 = 0
    
    nps =101
    f1s = np.linspace(.5, 2, nps)
    t1s = np.linspace(0, 2*np.pi, nps)
    threshs = np.linspace(0, 3, nps)
    
    tmax = 20
    nsamps = 201
    
    fr0 = 0.1
    f0 = 10*fr0
    t0=0
    fr1 = 0.13
    f1 = 11 * fr1
    t1= 1.2/f1
    thresh = 2.0
    
    rows = []
    data = []
    
    for f1 in f1s[::5]:
        data = []
        # for t1 in t1s:
        for thresh in threshs:
            wp = calculate_thresh_pulse_product(fr0, f0, t0, fr1, f1, t1, tmax, thresh, nsamps)
            data.append(wp)
            
        maxval = None
        sctxt = draw.scalar_to_text_nb(data, minval = 0, maxval = maxval, add_range = True)
        for r in sctxt:
            print(r)
    
    print(tabulate(rows))

def make_pulse_train(f_res, t_res, f_pulse, t_phase, max_t):
    
    sig = f_res
    
    def gauss(t, tc, sig):
        dt = (tc - t)
        return np.exp(-dt*dt/sig/sig/2)
    
    fulldom = np.linspace(-max_t/2, max_t/2, int(2/t_res))
    
    Tpulse = 1/f_pulse
    t_pulses = np.arange(-max_t/2, max_t/2, Tpulse)
    
    signal = np.zeros(fulldom.shape)
    for tp in t_pulses:
        v = gauss(fulldom, tp - t_phase, sig)
        signal += v
    
    return fulldom, np.array(signal)
    
def test_pulse_train():
    
    f_res = 0.1 # Hz
    t_res = 0.01 # s
    max_t = 20
    
    f_pulse = 10*f_res
    
    # for t_res in [0.005,0.01,0.02,0.05,0.1]:
    # for f_res in [0.05,0.1,0.2,0.5]:
    # for fp in [2,5, 10, 20]:
    for t_phase in [-1.5/f_pulse, 0, 1.5/f_pulse]:
        # f_pulse = fp*f_res
        tdom, pt = make_pulse_train(f_res, t_res, f_pulse, t_phase, max_t)
        
        sctxt = draw.scalar_to_text_nb(pt, minval = 0, maxval = 1)
        for r in sctxt:
            print(r)
        

def main():
    
    # test_wave_products()
    
    # test_pulse_train()
    
    test_pulse_products()
    
    
    pass
    
if __name__=="__main__":
    main()