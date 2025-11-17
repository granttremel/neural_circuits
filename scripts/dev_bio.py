
from neural_circuits import bio
from neural_circuits.bio import Consts, Ion, IonData, Electrolyte, IonChannel, MembraneConsts, Membrane, NeuronData
from neural_circuits.utils import BaseUnit, MagUnit, Units

import numpy as np

def get_channels(p_na = 1.0, p_k = 1.0, p_cl = 0.01):
    
    na_chan = IonChannel(Ion.Na, p_na, 0.0, 1.0, 0.0)
    k_chan = IonChannel(Ion.K, p_k, 0.0, 1.0, 0.0)
    cl_chan = IonChannel(Ion.Cl, p_cl, 0.0, 1.0, 0.0)
    
    return [na_chan, k_chan, cl_chan]

def get_channels_rest():
    
    na_chan = IonChannel(Ion.Na, 0.0, 0.0, 1.0, 0.0)
    k_chan = IonChannel(Ion.K, 0.15, 0.0, 1.0, 0.0)
    cl_chan = IonChannel(Ion.Cl, 0.001, 0.0, 1.0, 0.0)
    
    return [na_chan, k_chan, cl_chan]

def get_channels_spike():
    
    na_chan = IonChannel(Ion.Na, 1.0, 0.0, 1.0, 0.0)
    k_chan = IonChannel(Ion.K, 0.00, 0.0, 1.0, 0.0)
    cl_chan = IonChannel(Ion.Cl, 0.000, 0.0, 1.0, 0.0)
    
    return [na_chan, k_chan, cl_chan]

def test_ion():
    
    ion = Ion.Na
    na_data = IonData.from_ion(ion)
    
    for ion_name, ion_val in Ion._member_map_.items():
        print(ion_name, ion_val)
        ion = ion_val
        ion_data = IonData.from_ion(ion)
        print(ion_name, ion, ion_data)

def test_unit():
    bu = BaseUnit("mol")
    # bu2 = BaseUnit("moles")
    
    print("mol" in BaseUnit)
    print("moles" in BaseUnit)
    
def test_em_rest():
    rest_mem = Membrane.rest()
    
    Em_rest = rest_mem.Em()
    Em_nak = rest_mem.Em(Ion.Na, Ion.K)
    Em_nakcl = rest_mem.Em(Ion.Na, Ion.K, Ion.Cl)
    
    Em_na = rest_mem.Em(Ion.Na)
    Em_k = rest_mem.Em(Ion.K)
    Em_cl = rest_mem.Em(Ion.Cl)
    Em_ca = rest_mem.Em(Ion.Ca)
    
    print(f"rest = {Em_rest:0.3f}, Em_nak = {Em_nak:0.3f}, Em_nakcl = {Em_nakcl:0.3f}")
    print(f"Em_na = {Em_na:0.3f}, Em_k = {Em_k:0.3f}, Em_cl = {Em_cl:0.3f}, Em_ca = {Em_ca:0.3f}")
    
    # Em_depol = 
    

def test_em_spike():
    
    chans_rest = get_channels_rest()
    rest_mem = Membrane.rest(channels = chans_rest)
    Em_rest = rest_mem.Em()
    edr_na_rest = rest_mem.electrodiffusion_ratio(Ion.Na)
    edr_k_rest = rest_mem.electrodiffusion_ratio(Ion.K)
    
    chans_spike = get_channels_spike()
    spike_mem = Membrane.spike(channels = chans_spike)
    Em_spike = spike_mem.Em()
    
    edr_na_spike = spike_mem.electrodiffusion_ratio(Ion.Na)
    edr_k_spike = spike_mem.electrodiffusion_ratio(Ion.K)
    print(f"rest = {Em_rest:0.3f}, spike = {Em_spike:0.3f}")
    
    print(f"edr at rest: Na: in: {edr_na_rest[0]:0.3f}, out: {edr_na_rest[1]:0.3f}, K: in: {edr_k_rest[0]:0.3f}, out: {edr_k_rest[1]:0.3f}")
    print(f"edr at spike: Na: in: {edr_na_spike[0]:0.3f}, out: {edr_na_spike[1]:0.3f}, K: in: {edr_k_spike[0]:0.3f}, out: {edr_k_spike[1]:0.3f}")
    # print(f"edr at rest: Na: {edr_na_rest:0.3f}, K: {edr_k_rest:0.3f}")
    
    return rest_mem, spike_mem

def test_ion_em(test_ion):
    
    rest_mem = Membrane.rest()
    Em = rest_mem.Em(test_ion)
    
    print(f"Em for {test_ion.name} = {Em:0.3f}")
    
    print(f"Ex = {bio.Consts.Ex():0.3f}")
    print(f"Ex base 10 = {bio.Consts.Ex_b10():0.3f}")

def test_p_ion_em():
    rest_mem = Membrane.rest()
    p_na = IonData.Na().D / rest_mem.t
    p_k = IonData.K().D / rest_mem.t
    p_cl = IonData.Cl().D / rest_mem.t
    
    channels = get_channels(p_na = p_na, p_k = 0.0, p_cl = 0.0)
    rest_mem.set_channels(channels)
    Em = rest_mem.Em(Ion.Na, Ion.K, Ion.Cl)
    print(f"rest = {Em:0.3f}")
    
    channels = get_channels(p_na = 0.05*p_na, p_k = p_k, p_cl = 0.0)
    rest_mem.set_channels(channels)
    Em = rest_mem.Em(Ion.Na, Ion.K, Ion.Cl)
    print(f"rest = {Em:0.3f}")
    
    channels = get_channels(p_na = 0.05*p_na, p_k = p_k, p_cl = 0.2*p_cl)
    rest_mem.set_channels(channels)
    Em = rest_mem.Em(Ion.Na, Ion.K, Ion.Cl)
    print(f"rest = {Em:0.3f}")

def test_units():
    value_init = 0.9
    unit_str = "uF / cm2"
    
    units = Units.from_unit_str(value_init, unit_str)
    conv_units = units.convert("F", "m")
    
    conv_value = Units.quick_convert(value_init, unit_str, "F", "m")
    
    # unit = Units.from_unit_str(10, "km / s2")
    # print(unit.full_str())
    
    # unit_base = unit.as_base()
    # print(unit_base.full_str())
    
    # unit_base = unit.convert("mm", "us")
    # print(unit_base.full_str())
    
    # cm_unit = Units(0.9, uF = 1, cm = -2)
    # print(cm_unit.full_str(value_fmt = "0.3e"))
    # # tgt_unit = Units(1.0, F = 1, m = -2)
    # cm_std = cm_unit.convert("F","m")
    # print(cm_std.full_str(value_fmt = "0.3e"))


    # cm_std = cm_unit.convert("nF","km")
    # print(cm_std.full_str(value_fmt = "0.3e"))

def test_debye():
    elc_in = Electrolyte.intra_rest()
    elc_out = Electrolyte.extra_rest()
    
    k_in = Units.quick_convert(elc_in.calculate_debye_length(), "m","nm")
    k_out = Units.quick_convert(elc_out.calculate_debye_length(), "m","nm")
    
    print(f"debye lengths, intracellular: {k_in:0.3f}nm, extracellular: {k_out:0.3f}nm")
    
    for pH in range(1, 14):
        
        elc_in = Electrolyte.intra_rest(pH = pH)
        k_ph = Units.quick_convert(elc_in.calculate_debye_length(), "m","nm")
        print(f"debyte length, intracellular, pH = {pH}: {k_ph:0.3f}")

def test_em_ph():
    
    for pH in range(1, 14):
        
        elc_in = Electrolyte.intra_rest(pH = pH)
        elc_out = Electrolyte.extra_rest()
        
        mem = Membrane(elc_in, elc_out)
        
        Em_ph = mem.Em()
        print(f"debyte length, intracellular, pH = {pH}: {Em_ph:0.3f}")
    
    pass

def test_mean_vel():
    n = 2
    dt = 0.001 # 1ms
    
    for ion in Ion.all_ions():
        
        iond = IonData.from_ion(ion)
        
        D_ion = iond.D
        msd = 2*n*D_ion*dt # m2
        mean_disp = np.sqrt(msd)
        mean_vel = mean_disp / dt
        
        print(f"ion {ion.name} over {dt*1e3:0.1f}ms has msd {msd*1e6*1e6:0.3f}um2, mean_disp {mean_disp*1e6:0.3f}um, mean_vel {mean_vel*1e3:0.3f}um/ms ")
    

def test_electrodiffusion_ratio():
    d = Units(10, nm = 1)
    
    c_na = Units(145-10, mmol = 1, L = -1)
    
    dcdx = c_na.divide(d)
    
    RTdcdx = dcdx.value * Consts.R * 310 # mol/L m * J/K mol * K -> J/L m
    RTdcdx = RTdcdx * (10*10*10) / 1e9 # J/m3 m = kg / s2 m nm
    
    print(f"{RTdcdx:0.3f} kg / s2 m nm")
    
    d = 10e-9
    
    Vm = -0.07 # V
    
    dvdx = Vm / d
    print(f"{dvdx / 1e6:0.3f} mV / nm")
    
    czFdvdx_in = dvdx * Consts.F * 1 * 0.145 # inside, mol/L * C / mol * V / m = C V / L m
    czFdvdx_in = czFdvdx_in * 10*10*10 / 1e9 # inside, mol / m3 * C / mol * J / C m = J / m4 = kg / s2 m nm
    print(f"intra {czFdvdx_in:0.3f} kg / s2 m nm")
    
    czFdvdx_out = dvdx * Consts.F * 1 * 0.01 # outside, mol/L * C / mol * V / m = C V / L m
    czFdvdx_out = czFdvdx_out * 10*10*10 / 1e9 # outside, mol / m3 * C / mol * J / C m = J / m4 = kg / s2 m nm
    print(f"extra {czFdvdx_out:0.3f} kg / s2 m nm")
    
    print(f"electromigration / diffusion: {czFdvdx_in / RTdcdx:0.3f}")
    
    
def test_electrodiffusion_ratio2():
    
    chans = get_channels()
    mem_rest = Membrane.rest(channels = chans)
    
    Tinv_in, Tinv_out = mem_rest.reciprocal_debye_length()
    
    print(mem_rest.intra._concs)
    print(mem_rest.extra._concs)
    
    print(f"Na solo recip debye length: inside {Tinv_in*1e9:0.3f} nm, outside {Tinv_out*1e9:0.3f} nm")
    
    edr_in, edr_out = mem_rest.electrodiffusion_ratio(Ion.Na)
    
    print(f"Electrodiffusion ratios for Na, intra: {edr_in:0.3f}, extra: {edr_out:0.3f}")
    
    edr_in, edr_out = mem_rest.electrodiffusion_ratio(Ion.K)
    
    print(f"Electrodiffusion ratios for K, intra: {edr_in:0.3f}, extra: {edr_out:0.3f}")
    

def test_longitudinal():
        
    dx = 20e-9 # like what idk, between nodes ??
    
    chans_rest = get_channels_rest()
    rest_mem = Membrane.rest(channels = chans_rest)
    Em_rest = rest_mem.Em()
    
    chans_spike = get_channels_spike()
    spike_mem = Membrane.spike(channels = chans_spike)
    Em_spike = spike_mem.Em()
    
    c_ions_rest_in = [rest_mem.intra.get_conc(Ion.Na), rest_mem.intra.get_conc(Ion.K)]
    c_ions_rest_out = [rest_mem.extra.get_conc(Ion.Na), rest_mem.extra.get_conc(Ion.K)]
    c_ions_spike_in = [spike_mem.intra.get_conc(Ion.Na), spike_mem.intra.get_conc(Ion.K)]
    c_ions_spike_out = [spike_mem.extra.get_conc(Ion.Na), spike_mem.extra.get_conc(Ion.K)]
    
    p_ion_longi = [0.5, 0.5] # idkkkk
    
    z_ions = [1, 1]
    
    Em_in_rs = bio.GHF(c_ions_spike_in, c_ions_rest_in, p_ion_longi, z_ions)
    Em_out_rs = bio.GHF(c_ions_spike_out, c_ions_rest_out, p_ion_longi, z_ions)
    
    Em_rest2 = bio.GHF(c_ions_rest_out, c_ions_rest_in, [chans_rest[0].p_active, chans_rest[1].p_active], z_ions)
    Em_spike2 = bio.GHF(c_ions_spike_out, c_ions_spike_in, [chans_spike[0].p_active, chans_spike[1].p_active], z_ions)
    
    print(f"across a longitudinal distance of {1e9*dx:} nm:")
    
    print(f"Em, rest: {1000*Em_rest:0.3f} mV, spike: {1000*Em_spike:0.3f} mV")
    print(f"Em, rest: {1000*Em_rest2:0.3f} mV, spike: {1000*Em_spike2:0.3f} mV")
    print(f"El, intra: {1000*Em_in_rs:0.3f} mV, extra: {1000*Em_out_rs:0.3f} mV")
    
    rdl_rest_in = bio.reciprocal_debye_length(c_ions_rest_in)
    rdl_rest_out = bio.reciprocal_debye_length(c_ions_rest_out)
    rdl_spike_in = bio.reciprocal_debye_length(c_ions_spike_in)
    rdl_spike_out = bio.reciprocal_debye_length(c_ions_spike_out)
    
    print(f"recip debye at rest: intra: {1e9 * rdl_rest_in:0.3f} nm, extra: {1e9 * rdl_rest_out:0.3f} nm ")
    print(f"recip debye at spike: intra: {1e9 * rdl_spike_in:0.3f} nm, extra: {1e9 * rdl_spike_out:0.3f} nm ")
    
    
    NP_Na_in = bio.nernst_planck(0.1, c_ions_rest_in[0], c_ions_spike_in[0], dx, 1.0, z=1, T=310)
    # NP_Na_out = bio.nernst_planck(0.0, c_ions_rest_out[0], c_ions_spike_out[0], dx, 1.0, z=1, T=310)
    
    # NP_K_in = bio.nernst_planck(Em_in_rs, c_ions_rest_in[1], c_ions_spike_in[1], dx, 1.0, z=1, T=310)
    # NP_K_out = bio.nernst_planck(Em_out_rs, c_ions_rest_out[1], c_ions_spike_out[1], dx, 1.0, z=1, T=310)
    
    edf_Na_in = NP_Na_in[1] / NP_Na_in[0]
    # edf_Na_out = NP_Na_out[1] / NP_Na_out[0]
    edf_Na_out = -1
    # edf_K_in = NP_K_in[1] / NP_K_in[0]
    # edf_K_out = NP_K_out[1] / NP_K_out[0]
    
    print(f"edf for Na between rest and spike, intra: {edf_Na_in:0.3f}, extra: {edf_Na_out:0.3f}")
    # print(f"edf for K between rest and spike, intra: {edf_K_in:0.3f}, extra: {edf_K_out:0.3f}")
    
    
    pass

def test_np():
    
    v_ap_nonmyel = 0.050 # m/s
    rise_time = 5e-6 # s = 50us
    dx = v_ap_nonmyel * rise_time
    
    print(f"dx for an AP in non myelinated membrane: {1e6*dx:0.3f} um")
    
    V2 = -0.07 # at rest
    V1 = 0.03 # at spike
    
    c2 = 0.01 # at rest
    c1 = 0.06 # at spike
    # dx = 20e-9
    
    u = 1.0
    
    c_local = 0.06
    
    d_cont, e_cont = bio.nernst_planck(V1, V2, c1, c2, dx, u, c_local = c_local)
    
    edr = e_cont / d_cont
    
    print(f"electrodiffusion ratio: {edr:0.3f}")
    
    
    pass

def test_d_u():
    
    
    nadata = IonData.Na()
    
    
    D_Na = nadata.D
    u_Na = nadata.u()
    
    d_a = 2e-6
    
    t_avg = d_a*d_a / D_Na / 4
    
    print(f"D_Na {D_Na:0.3e} m2 / s")
    print(f"u_Na {u_Na:0.3e} mol s / kg")
    
    print(f"average time to diffuse {d_a*1e6/2:0.3f} um: {1e3*t_avg:0.3f} ms")
    
    
    pass

def main():
    
    # test_em_ph()
    # test_em_rest()
    # test_electrodiffusion_ratio2()
    # test_em_spike()

    
    # test_em_spike()
    
    # test_longitudinal()
    
    # test_np()
    
    test_d_u()
    
    pass
    
if __name__=="__main__":
    main()
