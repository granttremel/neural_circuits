
from neural_circuits import bio
from neural_circuits.bio import Consts, Ion, IonData, Electrolyte, IonChannel, MembraneConsts, MembraneData, NeuronData
from neural_circuits.bio import BaseUnit, MagUnit, Units

import numpy as np

def get_channels(p_na = 1.0, p_k = 1.0, p_cl = 0.01):
    
    na_chan = IonChannel(Ion.Na, p_na, 0.0, 1.0, 0.0)
    k_chan = IonChannel(Ion.K, p_k, 0.0, 1.0, 0.0)
    cl_chan = IonChannel(Ion.Cl, p_cl, 0.0, 1.0, 0.0)
    
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
    rest_mem = MembraneData.rest()
    
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

def test_ion_em(test_ion):
    
    rest_mem = MembraneData.rest()
    Em = rest_mem.Em(test_ion)
    
    print(f"Em for {test_ion.name} = {Em:0.3f}")
    
    print(f"Ex = {bio.Consts.Ex():0.3f}")
    print(f"Ex base 10 = {bio.Consts.Ex_b10():0.3f}")

def test_p_ion_em():
    rest_mem = MembraneData.rest()
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
        
        mem = MembraneData(elc_in, elc_out)
        
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
    mem_rest = MembraneData.rest(channels = chans)
    
    Tinv_in, Tinv_out = mem_rest.reciprocal_debye_length()
    
    print(mem_rest.intra._concs)
    print(mem_rest.extra._concs)
    
    print(f"Na solo recip debye length: inside {Tinv_in*1e9:0.3f} nm, outside {Tinv_out*1e9:0.3f} nm")
    
    edr_in, edr_out = mem_rest.electrodiffusion_ratio(Ion.Na)
    
    print(f"Electrodiffusion ratios for Na, intra: {edr_in:0.3f}, extra: {edr_out:0.3f}")
    
    edr_in, edr_out = mem_rest.electrodiffusion_ratio(Ion.K)
    
    print(f"Electrodiffusion ratios for K, intra: {edr_in:0.3f}, extra: {edr_out:0.3f}")
    

def main():
    
    # test_em_ph()
    # test_em_rest()
    test_electrodiffusion_ratio2()
    
    pass
    
if __name__=="__main__":
    main()
