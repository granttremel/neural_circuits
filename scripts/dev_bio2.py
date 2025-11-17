
import numpy as np

from neural_circuits.modeling import bio, physics, units, empirical
from neural_circuits.modeling.bio import Consts, Ion, IonData, Electrolyte, MembraneConsts, Membrane, NeuronData
from neural_circuits.modeling.channel import IonChannel, NaChannel, KChannel
from neural_circuits.modeling.empirical import MembraneData, MembraneDataset

def test_dataset():
        
    sigworth_data = MembraneDataset.sigworth_1980()
    
    sigworth_data.print_table()
    
    mean, sd, minn, maxx, n = sigworth_data.get_stats()
    
    print()
    print(mean)
    print(sd)
    print(minn)
    print(maxx)
    
    print()
    print(repr(mean.Node))
    
    
    phys = physics.Physics()
    
    phys._GHF()
    
    
def test_channel():
    
    
    
    na1 = NaChannel.Na1()
    
    print(f"Na1: g={na1.g:0.2f}, t_act = {na1.t_act:0.2f}, v_act = {na1.v_act:0.2f}, t_inact = {na1.t_inact:0.2f}, v_inact = {na1.v_inact:0.2f}")
    
    na1.set_count(1000)
    
    t = np.linspace(0, 100, 101)
    Vm = -70 + t
    
    opens = []
    
    for i in range(len(t)):
        ti = t[i]
        vmi = Vm[i]
        n_open = na1.get_open(ti, vmi)
        opens.append(n_open)
        print(f"t={ti:0.1f}, Vm={vmi:0.1f}, N_open={n_open}")
    # k_dr = KChannel.K1_delayedrect()
    
    
    
    
    
    
    pass

def main():
    
    test_channel()

    
    pass
    
if __name__=="__main__":
    main()
