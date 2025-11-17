

import numpy as np



def get_random_vectors(dim, mean, sd, num_vectors, normalize = False):
    
    rands = np.random.normal(mean, sd, size = (num_vectors, dim))
    
    if normalize:
        norm = np.sqrt(np.sum(np.power(rands,2), axis = -1))[:,None]
        rands = rands / norm
    
    return rands

def test_random_vectors(num_tests = 5):
    
    dim = 128
    mean = 0
    sd = 1
    num_vectors = 200
    
    for n in range(num_tests):
        opts = get_random_vectors(dim, mean, sd, num_vectors, normalize = True)
        A = get_random_vectors(dim, mean, sd, 1, normalize = True)
        B = -A
        
        Asum = 0
        Asumsq = 0
        Bsum = 0
        Bsumsq = 0
        
        for nv in range(num_vectors):
            
            Asim = np.dot(A, opts[nv])[0]
            Bsim = np.dot(B, opts[nv])[0]
            
            Asum += Asim
            Asumsq += Asim*Asim
            Bsum += Bsim
            Bsumsq += Bsim*Bsim
        
        print(f"Test {n}, vector A has sum {Asum:0.3f} and sum of squares {Asumsq:0.3f}")
        print(f"Test {n}, vector B has sum {Bsum:0.3f} and sum of squares {Bsumsq:0.3f}")
        
        # Amean = Asum / num_vectors
        # Ameansq = np.sqrt((Asumsq / num_vectors - Amean*Amean) / num_vectors)
        # Bmean = Bsum / num_vectors
        # Bmeansq = np.sqrt((Bsumsq / num_vectors - Bmean*Bmean) / num_vectors)
        
        # print(f"Test {n}, vector A has sum {Asum:0.1f} and sum of squares {Asumsq:0.3f}")
        # print(f"Test {n}, vector B has sum {Bsum:0.1f} and sum of squares {Bsumsq:0.1f}")
    

def main():
    
    test_random_vectors()
    
    pass


if __name__=="__main__":
    main()
    
    pass




