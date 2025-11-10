
import numpy as np


class Activation:
    
    def __init__(self, act_fn, **params):
        
        self.act_fn = act_fn
        self.params = params
    
    def I(self, x):
        return x
    
    def ReLU(self, x):
        return max(x, 0)

    def LeakyReLU(self, x):
        
        m = self.params.get("leaky_relu_m", 0.01)
        if x >=0:
            return x
        else:
            return -m*x

    def Sigmoid(self,x):
        k = self.params.get("sigmoid_k", 1.0)
        ex = np.exp(-k*x)
        return 1 / (1+ex)
    
    def Tanh(self, x):
        expp = np.exp(x)
        expm = np.exp(-x)
        return (expp-expm)/(expp+expm)
    
    def __call__(self, x):
        if self.act_fn == "ReLU":
            return self.ReLU(x)
        if self.act_fn == "LeakyReLU":
            return self.LeakyReLU(x)
        elif self.act_fn == "Sigmoid":
            return self.Sigmoid(x)
        elif self.act_fn == "Tanh":
            return self.Tanh(x)
        elif self.act_fn == "I":
            return self.I(x)
        else:
            return self.I(x)
    
    def __str__(self):
        return self.act_fn
    
    def to_dict(self):
        return {
            "act_fn":self.act_fn,
            "params":self.params
        }