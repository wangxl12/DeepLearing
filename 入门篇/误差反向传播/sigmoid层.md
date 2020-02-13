```python 
import numpy as np


class sigmoid(object):
    def __init__(self):
        self.out = None

    def forward(self, x):
        self.out = 1.0 / (1 + np.exp(-x))
        return self.out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out

        return dx
    
```