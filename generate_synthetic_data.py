import numpy as np
import matplotlib.pyplot as plt


class gauss(object):
    def __init__(self, mu, V):
        self.mu = mu # Mean
        self.V = V # Variance

    def sample(self, size=None):
        return np.random.normal(loc=self.mu, scale=np.sqrt(self.V), size=size)

    def __call__(self, x): # Probability density
        return np.exp(-0.5*(x-self.mu)**2/self.V)/np.sqrt(2*np.pi*self.V)

# Mixture model
class MM(object):
    def __init__(self, components, proportions):
        self.components = components
        self.proportions = proportions

    def sample(self, size=None):
        if size is None:
            nums = np.random.multinomial(1, self.proportions)
            c = nums.index(1) # which class got picked
            return self.components[c].sample()
        else:
            out = np.empty((size,), dtype=float)
            nums = np.random.multinomial(size, self.proportions)
            i = 0
            for component, num in zip(self.components, nums):
                out[i:i+num] = component.sample(size=num)
                i += num
            return out

    def __call__(self, x):
        return np.sum([p*c(x) for p, c in zip(self.proportions, self.components)], axis=0)

    def plot(self, axis=None, **kwargs):
        """ Plot the mixture model pdf."""
        if axis is None:
            axis = plt.gca()
        x = np.arange(-2,2,0.01)
        y = self(x)
        axis.plot(x, y, **kwargs)

def generate_dataset(N):
    a = np.ones(N)
    b = np.random.uniform(low=0, high=1, size=N)
    c = np.random.uniform(low=0, high=1, size=N)
    x = np.array([[a[i], b[i], c[i]] for i in range(0, N)])

    model = []

    for i in range(N):
        mu = [x[i, 1], x[i, 1] ** 4]
        V = [0.01, 0.04]
        p = [(np.exp(-2 * x[i, 1])), (1 - np.exp(-2 * x[i, 1]))]
        model.append(MM([gauss(mu0, V0) for mu0, V0 in zip(mu, V)], p))

    y = np.zeros((N))  # 500 samples
    for i in range(N):
        y[i] = model[i].sample(size=1)

    return x, y