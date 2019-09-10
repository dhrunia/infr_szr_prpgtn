import pymc3 as pm
import theano.tensor as tt


class Linear(pm.distributions.transforms.ElemwiseTransform):
    name = "linear"

    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def forward(self, x):
        return self.alpha * x + self.beta

    def forward_val(self, x, point=None):
        return self.alpha * x + self.beta

    def backward(self, x):
        return (1 / self.alpha) * (x - self.beta)

    def jacobian_det(self, x):
        return -tt.log(self.alpha)
