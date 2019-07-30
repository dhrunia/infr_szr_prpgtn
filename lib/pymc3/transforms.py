import pymc3.distributions.transforms as tr
import theano.tensor as tt


class Linear(tr.ElemwiseTransform):
    name = "linear"

    def __init__(self, alpha, beta):
        super(tr.ElemwiseTransform, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, x):
        return self.alpha * x + self.beta

    def backward(self, x):
        return (1 / self.alpha) * (x - self.beta)

    def jacobian_det(self, x):
        return -tt.log(self.alpha)
