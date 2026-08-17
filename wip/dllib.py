
import numpy as np

def _unbroadcast(grad, shape):
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, (g, s) in enumerate(zip(grad.shape, shape)):
        if s == 1 and g > 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad

class Tensor:
    def __init__(self, data):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self._back = lambda: None
        self._children = []

    def __repr__(self):
        return str(self.data)

    def __add__(self, other):
        out = Tensor(self.data + other.data)
        def _back():
            self.grad  += _unbroadcast(out.grad, self.data.shape)
            other.grad += _unbroadcast(out.grad, other.data.shape)
        out._back = _back
        out._children = [self, other]
        return out

    def __mul__(self, other):
        out = Tensor(self.data * other.data)
        def _back():
            self.grad  += _unbroadcast(other.data * out.grad, self.data.shape)
            other.grad += _unbroadcast(self.data  * out.grad, other.data.shape)
        out._back = _back
        out._children = [self, other]
        return out

    def __sub__(self, other):
        return self + (Tensor([-1]) * other)

    def __matmul__(self, other):
        out = Tensor(self.data @ other.data)
        def _back():
            # grad w.r.t. self
            if self.data.ndim == 2 and other.data.ndim == 1:
                self.grad += np.outer(out.grad, other.data)
            else:
                self.grad += out.grad @ other.data.T

            # grad w.r.t. other
            if self.data.ndim == 1 and other.data.ndim == 2:
                other.grad += np.outer(self.data, out.grad)
            else:
                other.grad += self.data.T @ out.grad

        out._back = _back
        out._children = [self, other]
        return out

    def relu(self):
        out = Tensor(np.maximum(0, self.data))
        def _back():
            self.grad += (out.data > 0) * out.grad
        out._back = _back
        out._children = [self]
        return out

    def reshape(self, *shape):
        out = Tensor(self.data.reshape(*shape))
        def _back():
            self.grad += out.grad.reshape(self.data.shape)
        out._back = _back
        out._children = [self]
        return out

    def flatten(self, start_dim=0):
        shape = self.data.shape
        new_shape = shape[:start_dim] + (-1,)
        return self.reshape(*new_shape)

    def __getitem__(self, idx):
        orig_shape = self.data.shape
        out = Tensor(self.data[idx])
        def _back():
            grad = np.zeros(orig_shape, dtype=self.data.dtype)
            grad[idx] += out.grad
            self.grad += grad
        out._back = _back
        out._children = [self]
        return out

    def exp(self):
        out = Tensor(np.exp(self.data))
        def _back():
            self.grad += out.data * out.grad
        out._back = _back
        out._children = [self]
        return out

    def log(self):
        out = Tensor(np.log(self.data))
        def _back():
            self.grad += (1.0 / self.data) * out.grad
        out._back = _back
        out._children = [self]
        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims))
        def _back():
            grad = out.grad
            if axis is not None and not keepdims:
                grad = np.expand_dims(grad, axis)
            self.grad += np.broadcast_to(grad, self.data.shape)
        out._back = _back
        out._children = [self]
        return out

    # def mean(self):
    #     n = self.data.size
    #     out = Tensor(self.data.mean())
    #     def _back():
    #         grad = out.grad * (1/n)
    #         self.grad += np.broadcast_to(grad, self.data.shape)
    #     out._back = _back
    #     return out

    def _topo_sort(self):
        topo = []
        def visit(v):
            if v not in topo:
                for child in v._children:
                    visit(child)
                topo.append(v)
        visit(self)
        return topo

    def backward(self):
        self.grad = np.ones_like(self.data)
        for v in reversed(self._topo_sort()):
            v._back()

class Linear:
    def __init__(self, in_features: int, out_features: int):
        self.W = Tensor(np.random.randn(in_features, out_features)*0.01)
        self.b = Tensor(np.zeros((1, out_features)))

    def __call__(self, x: Tensor) -> Tensor:
        return x @ self.W + self.b

    def parameters(self) -> list:
        return [self.W, self.b]

class Flatten:
    def __call__(self, x: Tensor) -> Tensor:
        return x.flatten()

        
class Kernel:
    def __init__(self, kernel_size: int):
        self.kernel_size = kernel_size

        # # Keeping kernel flat so we can use @
        self.W = Tensor(np.random.randn(kernel_size*kernel_size, 1)*0.01)
        self.b = Tensor(np.zeros((1, 1)))

    def apply(self, x: Tensor) -> Tensor:
        flat_x = x.flatten()
        return flat_x @ self.W + self.b

    def parameters(self) -> list:
        return [self.W, self.b]

class Conv2D:
    def __init__(self, kernel_count: int, kernel_size: int):
        self.kernel_count = kernel_count
        self.kernel_size = kernel_size
        self.kernels = [Kernel(kernel_size) for _ in range(kernel_count)]

    def __call__(self, x: Tensor) -> Tensor:
        output_size_x = x.data.shape[0] - self.kernel_size + 1
        output_size_y = x.data.shape[1] - self.kernel_size + 1

        out = Tensor(np.zeros((output_size_x, output_size_y, self.kernel_count)))
        out_backs = []

        # We will assume the input image is 2D for now as well.
        for i in range(output_size_x):
            for j in range(output_size_y):
                image_slice = x[i:i+self.kernel_size, j:j+self.kernel_size]
                for k, kernel in enumerate(self.kernels):
                    y = kernel.apply(image_slice)
                    out_backs.append(y._back)
                    out.data[i, j, k] += y.data

        def _back():
            for back in out_backs:
                back()

        out._back = _back
        return out

    def parameters(self) -> list:
        params = []
        for kernel in self.kernels:
            params.extend(kernel.parameters())
        return params

class ReLU:
    def __call__(self, x): return x.relu()

class Sequential:
    def __init__(self, *layers):
        self.layers = layers

    def __call__(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
            # print("layer ", type(layer), " outputted ", x)
        return x

    def parameters(self) -> list:
        params = []
        for layer in self.layers:
            if hasattr(layer, 'parameters'):
                params.extend(layer.parameters())
        return params

def cross_entropy(logits: Tensor, target: int) -> Tensor:
    """
    logits: Tensor of shape (C,) — raw scores for one sample
    target: int — true class index
    """
    exps = logits.exp()          # exp(x_i) for each class
    sum_exps = exps.sum()        # sum_i exp(x_i)
    log_sum_exps = sum_exps.log()  # log(sum_i exp(x_i))
    logit_target = logits[target]  # x_target

    # cross entropy = log(sum(exp(x))) - x_target
    #   (this is algebraically -log(softmax(x)_target))
    loss = log_sum_exps - logit_target
    return loss