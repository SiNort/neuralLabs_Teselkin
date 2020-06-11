import numpy as np


class RadialBasisNetwork:
    def __init__(self, hiddenNum, sigma=1.0):

        self.hiddenNum = hiddenNum
        self.sigma = sigma
        self.centers = 0
        self.weights = 0

    def fit(self, x, y):
        self.centers = x[np.random.choice(len(x), self.hiddenNum)]
        inter_matrix = self.calc_interpolation_matrix(x)
        inverse_inter_matrix = np.linalg.pinv(inter_matrix)
        self.weights = inverse_inter_matrix @ y

    def predict(self, x):
        inter_matrix = self.calc_interpolation_matrix(x)
        return inter_matrix @ self.weights

    def calc_interpolation_matrix(self, x):
        inter_matrix = np.zeros((len(x), self.hiddenNum))
        for i, point in enumerate(x):
            for j, center in enumerate(self.centers):
                inter_matrix[i, j] = self.radial_basis_func(point, center)
        return inter_matrix

    def radial_basis_func(self, point, center):
        return np.exp(-np.linalg.norm((point - center) ** 2 / (2 * self.sigma ** 2)))


class NeuralNetwork:
    def __init__(self, learningRate=0.01, mu=0.85, ):

        self.learningRate = learningRate
        self.mu = mu

        self.output = 0
        self.layers = []
        self.parameters = []

    def add_layer(self, inputNum, outputNum, activationFunc="nope", isBias=False):
        self.layers.append(Layer(inputNum, outputNum, activationFunc, isBias))
        self.parameters.append({"main": {"weight": 0, "bias": 0}})

    def forward(self, x):
        tmp_input = x
        for layer, parameter in zip(self.layers, self.parameters):
            tmp_input = layer.forward(tmp_input, parameter["main"], self.mu)
        self.output = tmp_input
        return tmp_input

    def backward(self, expected):
        grad = self.mean_square_error(expected, True)
        for layer, parameter in zip(list(reversed(self.layers)), list(reversed(self.parameters))):
            grad = layer.backward(grad, parameter["main"], self.mu)

    def predict(self, x):
        tmp_input = x
        for layer in self.layers:
            tmp_input = layer.forward(tmp_input, {"weight": 0, "bias": 0}, 0)
        self.output = tmp_input
        return tmp_input

    def update(self):
        for layer, parameter in zip(self.layers, self.parameters):
            self.nesterov_gradient(layer.weights, layer.deltaWeights, parameter)
            if layer.isBias:
                self.nesterov_gradient(layer.bias, layer.deltaBias, parameter, "bias")

    def mean_square_error(self, expected, der=False):
        if not der:
            return np.mean((self.output - expected) ** 2)
        return self.output - expected

    def nesterov_gradient(self, weights, deltaWeights, parameter, key="weight"):
        parameter["main"][key] = self.mu * parameter["main"][key] + self.learningRate * deltaWeights
        weights -= parameter["main"][key]


class Layer:
    def __init__(self, inputNum, outputNum, activationFunc="none", isBias=False):

        self.bias = 0
        self.isBias = isBias
        if isBias:
            self.bias = np.random.uniform(low=-np.sqrt(1 / inputNum),
                                          high=np.sqrt(1 / inputNum), size=[1, outputNum])

        self.weights = np.random.uniform(low=-np.sqrt(1 / inputNum),
                                         high=np.sqrt(1 / inputNum), size=[inputNum, outputNum])

        self.activationFunc = lambda x: x
        self.derActivationFunc = lambda x: 1

        if activationFunc == "tanh":
            self.activationFunc = lambda x: np.tanh(x)
            self.derActivationFunc = lambda x: 1 - self.activationFunc(x) ** 2

    def forward(self, x, parameter, mu):
        self.input = x
        self.middleInput = x @ (self.weights - mu * parameter["weight"])
        if self.isBias:
            self.middleInput += (self.bias - mu * parameter["bias"])
        self.output = self.activationFunc(self.middleInput)
        return self.output

    def backward(self, grad, parameters, mu):
        delta_output = grad * self.derActivationFunc(self.middleInput)
        self.deltaWeights = self.input.T @ delta_output
        if self.isBias:
            self.deltaBias = (delta_output - mu * parameters["bias"])
        delta_input = delta_output @ (self.weights - mu * parameters["weight"]).T
        return delta_input


class ElmanNetwork:
    def __init__(self, inputNum, hiddenNum, outputNum, learningRate):
        self.learningRate = learningRate

        self.input_hidden_weights = np.random.uniform(-np.sqrt(1 / inputNum), np.sqrt(1 / inputNum),
                                                      size=[inputNum, hiddenNum])
        self.input_hidden_bias = np.random.uniform(size=[1, hiddenNum])

        self.hidden_hidden_weights = np.random.uniform(-np.sqrt(1 / hiddenNum), np.sqrt(1 / hiddenNum),
                                                       size=[hiddenNum, hiddenNum])
        self.hidden_hidden_bias = np.random.uniform(size=[1, hiddenNum])

        self.hidden = np.zeros(shape=[1, hiddenNum])
        self.hidden_save = np.zeros(shape=[1, hiddenNum])

        self.weights = np.random.uniform(-np.sqrt(1 / hiddenNum), np.sqrt(1 / hiddenNum),
                                         size=[hiddenNum, outputNum])

    def forward(self, x):
        self.input = x
        self.hidden_save = self.hidden
        self.hidden = self.input @ self.input_hidden_weights + self.input_hidden_bias + self.hidden_save @ self.hidden_hidden_weights + self.hidden_hidden_bias
        self.hidden = np.tanh(self.hidden)
        self.output = self.hidden @ self.weights
        return self.output

    def backward(self, y):
        delta_loss = self.output - y
        self.delta_weights = self.hidden.T @ delta_loss
        delta_hidden = delta_loss @ self.weights.T
        grad = (1 - np.tanh(self.hidden) ** 2) * delta_hidden
        self.delta_weights_ih = self.input.T @ grad
        self.delta_bias_ih = 1 * grad
        self.delta_weights_hh = self.hidden_save.T @ grad
        self.delta_bias_hh = 1 * grad

    def update(self):
        self.weights -= self.learningRate * self.delta_weights
        self.input_hidden_weights -= self.learningRate * self.delta_weights_ih
        self.input_hidden_bias -= self.learningRate * self.delta_bias_ih
        self.hidden_hidden_weights -= self.learningRate * self.delta_weights_hh
        self.hidden_hidden_bias -= self.learningRate * self.delta_bias_hh
