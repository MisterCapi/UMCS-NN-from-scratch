import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork():
    def __init__(self):
        self.weigths = np.random.normal(size=6) #zamien na np.random.normal(size=6) -> normal oznacza ze ma byc rozklad normalny w losowaniu bo zwykly ran daje rownomierny rozklad -> kiedys wytlumacze ock
        self.bias = np.random.normal(size=3)

    def sigmoid(self, x, derivative=False):
        sigm = 1 / (1 + np.exp(-x))
        if derivative:
            return sigm * (1 - sigm)
        else:
            return sigm

    def mse_loss(self, y_train, y_predict, derivative=False):
        if derivative:
            return (-2)*(y_train-y_predict)
        else:
            return np.mean(np.square(y_train-y_predict))

    def predict(self, X_values): # input shape --> [x1, x2]
        H0 = self.sigmoid(X_values[0]*self.weigths[0] + X_values[1]*self.weigths[2] + self.bias[0])
        H1 = self.sigmoid(X_values[0]*self.weigths[1] + X_values[1]*self.weigths[3] + self.bias[1])
        return self.sigmoid(H0*self.weigths[4] + H1*self.weigths[5] + self.bias[2])

    # input shape is hard coded to be --> X_train (4, 2); y_train (4)
    def train(self, X_train, y_train, lr=0.1, epochs=1000):
        self.weigths = np.random.normal(size=6)
        self.bias = np.random.normal(size=3)
        starting_weigths = np.mean(self.weigths)
        starting_biases = np.mean(self.bias)
        for epoch in range(epochs):
            for X_values, y_value in zip(X_train, y_train):
                # Forward propagation
                H0 = self.sigmoid(X_values[0]*self.weigths[0] + X_values[1]*self.weigths[2] + self.bias[0])
                H1 = self.sigmoid(X_values[0]*self.weigths[1] + X_values[1]*self.weigths[3] + self.bias[1])
                y_predict = self.sigmoid(H0*self.weigths[4] + H1*self.weigths[5] + self.bias[2]) # value of the last neuron O0 after activation

                # zachowane sumy wazone
                H0_weighted_sum = X_values[0]*self.weigths[0] + X_values[1]*self.weigths[2]
                H1_weighted_sum = X_values[0]*self.weigths[1] + X_values[1]*self.weigths[3]
                O0_weighted_sum = H0*self.weigths[4] + H1*self.weigths[5]

                derivative_mse = self.mse_loss(y_value, y_predict, derivative=True)

                dWeigths = np.zeros((6), dtype=np.float64)
                dBias = np.zeros((3), dtype=np.float64)

                # Deltas for weigths and biases
                dBias[2] = self.sigmoid(O0_weighted_sum, derivative=True)
                dWeigths[5] = H1 * self.bias[2]
                dWeigths[4] = H0 * self.bias[2]

                dH0 = self.weigths[4] * self.bias[2]
                dH1 = self.weigths[5] * self.bias[2]

                dBias[0] = self.sigmoid(H0_weighted_sum, derivative=True)
                dWeigths[0] = X_values[0] * self.bias[0]
                dWeigths[2] = X_values[1] * self.bias[0]

                dBias[1] = self.sigmoid(H1_weighted_sum, derivative=True)
                dWeigths[1] = X_values[0] * self.bias[1]
                dWeigths[3] = X_values[1] * self.bias[1]

                # Weigths correction
                self.weigths[0] -= lr * dWeigths[0] * dH0 * derivative_mse
                self.weigths[1] -= lr * dWeigths[1] * dH1 * derivative_mse
                self.weigths[2] -= lr * dWeigths[2] * dH0 * derivative_mse
                self.weigths[3] -= lr * dWeigths[3] * dH1 * derivative_mse
                self.bias[0] -= lr * dBias[0] * dH0 * derivative_mse
                self.bias[1] -= lr * dBias[1] * dH1 * derivative_mse

                self.weigths[4] -= lr * dWeigths[4] * derivative_mse
                self.weigths[5] -= lr * dWeigths[5] * derivative_mse
                self.bias[2] -= lr * dBias[2] * derivative_mse

                if epoch % 100 == 0:
                    result = [self.predict(x) for x in X_train]
                    loss = self.mse_loss(y_train, result)
                    # print(f'Loss {loss:1.4f}\n{result}')
        return loss, dWeigths, dBias, dH0, dH1, derivative_mse

X_train = np.array([[0, 0],
                    [0, 1],
                    [1, 0],
                    [1, 1]])

y_train = np.array([0, 1, 1, 1])

nn = NeuralNetwork()

losses = []
dWeights = []
dBiases = []
dH0s = []
dH1s = []
dO0s = []

N = 50
for i in range(N):
    variables = nn.train(X_train, y_train)
    losses.append(variables[0])
    dWeights.append(variables[1])
    dBiases.append(variables[2])
    dH0s.append(variables[3])
    dH1s.append(variables[4])
    dO0s.append(variables[5])

dWeights = np.array(dWeights).T
dBiases = np.array(dBiases).T

fig, ax = plt.subplots(2, 6)
fig.suptitle("output = [0, 1, 1, 1]")

ax[0][0].scatter(dWeights[0], losses)
ax[0][0].set_ylabel("loss")
ax[0][0].set_xlabel("dWeigths[0]")

ax[0][1].scatter(dWeights[1], losses)
ax[0][1].set_xlabel("dWeights[1]")

ax[0][2].scatter(dWeights[2], losses)
ax[0][2].set_xlabel("dWeights[2]")

ax[0][3].scatter(dWeights[3], losses)
ax[0][3].set_xlabel("dWeights[3]")

ax[0][4].scatter(dWeights[4], losses)
ax[0][4].set_xlabel("dWeights[4]")

ax[0][5].scatter(dWeights[5], losses)
ax[0][5].set_xlabel("dWeights[5]")

ax[1][0].scatter(dH0s, losses)
ax[1][0].set_ylabel("loss")
ax[1][0].set_xlabel("dH0")

ax[1][1].scatter(dH1s, losses)
ax[1][1].set_xlabel("dH1")

ax[1][2].scatter(dO0s, losses)
ax[1][2].set_xlabel("dO0 (derivative_mse)")

ax[1][3].scatter(dBiases[0], losses)
ax[1][3].set_xlabel("dBiases[0]")

ax[1][4].scatter(dBiases[1], losses)
ax[1][4].set_xlabel("dBiases[1]")

ax[1][5].scatter(dBiases[2], losses)
ax[1][5].set_xlabel("dBiases[2]")

plt.show()
