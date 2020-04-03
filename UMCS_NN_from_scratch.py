import numpy as np

class NeuralNetwork():
    def __init__(self):
        self.weigths = np.random.rand(6) #zamien na np.random.normal(size=6) -> normal oznacza ze ma byc rozklad normalny w losowaniu bo zwykly ran daje rownomierny rozklad -> kiedys wytlumacze ock
        self.bias = np.random.rand(3)

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

                dWeigths = np.zeros((6))
                #Stworz sobie podobnie dBias bo to co liczysz te biasy to sa delty na biasach - wagi i biasy aktualizujemy forward w kolejnym kroku

                # Deltas for weigths and biases

                # to nie jest self.bias[2] tylko dBias[2]                
                self.bias[2] = self.sigmoid(O0_weighted_sum, derivative=True)
                dWeigths[5] = H1 * self.bias[2]
                dWeigths[4] = H0 * self.bias[2]
                
                #tutaj dWeights zamien na Weights -> ten blad co wyslalem na grupie
                dH0 = dWeigths[4] * self.bias[2]
                dH1 = dWeigths[5] * self.bias[2]
                
                # U ciebie sa inaczej wagi polaczone -> waga 0 i 2 jest polaczona do H0, a 1 i 3 do H1 stad zamienile 0,1 na 0,2 oraz 2,3 na 1,3
                self.bias[0] = self.sigmoid(H0_weighted_sum, derivative=True)
                dWeigths[0] = X_values[0] * self.bias[0]
                dWeigths[2] = X_values[1] * self.bias[0]

                self.bias[1] = self.sigmoid(H1_weighted_sum, derivative=True)
                dWeigths[1] = X_values[0] * self.bias[1]
                dWeigths[3] = X_values[1] * self.bias[1]
                # Weigths correction


                #Zaktualizuj podobnie biasy wedlug dBias

                self.weigths[0] -= lr * dWeigths[0] * dH0 * derivative_mse
                self.weigths[1] -= lr * dWeigths[1] * dH1 * derivative_mse
                self.weigths[2] -= lr * dWeigths[2] * dH0 * derivative_mse
                self.weigths[3] -= lr * dWeigths[3] * dH1 * derivative_mse
                
                self.weigths[4] -= lr * dWeigths[4] * derivative_mse
                self.weigths[5] -= lr * dWeigths[5] * derivative_mse

                if epoch % 100 == 0:
                    result = [self.predict(x) for x in X_train]
                    print(f'Loss {self.mse_loss(y_train, result):1.4f}\n{result}')

            #if epoch % 100 == 0:
                #print(f'Loss {self.mse_loss(y_train, result):1.4f}\n{result}')


X_train = np.array([[0, 0],
                    [0, 1],
                    [1, 0],
                    [1, 1]])

y_train = np.array([0, 0, 0, 1])

nn = NeuralNetwork()

nn.train(X_train, y_train)
