"""
Multilayer Perceptron (MLP) python code written only using numpy library, created for "Fundamental of Intelligent System" Class, BME.
Code consist of simple feedforward and backpropagation algorithm. Activation function done by using sigmoid function.
Code implemented for XOR logic, able to train the weight, theta (bias), and delta value.
(c) AS. - 2024
"""

import numpy as np


class main_mlp:
    def __init__(self):
        self.input = [[0, 0], [0, 1], [1, 0], [1, 1]]  # define input value
        self.output = [0, 1, 1, 0]  # define output value
        n_in, n_hidden, n_out = 2, 4, 1
        self.n_node = [n_in, n_hidden, n_out]  # number of nodes each layer (input, hidden, output)
        self.weight_1, self.weight_2 = np.zeros((self.n_node[1], self.n_node[0])), np.zeros(
            (self.n_node[2], self.n_node[1]))
        self.theta_1, self.theta_2 = np.zeros(self.n_node[1]), np.zeros(self.n_node[2])
        for j in range(self.n_node[1]):
            for i in range(self.n_node[0]):
                self.weight_1[j][i] = np.random.uniform(0, 0.5)     # define randomized weight
                self.theta_1[j] = np.random.uniform(0, 0.5)         # define bias
        for k in range(self.n_node[2]):
            for j in range(self.n_node[1]):
                self.weight_2[k][j] = np.random.uniform(0, 0.5)     # define randomized weight
                self.theta_2[k] = np.random.uniform(0, 0.5)         # define bias

        self.v_1, self.y_1, self.g_1, self.delta_1 = [np.zeros(self.n_node[1]) for _ in range(4)]
        self.v_2, self.y_2, self.g_2, self.delta_2 = [np.zeros(self.n_node[2]) for _ in range(4)]

        self.error = 1      # define current error
        self.tolerance = 0.0001    # define threshold for error value

        self.alpha = 0.5    # regulation strength
        self.miu = 1   # learning rate

        self.iteration = 0  # number of iteration

        self.mlp_process()

    def mlp_process(self):
        while np.abs(self.error) >= self.tolerance:     # code will run until error < tolerance
            self.error = 0  # reset

            for dat in range(len(self.input)):

                # feedforward algorithm
                for j in range(self.n_node[1]):
                    temp = 0
                    for i in range(self.n_node[0]):
                        temp += self.input[dat][i] * self.weight_1[j][i]
                    self.v_1[j] = temp + self.theta_1[j]  # compute weight summation
                    self.y_1[j] = 1 / (1 + np.exp(
                        -1 * self.alpha * self.v_1[j]))  # compute output using unipolar sigmoid function
                    self.g_1[j] = self.alpha * self.y_1[j] * (1 - self.y_1[j])  # compute the derivative of y

                for k in range(self.n_node[2]):
                    temp = 0
                    for j in range(self.n_node[1]):
                        temp += self.y_1[j] * self.weight_2[k][j]
                    self.v_2[k] = temp + self.theta_2[k]  # compute weight summation
                    self.y_2[k] = 1 / (1 + np.exp(
                        -1 * self.alpha * self.v_2[k]))  # compute output using unipolar sigmoid function
                    self.g_2[k] = self.alpha * self.y_2[k] * (1 - self.y_2[k])  # compute the derivative of y

                # compute error
                for k in range(self.n_node[2]):
                    self.error += (self.output[dat] - self.y_2[k]) ** 2

                # backpropagation algorithm
                for k in range(self.n_node[2]):
                    self.delta_2[k] = (self.output[dat] - self.y_2[k]) * self.g_2[k]    # compute delta

                for j in range(self.n_node[1]):
                    temp = 0
                    for k in range(self.n_node[2]):
                        temp += self.delta_2[k] * self.weight_2[k][j]
                    self.delta_1[j] = temp * self.g_1[j]        # compute delta

                for k in range(self.n_node[2]):
                    for j in range(self.n_node[1]):
                        self.weight_2[k][j] += self.miu * self.delta_2[k] * self.y_1[j]     # update weight
                    self.theta_2[k] += self.miu * self.delta_2[k]   # update bias each node

                for j in range(self.n_node[1]):
                    for i in range(self.n_node[0]):
                        self.weight_1[j][i] += self.miu * self.delta_1[j] * self.input[dat][i]  # update weight
                    self.theta_1[j] += self.miu * self.delta_1[j]   # update bias each node

            self.iteration += 1
            print(self.iteration, self.error)


if __name__ == '__main__':
    main_mlp()
