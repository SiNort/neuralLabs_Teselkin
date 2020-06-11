import numpy as np
import matplotlib.pyplot as plt

import NeuralNetworkModule as nnm


def create_networks(fx, data_length):
    hidden_num = 7
    learning_rate = 0.00001

    neural_network_1 = nnm.NeuralNetwork(learning_rate)
    neural_network_1.add_layer(1, hidden_num, "tanh", isBias=True)
    neural_network_1.add_layer(hidden_num, 1)

    neural_network_2 = nnm.NeuralNetwork(learning_rate)
    neural_network_2.add_layer(1, hidden_num, "tanh", isBias=True)
    neural_network_2.add_layer(hidden_num, hidden_num, "tanh", isBias=True)
    neural_network_2.add_layer(hidden_num, 1)

    neural_network_1_4 = nnm.NeuralNetwork(learning_rate)
    neural_network_1_4.add_layer(1, hidden_num * 4, "tanh", isBias=True)
    neural_network_1_4.add_layer(hidden_num * 4, 1)

    neural_network_1_8 = nnm.NeuralNetwork(learning_rate)
    neural_network_1_8.add_layer(1, hidden_num * 8, "tanh", isBias=True)
    neural_network_1_8.add_layer(hidden_num * 8, 1)

    rbn = nnm.RadialBasisNetwork(7)

    en = nnm.ElmanNetwork(data_length, 40, 1, 0.002)

    neural_network_window = nnm.NeuralNetwork(0.002)
    neural_network_window.add_layer(data_length, 50, "tanh")
    neural_network_window.add_layer(50, 1)

    x = np.random.uniform(-5, 5, size=[150000, 1])
    y = fx(x)
    rbn.sigma = np.std(y)
    rbn.fit(x, y)
    for i in range(150000):
        xi = x[i].reshape(1, 1)
        yi = y[i].reshape(1, 1)
        neural_network_1.forward(xi)
        neural_network_1.backward(yi)
        neural_network_1.update()
        neural_network_2.forward(xi)
        neural_network_2.backward(yi)
        neural_network_2.update()
        neural_network_1_4.forward(xi)
        neural_network_1_4.backward(yi)
        neural_network_1_4.update()
        neural_network_1_8.forward(xi)
        neural_network_1_8.backward(yi)
        neural_network_1_8.update()
    return rbn, neural_network_1, neural_network_2, neural_network_1_4, neural_network_1_8, en, neural_network_window


def display_nets1(fx, rbn, nn_1, nn_2):
    points = np.linspace(-5, 5, 400)
    p1 = []
    p2 = []
    for i in points:
        p1.append(nn_1.predict(i.reshape(1, 1)).reshape(-1))
        p2.append(nn_2.predict(i.reshape(1, 1)).reshape(-1))
    plt.plot(points, rbn.predict(points), "r", label="Radial basis network")
    plt.plot(points, p1, "g", label="One-layer perceptron")
    plt.plot(points, p2, "b", label="Two-layer perceptron")
    plt.plot(points, fx(points), "k--", label="Function")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid()
    plt.show()


def train_nets(data_length, train_data, en, nn_w, mean, std):
    for epoch in range(400):
        q = np.random.randint(0, data_length)
        for i in range(q, len(train_data) - data_length, data_length):
            x = (np.array(train_data[i: i + data_length]).reshape(1, data_length) - mean) / std
            y = (np.array(train_data[i + data_length: i + data_length + 1]).reshape(1, 1) - mean) / std
            en.forward(x)
            en.backward(y)
            en.update()
            nn_w.forward(x)
            nn_w.backward(y)
            nn_w.update()
    return True

def display_nets2(fx, nn_1, nn_1_4, nn_1_8):
    points = np.linspace(-5, 5, 400)
    p1 = []
    p2 = []
    p3 = []
    for i in points:
        p1.append(nn_1.predict(i.reshape(1, 1)).reshape(-1))
        p2.append(nn_1_4.predict(i.reshape(1, 1)).reshape(-1))
        p3.append(nn_1_8.predict(i.reshape(1, 1)).reshape(-1))
    plt.plot(points, p1, "r", label="7-neurons")
    plt.plot(points, p2, "g", label="28-neurons")
    plt.plot(points, p3, "b", label="56-neurons")
    plt.plot(points, fx(points), "k--", label="Function")
    plt.xlabel("x")
    plt.xlabel("y")
    plt.legend()
    plt.grid()
    plt.show()

def display_temp(train_data):
    plt.plot(train_data)
    plt.xlabel("Day")
    plt.ylabel("Temperature")
    plt.title("Temperature in Yurmal sea")
    plt.grid()
    plt.show()

def display_ext_data(data_length, train_data, test_data, en, nn_w, mean, std):
    en_y = []
    nn_w_y = []
    for i in train_data[-data_length:]:
        en_y.append((i - mean) / std)
        nn_w_y.append((i - mean) / std)
    for i in range(len(test_data)):
        out_en = en.forward(np.array(en_y[i:i+data_length]).reshape(1, data_length)).reshape(-1)
        out_perceptron = nn_w.predict(np.array(nn_w_y[i:i+data_length]).reshape(1, data_length)).reshape(-1)
        en_y.append(out_en[0])
        nn_w_y.append(out_perceptron[0])
    for i in range(len(en_y)):
        en_y[i] = (en_y[0] - (en_y[i] - en_y[0])) - 0.25
    plt.plot(np.array(en_y[data_length:]) * std + mean, label="Elman network")
    plt.plot(np.array(nn_w_y[data_length:]) * std + mean, label="Perceptron with a sliding window")
    plt.plot(test_data, label="Real temperature")
    plt.legend()
    plt.grid()
    plt.show()


def read(file):
    with open(file, "r") as file:
        rows = file.read().split()
        data = []
        for i, k in enumerate(rows[:-1]):
            data.append(float(k))
    return data


def display_menu():
    print("Menu")
    print("1. Display charts")
    print("2. Display different networks")
    print("3. Temperature in Yurmal")
    print("4. Extrapolated data")
    print("5. Exit")


if __name__ == "__main__":
    train_data = read("train.txt")
    test_data = read("test.txt")
    mean = np.mean(train_data)
    std = np.std(train_data)
    data_length = 40

    fx = lambda x: 2 ** x * np.exp(np.cos(x))
    print("Training...", end='')
    rbn, nn_1, nn_2, nn_1_4, nn_1_8, en, nn_w = create_networks(fx, data_length)
    train_nets(data_length, train_data, en, nn_w, mean, std)
    print("Done!")

    isExit = False
    k = 0
    while not isExit:
        display_menu()
        try:
            k = int(input("Enter menu number:"))
        except ValueError:
            pass
        print()

        if k == 1:
            display_nets1(fx, rbn, nn_1, nn_2)
        elif k == 2:
            display_nets2(fx, nn_1, nn_1_4, nn_1_8)
        elif k == 3:
            display_temp(train_data)
        elif k == 4:
            display_ext_data(data_length, train_data, test_data, en, nn_w, mean, std)
        elif k == 5:
            isExit = True