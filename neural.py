import numpy as np
import random
from matplotlib import pyplot as plt
import sys


class NeuralNetwork:
    def __init__(self, nnStructure):
        """The list ``nnStructure`` contains the number of neurons in the
        respective layers of the network."""
        self.nnSize = len(nnStructure)
        self.nn_struct = nnStructure
        self.W, self.B = self.weights()
        self.total_E = list()
        self.E_valid = list()
        self.E_batch = list()

    def weights(self):
        """This method forms the biases and wights for network randomly
         using network structure
         :return weights and biases for each layer"""
        w = {}
        b = {}
        for l in range(self.nnSize-1):
            input = self.nn_struct[l]
            neurals_in_layer = self.nn_struct[l + 1]
            w[l + 1] = [np.random.uniform(-0.3, 0.3) for i in range(input * neurals_in_layer)]
            b[l + 1] = np.array([np.random.uniform(-0.3, 0.3) for i in range(neurals_in_layer)])
            w[l + 1] = np.array(w[l + 1]).reshape(self.nn_struct[l], self.nn_struct[l + 1])
        return w, b

    def convert_y_to_vect(self, y):
        """This method transform digit from ''y'' into vector of 0s and 1.
        The position of 1 - index which defines by digit from ''y''.
        For example: y = 1 -> y_vect = [0,1,0,0,0,0,0,0,0,0,0]
        :return list of 0s and 1"""
        y_vect = np.zeros((len(y), 11))
        for i, el in enumerate(y):
            y_vect[i, el] = 1
        if 1 not in y_vect:  # if 'y' isn't number in the range from 0 to 9
            y_vect[10] = None
        return y_vect

    def convert_y_to_vect_batch(self, y_batch):
        """This method transform the vector of digits ''y_batch''
         into vector with vectors of 0s and 1.
         :return list with lists of 0s and 1"""
        y_vect_batch = np.zeros((len(y_batch), 11))
        for i, yj in enumerate(y_batch):
            y_vect_batch[i, yj] = 1
        for i in range(len(y_batch)):  # if 'y_batch[j]' isn't number in the range from 0 to 9
            if 1 not in y_vect_batch[i]:
                y_vect_batch[i, 10] = None
        return y_vect_batch

    @staticmethod
    def train_test_split(data_x, data_y, valid_size):
        """
        This method gets data and splits it into training and testing sets.
        :param data_x: X_data
        :param data_y: Y_data
        :param train_size: size of training set from 0. till 1.
        :return: X_training list, X_testing list, Y_training list, Y_testing list
        """
        return data_x[:int(len(data_x) * (1-valid_size))], data_x[int(len(data_x) * (1-valid_size)):], \
               data_y[:int(len(data_y) * (1-valid_size))], data_y[int(len(data_y) * (1-valid_size)):]

    @staticmethod
    def normalize(images):
        """This method casts pixel values in the range 0 to 255 to the range 0 to 1"""
        for i, img in enumerate(images):
            images[i] = np.array(img) / 255
        return images

    def softmax(self, x):
        """The softmax function"""
        out = np.exp(x)
        return out / np.sum(out)

    def softmax_batch(self, x_batch):
        """The softmax function for batches"""
        out = np.exp(x_batch)
        return out / np.sum(out, axis=1, keepdims=True)

    def relu(self, x):
        """The relu function"""
        return np.maximum(x, 0)

    def df_relu(self, x):
        """Derivative of the relu function"""
        return (x >= 0).astype(float)

    def sparse_cross_entropy(self, h, y):
        """The coast function"""
        return -np.log(h[0, y])

    def sparse_cross_entropy_batch(self, h_out_batch, y_batch):
        """The coast function for batches"""
        return -np.log(np.array([h_out_batch[j, y_batch[j]] for j in range(len(y_batch))]))

    def feed_forward(self, x, y):
        """Returns inputs and outputs for each layer of network
        :param x: X_data
        :param y: Y_data"""
        h = {0: x}
        z = {}
        for l in range(len(self.nn_struct)-1):
            if l == 0:
                input_node = x
                z[l + 1] = np.dot(input_node, self. W[l + 1]) + self.B[l + 1]  # z1 = X*W1 + B1
                h[l + 1] = self.relu(z[l + 1])  # h1 = relu(z1)
            elif l == 1:
                input_node = h[l]
                z[l + 1] = np.dot(input_node, self.W[l + 1]) + self.B[l + 1]  # z2 = h1*W2 + B2
                h[l + 1] = self.softmax(z[l + 1])  # h2 = softmax(z2)
        E = self.sparse_cross_entropy(h[2], y)
        return h, z, E

    def feed_forward_batch(self, x, y):
        """Returns inputs per batch and outputs per batch for each layer of network
        :param x: X_data_batch
        :param y: Y_data_batch"""
        h = {0: x}
        z = {}
        for l in range(len(self.nn_struct) - 1):
            if l == 0:
                input_node = x
                z[l + 1] = np.dot(input_node, self.W[l + 1]) + self.B[l + 1]  # z1 = X*W1 + B1
                h[l + 1] = self.relu(z[l + 1])  # h1 = relu(z1)
            elif l == 1:
                input_node = h[l]
                z[l + 1] = np.dot(input_node, self.W[l + 1]) + self.B[l + 1]  # z2 = h1*W2 + B2
                h[l + 1] = self.softmax_batch(z[l + 1])  # h2 = softmax(z2)
        E = np.sum(self.sparse_cross_entropy_batch(h[2], y))
        return h, z, E

    def go_backward(self, y_vec, x, h, z):
        """Defines gradient of coast function for update weights and biases
        :return correction of wights and biases for each layer"""
        dE_dz2 = (h[2] - y_vec)  # 1x11
        dE_dw2 = np.dot(h[1].T, dE_dz2)  # 128x11
        dE_db2 = dE_dz2  # 1x11
        dE_dh1 = np.dot(dE_dz2, self.W[2].T)  # 1x128
        dE_dz1 = dE_dh1 * self.df_relu(z[1])  # 1x128
        dE_dw1 = np.dot(x.T, dE_dz1)  # 784x128
        dE_db1 = dE_dz1  # 1x128
        return dE_dw1, dE_db1, dE_dw2, dE_db2

    def go_backward_batch(self, y_vec, x, h, z):
        """Defines gradient of coast function per batch for update weights and biases
        :return correction of wights and biases for each layer"""
        dE_dz2 = (h[2] - y_vec)
        dE_dw2 = np.dot(h[1].T, dE_dz2)
        dE_db2 = np.sum(dE_dz2, axis=0, keepdims=True)
        dE_dh1 = np.dot(dE_dz2, self.W[2].T)
        dE_dz1 = dE_dh1 * self.df_relu(z[1])
        dE_dw1 = np.dot(x.T, dE_dz1)
        dE_db1 = np.sum(dE_dz1, axis=0, keepdims=True)
        return dE_dw1, dE_db1, dE_dw2, dE_db2

    def update_weights_gd(self, alpha, dataset, dataset_valid, NUM_EPOCH, BATCH_SIZE):
        """Updates wights and biases for each layer using gradient descent"""
        for j in range(NUM_EPOCH):
            random.shuffle(dataset)
            print("EPOCH:%s>" % (j + 1), end="")
            del self.E_batch[:]
            for i in range(len(dataset) // BATCH_SIZE):
                batch_x, batch_y, batch_y_vec = zip(*dataset[i * BATCH_SIZE: i * BATCH_SIZE + BATCH_SIZE])
                batch_x = np.concatenate(batch_x, axis=0)
                batch_y = np.array(batch_y)
                batch_y_vec = self.convert_y_to_vect_batch(batch_y)
                h, z, E = self.feed_forward_batch(batch_x, batch_y)
                self.E_batch.append(E)
                self.total_E.append(E)
                grad = self.go_backward_batch(batch_y_vec, batch_x, h, z)
                dE_dw1, dE_db1, dE_dw2, dE_db2 = grad
                self.W[1] = self.W[1] - alpha * dE_dw1
                self.W[2] = self.W[2] - alpha * dE_dw2
                self.B[1] = self.B[1] - alpha * dE_db1
                self.B[2] = self.B[2] - alpha * dE_db2
                if i % (len(dataset) // BATCH_SIZE/20) == 0:
                    print("=", end='')
                elif i == len(dataset) // BATCH_SIZE - 1:
                    print("> ", len(dataset) * (j + 1), "loss:%.4f" % (self.average_loss(self.E_batch)), end='')
                    self.validation(dataset_valid)

    def average_loss(self, loss):
        return np.sum(loss)/(len(loss))

    def update_weights_nesterov(self, alpha, gamma, dataset, dataset_valid, NUM_EPOCH, BATCH_SIZE):
        """Updates wights and biases for each layer using nesterov algorithm"""
        prev_v_w1 = np.zeros((784, 128))
        prev_v_b1 = np.zeros((1, 128))
        prev_v_w2 = np.zeros((128, 11))
        prev_v_b2 = np.zeros((1, 11))
        v_w = 0
        for j in range(NUM_EPOCH):
            del self.E_batch[:]
            dw1, db1 = np.zeros((784, 128)), np.zeros((1, 128))
            dw2, db2 = np.zeros((128, 11)), np.zeros((1, 11))
            v_w1 = gamma * prev_v_w1
            v_b1 = gamma * prev_v_b1
            v_w2 = gamma * prev_v_w2
            v_b2 = gamma * prev_v_b2
            random.shuffle(dataset)

            print("EPOCH:%s>" % (j + 1), end="")
            for i in range(len(dataset) // BATCH_SIZE):
                batch_x, batch_y, batch_y_vec = zip(*dataset[i * BATCH_SIZE: i * BATCH_SIZE + BATCH_SIZE])
                batch_x = np.concatenate(batch_x, axis=0)
                batch_y = np.array(batch_y)
                batch_y_vec = self.convert_y_to_vect_batch(batch_y)
                h, z, E = self.feed_forward_batch(batch_x, batch_y)
                self.E_batch.append(E)
                self.total_E.append(E)
                grad = self.go_backward_batch(batch_y_vec, batch_x, h, z)
                dE_dw1, dE_db1, dE_dw2, dE_db2 = grad
                dw2 = dE_dw2
                dw1 = dE_dw1
                db2 = dE_db2
                db1 = dE_db1
                v_w1 = gamma * prev_v_w1 + alpha * dw1
                v_b1 = gamma * prev_v_b1 + alpha * db1
                v_w2 = gamma * prev_v_w2 + alpha * dw2
                v_b2 = gamma * prev_v_b2 + alpha * db2
                prev_v_w1, prev_v_b1 = v_w1, v_b1
                prev_v_w2, prev_v_b2 = v_w2, v_b2
                self.W[1] = self.W[1] - v_w1
                self.W[2] = self.W[2] - v_w2
                self.B[1] = self.B[1] - v_b1
                self.B[2] = self.B[2] - v_b2
                if i % (len(dataset) // BATCH_SIZE/20) == 0:
                    print("=", end='')
                elif i == len(dataset) // BATCH_SIZE - 1:
                    print("> ", len(dataset) * (j + 1), "loss:%.4f" % (self.average_loss(self.E_batch)), end='')
                    self.validation(dataset_valid)

    def train_nn(self, X_train, Y_train, alpha=0.001, gamma=0.9, NUM_EPOCH=20, BATCH_SIZE=20, update_method="nesterov",
                 valid_size=0.2):
        """
        Trains the neural network using chosen method.
        :param X_train: X_train data
        :param Y_train: Y_train data
        :param alpha: coefficient for an update method
        :param gamma: coefficient for the nesterov method
        :param NUM_EPOCH: count of epoch
        :param BATCH_SIZE: batch size
        :param update_method: update method ('nesterov' or 'sgd')
        """
        print("TRAINING...")
        X_train, X_valid, Y_train, Y_valid = self.train_test_split(X_train, Y_train, valid_size)
        X_train = [i[np.newaxis] for i in X_train]
        y_v_train = self.convert_y_to_vect(Y_train)
        dataset = list(zip(X_train, Y_train, y_v_train))
        dataset_valid = list(zip(X_valid, Y_valid))
        if update_method.lower() == "gd":
            self.update_weights_gd(alpha, dataset,dataset_valid, NUM_EPOCH, BATCH_SIZE)
        elif update_method.lower() == "nesterov":
            self.update_weights_nesterov(alpha, gamma, dataset, dataset_valid, NUM_EPOCH, BATCH_SIZE)


    def show_loss_after_train(self):
        """This method shows dependence error per iteration after train"""
        plt.plot(self.total_E)
        plt.ylabel('E_train')
        plt.xlabel('Iter')
        plt.show()

    def show_loss_validate(self):
        """This method shows dependence error per iteration of validation data"""
        plt.plot(self.E_valid)
        plt.ylabel('E_valid')
        plt.xlabel('Iter')
        plt.show()

    def validation(self, dataset_valid):
        """This method testing how network have learned"""
        counter = 0
        random.shuffle(dataset_valid)
        del self.E_valid[:]
        for i in range(len(dataset_valid)):
            h_pr, z_pr, E_pr = self.feed_forward(dataset_valid[i][0], dataset_valid[i][1])
            index = h_pr[2].tolist()[0].index(np.max(h_pr[2].tolist()[0]))
            self.E_valid.append(E_pr)
            if index == dataset_valid[i][1]:
                counter += 1

        print(" - val_loss: %.4f" %(self.average_loss(self.E_valid)), end=" - ")
        print("val_accuracy:%.3f" % (counter/len(dataset_valid)*100) + "%")

    def test(self, X_test, Y_test):
        """This method testing how network have learned"""
        counter = 0
        progress_bar = 100
        dataset_test = list(zip(X_test, Y_test))
        print("TESTING...")
        for i in range(len(dataset_test)):
            h_pr, z_pr, E_pr = self.feed_forward(dataset_test[i][0], dataset_test[i][1])
            index = h_pr[2].tolist()[0].index(np.max(h_pr[2].tolist()[0]))
            if index == dataset_test[i][1]:
                counter += 1

            sys.stdout.write("\r %s" % (round(i/(len(dataset_test)/progress_bar)))+"%")
            sys.stdout.flush()
        print("\nPredicted:%s from %s" % (counter, len(Y_test)))
        print("Accuracy:%.3f" % (counter/len(Y_test)*100) + "%")
