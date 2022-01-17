from neural import NeuralNetwork as nn
from mnist import MNIST


mndata = MNIST('C:/Users/Lenovo/PycharmProjects/Network')
mndata.gz = True
images_train, labels_train = mndata.load_training()
images_test, labels_test = mndata.load_testing()
nn_structure = [784, 128, 11]
images_train = nn.normalize(images_train)
images_test = nn.normalize(images_test)
train = nn(nn_structure)
train.train_nn(images_train, labels_train, alpha=0.001, gamma=0.925, NUM_EPOCH=30, BATCH_SIZE=20, update_method="nesterov")
train.show_loss_after_train()
train.show_loss_validate()
train.test(images_test, labels_test)





