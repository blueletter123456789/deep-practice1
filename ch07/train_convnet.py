import sys, os

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.pardir)
from common.trainer import Trainer
from simple_convnet import SimpleConvNet
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

x_train, t_train = x_train[:5000], t_train[:5000]
x_test, t_test = x_test[:1000], t_test[:1000]

max_epochs = 50

network = SimpleConvNet(input_dim=(1, 28, 28), conv_param={
    'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
    hidden_size=100, output_size=10, weight_std_init=0.01)

trainer = Trainer(network, x_train, t_train, x_test, t_test,
    epochs=max_epochs, mini_batch_size=100, optimizer='Adam', 
    optimizer_param={'lr': 0.001}, evaluate_sample_num_per_epoch=1000)

trainer.train()

network.save_params('params.pkl')
print('Saved Networks parameters!')

markers = {'train': 'o', 'test': 's'}
x = np.arange(max_epochs)
plt.plot(x, trainer.train_acc_list, marker='o', label='train', markevery=2)
plt.plot(x, trainer.test_acc_list, marker='s', label='test', markevery=2)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()
