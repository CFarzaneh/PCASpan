import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy import linalg
from VAE_Models.VAE import VAE as model
from VAE_Models.architectures import DNN
import tensorflow as tf

def genData():
    dataset = np.ndarray((1000,3))

    for i in range(1000):
        theRange = 5
        x = np.random.uniform(-theRange, theRange)
        y = np.random.uniform(-theRange, theRange)
        z = (3*x)+(2*y) + np.random.uniform(-2, 2)
        u1 = [x,y,z]

        dataset[i] = u1

    return dataset

dataset = genData()

dataTrans = dataset.transpose()
cov = np.matmul(dataTrans, dataset)

w, v = linalg.eig(cov)

print(w)
print(v)

arr = [0,0,0]
v1 = np.concatenate((arr,v[:,0]*5))
v2 = np.concatenate((arr,v[:,1]*5))
v3 = np.concatenate((arr,v[:,2]*5))

fig = plt.figure()
ax = Axes3D(fig)

soa = np.array([v1,v2,v3])

X, Y, Z, U, V, W = zip(*soa)

ax.quiver(X, Y, Z, U, V, W, color='r')

ax.scatter(dataset[:,0], dataset[:,1], dataset[:,2])
plt.show()

print("\nRunning Linear Autoencoder with two dimensional latency space...\n")

input_dim = 3
latency_dim = 2
theEncoder = 3 # num neurons in each layer of encoder network
theDecoder = 3 # num neurons in each layer of generator network

batch_size=100
learning_rate=0.001

encoder = DNN(theEncoder, tf.nn.relu)
decoder = DNN(theDecoder, tf.nn.relu)

hyperParams = {'reconstruct_cost': 'gaussian',
			   'learning_rate': learning_rate,
			   'optimizer': tf.train.AdamOptimizer,
			   'batch_size': batch_size
			  }

autoEncoder = model(input_dim, encoder, latency_dim, decoder, hyperParams)
