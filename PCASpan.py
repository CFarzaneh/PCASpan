import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy import linalg
from VAE_Models.VAE import VAE as model
from VAE_Models.architectures import DNN
import tensorflow as tf
from tqdm import tqdm

def genData(size):
    dataset = np.ndarray((size,3))

    for i in range(size):
        theRange = 5
        x = np.random.uniform(-theRange, theRange)
        y = np.random.uniform(-theRange, theRange)
        z = (3*x)+(2*y) + np.random.uniform(-2, 2)
        u1 = [x,y,z]

        dataset[i] = u1

    return dataset

dataset = genData(1000)

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

print("\nRunning Linear Autoencoder with two dimensional latency space...\n")

input_dim = (3,1)
latency_dim = 2
theEncoder = [] # num neurons in each layer of encoder network
theDecoder = [] # num neurons in each layer of generator network

batch_size=1000
learning_rate=0.001

encoder = DNN(theEncoder)
decoder = DNN(theDecoder)

hyperParams = {'reconstruct_cost': 'gaussian',
			   'learning_rate': learning_rate,
			   'optimizer': tf.train.AdamOptimizer,
			   'batch_size': batch_size,
			   'variational': False
			  }

autoEncoder = model(input_dim, encoder, latency_dim, decoder, hyperParams)

for i in tqdm(range(10000)):
	cost, reconstr_loss, KL_loss = autoEncoder(genData(batch_size))

weights = autoEncoder.get_latent_weights()[0]

print(weights)
print(weights[:,0])

a1 = np.concatenate((arr,weights[:,0]*25))
a2 = np.concatenate((arr,weights[:,1]*25))

soaa = np.array([a1,a2])

X, Y, Z, U, V, W = zip(*soaa)

ax.quiver(X, Y, Z, U, V, W, color='g')

plt.show()
