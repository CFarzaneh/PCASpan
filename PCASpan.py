import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy import linalg
import torch
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm
import sys

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

Batch_Size = 1000
LearningRate = 0.002

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(3, 2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 3),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

autoencoder = AutoEncoder()
autoencoder.cuda()

optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LearningRate)
loss_func = nn.MSELoss()

for _ in tqdm(range(50)):
	for i in tqdm(range(10000)):
		x = torch.Tensor(genData(Batch_Size)).cuda()
		b_x = Variable(x.view(-1, 3)).cuda()
		b_y = Variable(x.view(-1, 3)).cuda()

		encoded, decoded = autoencoder(b_x)

		loss = loss_func(decoded, b_y)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	weights = autoencoder.encoder[0].weight.data.numpy()

	a1 = np.concatenate((arr,weights[0]*5))
	a2 = np.concatenate((arr,weights[1]*5))

	soaa = np.array([a1,a2])

	X, Y, Z, U, V, W = zip(*soaa)

	ax.quiver(X, Y, Z, U, V, W, color='g')

plt.show()


'''
input_dim = (3,1)
latency_dim = 2
theEncoder = [] # num neurons in each layer of encoder network
theDecoder = [] # num neurons in each layer of generator network

batch_size=1000
learning_rate=0.002

encoder = DNN(theEncoder)
decoder = DNN(theDecoder)

hyperParams = {'reconstruct_cost': 'gaussian',
                           'learning_rate': learning_rate,
                           'optimizer': tf.train.AdamOptimizer,
                           'batch_size': batch_size,
                           'variational': False
                          }

for j in tqdm(range(50)):
        autoEncoder = model(input_dim, encoder, latency_dim, decoder, hyperParams)

        for i in tqdm(range(10000)):
                data = genData(batch_size)
                #print("Data = ", data)
                cost, reconstr_loss, KL_loss = autoEncoder(data)

                print("output = ", autoEncoder.reconstruct(data))
                print("cost = ", cost)
                print("alpha = ", autoEncoder.alpha)
                print("reconstruction loss = ", reconstr_loss)
                print("KL_Loss = ", KL_loss)
                sys.exit()

        weights = autoEncoder.get_latent_weights()[0]
        weights = weights.transpose()
        print(weights)
        print(weights[0])

        a1 = np.concatenate((arr,weights[0]*5))
        a2 = np.concatenate((arr,weights[1]*5))
        
        soaa = np.array([a1,a2])

        X, Y, Z, U, V, W = zip(*soaa)

        ax.quiver(X, Y, Z, U, V, W, color='g')

        autoEncoder.reset()

plt.show()
'''
