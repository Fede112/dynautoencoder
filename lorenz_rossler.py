import numpy as np
import matplotlib.pyplot as plt
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  
from matplotlib import animation
import sys
import id2nnal

from sklearn.metrics import pairwise_distances


def lorenz(x, y, z, s=10, r=28, b=2.667):
    '''
    Given:
       x, y, z: a point of interest in three dimensional space
       s, r, b: parameters defining the lorenz attractor
    Returns:
       x_dot, y_dot, z_dot: values of the lorenz attractor's partial
           derivatives at the point x, y, z
    '''
    # Set of equations
    x_dot = s*(y - x)
    y_dot = r*x - y - x*z
    z_dot = x*y - b*z
    
    return x_dot, y_dot, z_dot


def rossler(x, y, z, a=0.2, b=0.2, c=5.7):
    """
    Calculate the next coordinate X, Y, Z for 3rd-order Rossler system
    Parameters
    ----------
    x, y, z : float
        Input coordinates Z, Y, Z respectively
    
    """
    # Set of equations
    x_dot = -(y + z)
    y_dot = x + a * y
    z_dot = b + z * (x - c)

    return x_dot, y_dot, z_dot


def gaussian(x, y, mu_x, mu_y, sig_x=1, sig_y=1):
    return np.exp(    -( np.power(x - mu_x, 2.) / (2 * np.power(sig_x, 2.)) + \
            np.power(y - mu_y, 2.) / (2 * np.power(sig_y, 2.))  )   )





# -----------------------------------------------------------
# Frame parameters
# -----------------------------------------------------------

# size of image
pixels = 28
# rossler's x axis length
rx_axis = np.linspace(-10, 15, pixels)
# lorenz's x axis length
lx_axis = np.linspace(-25, 25, pixels)
# sigma of the gaussian. It defines the size of the particle.
rx_sig = (15 - (-10)) / pixels
lx_sig = (25 - (-25)) / pixels

print(f'rx_sig: {rx_sig}')
print(f'lx_sig: {lx_sig}')




# -----------------------------------------------------------
# System evolution
# -----------------------------------------------------------

# Integration parameters
dt = 0.01
num_steps = 500000

# Need one more for the initial values
lxs = np.empty(num_steps + 1)
lys = np.empty(num_steps + 1)
lzs = np.empty(num_steps + 1)
rxs = np.empty(num_steps + 1)
rys = np.empty(num_steps + 1)
rzs = np.empty(num_steps + 1)

# Set initial values
lxs[0], lys[0], lzs[0] = (0., 1., 1.05)
rxs[0], rys[0], rzs[0] = (0., 1., 1.05)

# Step through "time", calculating the partial derivatives at the current point
# and using them to estimate the next point
for i in range(num_steps):
    lx_dot, ly_dot, lz_dot = lorenz(lxs[i], lys[i], lzs[i])
    lxs[i + 1] = lxs[i] + (lx_dot * dt)
    lys[i + 1] = lys[i] + (ly_dot * dt)
    lzs[i + 1] = lzs[i] + (lz_dot * dt)
    rx_dot, ry_dot, rz_dot = rossler(rxs[i], rys[i], rzs[i])
    rxs[i + 1] = rxs[i] + (rx_dot * dt)
    rys[i + 1] = rys[i] + (ry_dot * dt)
    rzs[i + 1] = rzs[i] + (rz_dot * dt)


# Define mesh where we evaluate the particle dynamic
x, y = np.meshgrid(lx_axis, rx_axis)
x = x.flatten()
y = y.flatten()

# Frame definition
# frame = gaussian(x, y, 0, 0, lx_sig, rx_sig)




# -----------------------------------------------------------
# Animation/Plots
# -----------------------------------------------------------

## 2d/2d plot
# fig = plt.figure()
# plt.imshow(result.reshape(56,56))
# plt.show()

## 3d plot
# fig_3d = plt.figure()
# ax = fig_3d.gca(projection='3d')
# ax.plot(x, y, result, 'o')
# plt.show()

## Animation
# initialization function: plot the background of each frame
def init():
    a=gaussian(x, y, lxs[0], rxs[0], lx_sig, rx_sig)
    return [im]

# animation function.  This is called sequentially
def animate(i):
    a=im.get_array()
    # a=gaussian(x, y, lxs[i], rxs[i], lx_sig, rx_sig)
    a=gaussian(x, y, lxs[i], rxs[i], lx_sig, rx_sig)
    im.set_array(a.reshape(pixels,pixels))
    return [im]


# fig = plt.figure()
# a=gaussian(x, y, lxs[0], rxs[0], lx_sig, rx_sig)
# im=plt.imshow(a.reshape(pixels,pixels),interpolation='none')
# anim = animation.FuncAnimation(fig, animate, init_func=init,
#                                frames=7000, interval=5, blit=True)
# plt.show()




# -----------------------------------------------------------
# Creating dataset
# -----------------------------------------------------------

# Save frames as a matrix
dataset = []
sample_rate = 20        # every 20 iterations you sample the system
deltaT = 20             # takens time shift
num_observations = 5    # number of observations 
for i in range(0,300000,sample_rate):
    observations = gaussian(x, y, lxs[i], rxs[i], lx_sig, rx_sig)    
    for j in range(1,num_observations):
        observations = np.concatenate((observations, gaussian(x, y, lxs[i+j*deltaT], rxs[i+j*deltaT], lx_sig, rx_sig) ) )
    dataset.append(observations)
dataset = np.array(dataset)
print(dataset.shape)



# -----------------------------------------------------------
# Calculate intrinsic dimension
# -----------------------------------------------------------

# Calculate 
print('Building distance matrix...')
d_mat = pairwise_distances(dataset)
print('Finished building distance matrix')

print('Estimating intrinsic dimension:')
blocks_id, blocks_id_std, blocks_size = id2nnal.block_analysis(d_mat, fraction=.9)
# blocks_dim, blocks_dim_std, blocks_size = id2nn.two_nn_block_analysis(d_mat, frac=.9)

plt.plot(blocks_size, blocks_id, "r.-")
plt.errorbar(blocks_size, blocks_id, fmt = "r.-", yerr = np.array(blocks_id_std))
plt.show()
print(blocks_id[0])


