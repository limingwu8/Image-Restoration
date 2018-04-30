import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from scipy.interpolate import interp1d
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from PIL import Image
from skimage.io import imread
from skimage.color import rgb2gray
import random

save_info = 1

def get_shift_value(num):
    deform_fun = bool(random.getrandbits(1))
    deform_fun = 6
    if deform_fun == 0:
        # generate random parameters
        a,b,c,d = np.random.uniform(-10, 10, 4)
        a,b,c,d = -5, 5, -5, 5
        x = np.linspace(a, b, num)
        y = np.linspace(c, d, num)
        x, y = np.meshgrid(x,y)
        z = np.multiply(y, np.sin(x)) - np.multiply(x,np.cos(y))
        # plot_surface(x,y,z)
        z = z*2
    elif deform_fun == 1:
        # generate random parameters
        a, b, c, d = np.random.uniform(-10, 10, 4)
        a, b, c, d = -5, 5, -5, 5
        x = np.linspace(a, b, num)
        y = np.linspace(c, d, num)
        x, y = np.meshgrid(x, y)
        z = np.multiply(y, np.sin(y)) - np.multiply(x, np.cos(x))
        # plot_surface(x,y,z)
        z = z * 2
    elif deform_fun == 2:
        # generate random parameters
        a, b, c, d = np.random.uniform(-10, 10, 4)
        a, b, c, d = -5, 5, -5, 5
        x = np.linspace(a, b, num)
        y = np.linspace(c, d, num)
        x, y = np.meshgrid(x, y)
        z = np.multiply(y, np.sin(x)) + np.multiply(x, np.cos(y))
        # plot_surface(x,y,z)
        z = z * 2
    elif deform_fun == 3:
        # generate random parameters
        a, b, c, d = np.random.uniform(-10, 10, 4)
        a, b, c, d = -5, 5, -5, 5
        x = np.linspace(a, b, num)
        y = np.linspace(c, d, num)
        x, y = np.meshgrid(x, y)
        z = np.multiply(y, np.sin(y)) + np.multiply(x, np.cos(x))
        # plot_surface(x,y,z)
        z = z * 2
    elif deform_fun == 4:
        # generate random parameters
        a, b, c, d = np.random.uniform(-10, 10, 4)
        a, b, c, d = -5, 5, -5, 5
        x = np.linspace(a, b, num)
        y = np.linspace(c, d, num)
        x, y = np.meshgrid(x, y)
        z = np.multiply(y, np.sin(y)) + np.multiply(x, np.cos(x)) + np.multiply(y, np.sin(x)) + np.multiply(x, np.cos(y))
        # plot_surface(x,y,z)
        z = z * 2
    elif deform_fun == 5:
        # Generate 2D Gaussian like matrix
        x, y = np.meshgrid(np.linspace(-1, 1, num), np.linspace(-1, 1, num))
        d = np.sqrt(x * x + y * y)
        sigma, mu = 1.0, 0.0
        z = 100*(np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2))) - 0.5)
    elif deform_fun == 6:
        # Generate multi-Gaussian like matrix
        x, y = np.meshgrid(np.linspace(-15, 15, num), np.linspace(-15, 15, num))
        z = np.zeros(x.shape)
        for i in range(4):
            mu = (np.random.uniform(-10, 10), np.random.uniform(-10, 10))
            sigma = (np.random.uniform(2, 10), np.random.uniform(2, 10))
            z += 10*(np.exp(-(np.power(x-mu[0],2)/(2*sigma[0]**2) + np.power(y-mu[1],2)/(2*sigma[1]**2))) - 0.2)
        # plot_surface(x, y, z)
    else:
        N = (num, num)
        F = 1
        x = np.linspace(0, num, num)
        y = np.linspace(0, num, num)
        x, y = np.meshgrid(x, y)
        i = np.minimum(x - 1, N[0] - x + 1)
        j = np.minimum(y - 1, N[1] - y + 1)
        H = np.exp(-.5*(np.power(i, 2) + np.power(j, 2))/(F**2))
        z = 10*np.real(np.fft.ifft2(np.multiply(H, np.fft.fft2(np.random.randn(N[0], N[1])))))
        # plot_surface(x,y,z)
    return z

def read_img(path):
    # Cannot convert png to L directly, otherwise the background will be filled with weird color
    # Instead, should add a black background to png first and convert to L
    img = Image.open(path).convert('L')
    return np.array(img)

def shift(slice, shift_values):
    """
    Perform a shift on the slice
    :param slice: Dictionary, keys are the position of pixel, values are the pixel value
    :param shift: Array, the shifting distance for each pixel
    :return: Dictionary, shifted slice, keys are the new position of pixel, values are the new pixel value
    """
    for i in range(0, len(shift_values)):
        img_position = i
        shift_value = shift_values[i]
        if shift_value==0:
            continue
        img_new_position = img_position + shift_value
        if img_new_position in slice.keys():
            if slice[img_new_position].astype(np.uint16) + slice[img_position].astype(np.uint16) <= 255:
                slice[img_new_position] += slice[img_position]
            else:
                # slice[img_new_position] = np.uint8(255)
                slice[img_new_position] += slice[img_position]
            del slice[img_position]
        else:
            img_new_value = slice[img_position]
            del slice[img_position]
            slice[img_new_position] = img_new_value
    return slice

def interpolate(slice, size):
    x = list(slice.keys())
    x.sort()
    x = np.array(x)
    y = [slice[key] for key in slice.keys()]
    y = np.array(y)
    f = interp1d(x, y, bounds_error=False, fill_value=0)
    x_new = np.array(range(0, size))
    y_new = f(x_new)

    return y_new

def deforme(img):
    img = img.astype(np.int16)
    width = img.shape[0]
    new_img = np.zeros(img.shape, dtype=np.int16)
    shift_matrix = get_shift_value(width)
    shift_matrix = np.round(shift_matrix, 2)
    for i in range(0,width):
        # get one slice from img
        # slice_value = img[i].astype(np.uint16)
        slice_value = img[i]
        # save slice as dictionary, key: index, value: pixel value
        slice = dict(zip(range(len(slice_value)), slice_value))
        # sample shifts randomly
        shift_values = shift_matrix[:,i]
        shifted_slice = shift(slice, shift_values)

        new_slice = interpolate(shifted_slice, width)
        new_img[i] = new_slice
    return new_img

def plot_surface(x,y,z):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    if save_info:
        fig.tight_layout()
        fig.savefig('./gaussian'+ str(idx) + '.png')
    plt.show()

def plot_deformed():
    fig = plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.imshow(new_img, cmap='gray')
    if save_info:
        fig.tight_layout()
        fig.savefig('./deformed'+ str(idx) + '.png')
    plt.show()

if __name__ == '__main__':
    for idx in range(3):
        img = read_img('./test4.png')
        new_img = deforme(img)
        plot_deformed()