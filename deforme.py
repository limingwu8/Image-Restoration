import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from scipy.interpolate import interp1d
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from PIL import Image


def get_shift_value(num):
    # generate random parameters
    a,b,c,d = np.random.uniform(-10, 10, 4)
    x = np.linspace(a, b, num)
    y = np.linspace(c, d, num)
    x, y = np.meshgrid(x,y)
    z = np.multiply(y, np.sin(x)) - np.multiply(x,np.cos(y))
    # plot_surface(x,y,z)
    z = z*2
    return z

def read_img(path):
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
                slice[img_new_position] = np.uint8(255)
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
    width = img.shape[0]
    new_img = np.zeros(img.shape)
    shift_matrix = get_shift_value(width)
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
    # plt.show()

if __name__ == '__main__':
    img = read_img('./circle.png')
    plt.subplot(1,2,1)
    plt.imshow(img, cmap='gray')
    new_img = deforme(img)
    plt.subplot(1,2,2)
    plt.imshow(new_img, cmap='gray')
    plt.show()