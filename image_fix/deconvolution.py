
import matplotlib.pyplot as plt

from scipy.signal import convolve2d as conv2

from skimage import color, restoration

from PIL import Image
import numpy as np
import sys

def gaussuian_filter(kernel_size, sigma=1, muu=0):
 
    # Initializing value of x,y as grid of kernel size
    # in the range of kernel size
 
    x, y = np.meshgrid(np.linspace(-1, 1, kernel_size),
                       np.linspace(-1, 1, kernel_size))
    dst = np.sqrt(x**2+y**2)
 
    # Calculating Gaussian filter
    gauss = np.exp(-((dst-muu)**2 / (2.0 * sigma**2)))
    norm = np.sum(gauss)
    gauss /= norm

    return gauss


rng = np.random.default_rng()


im = Image.open('saturn.png')
orig = np.array(im)
orig = orig[:,:,0]
orig = orig / np.amax(orig)

#psf = np.ones((5, 5)) / 25
# Add Noise to Image
psf = gaussuian_filter(10, sigma=0.3)
print(psf.shape)

# Restore Image using Richardson-Lucy algorithm
deconvolved_RL = restoration.richardson_lucy(orig, psf)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 5))
plt.gray()

for a in (ax[0], ax[1]):
       a.axis('off')

ax[0].imshow(orig, cmap="gray")
ax[0].set_title('Original Data')

ax[1].imshow(deconvolved_RL, cmap="gray")
ax[1].set_title('Restoration using\nRichardson-Lucy')


fig.subplots_adjust(wspace=0.02, hspace=0.2,
                    top=0.9, bottom=0.05, left=0, right=1)
plt.show()

