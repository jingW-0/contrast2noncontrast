import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# Generate a random patch (256x256)
patch = np.random.rand(256, 256) * 255
patch = patch.astype(np.uint8)

# Apply Gaussian filter with different sigma values
sigmas = [0.5, 1.5, 3, 4.5, 6]
fig, axes = plt.subplots(1, len(sigmas), figsize=(15, 3))

for ax, sigma in zip(axes, sigmas):
    filtered_patch = gaussian_filter((np.ones_like(patch) * 255).astype(np.uint8),  sigma=sigma)
    print(filtered_patch)
    ax.imshow(filtered_patch, cmap='gray')
    ax.set_title(f'Sigma = {sigma}')
    ax.axis('off')

plt.show()