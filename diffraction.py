import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j1
from functools import partial

# Airy pattern function
def airy_pattern(x, y, airy_x=1, airy_y=1, x0=0, y0=0, I0=1.0):
    r = np.sqrt(((x - x0) / airy_x) ** 2 + ((y - y0) / airy_y) ** 2)
    #r *= np.pi
    r *= 3.8317
    with np.errstate(divide='ignore', invalid='ignore'):
        z = np.where(r == 0, I0, I0 * (2 * j1(r) / r) ** 2)
    return z

# Function to convolve Airy patterns centered on (xdots, ydots)
def convolve_dots(x, y, xdots, ydots, pattern=airy_pattern, **pattern_kwargs):
    # Handle scalar inputs for x, y (single point) by making them arrays
    if np.isscalar(x) and np.isscalar(y):
        x = np.array([x])
        y = np.array([y])
    
    # Create a meshgrid for x, y if they are arrays
    xx, yy = np.meshgrid(x, y)
    
    # Initialize the result array for the final convolved pattern
    convolved_result = np.zeros_like(xx)
    
    # Loop over the dot positions and convolve the patterns
    for x0, y0 in zip(xdots, ydots):
        # Partial application of the airy pattern for each (xdot, ydot)
        local_pattern = partial(pattern, x0=x0, y0=y0, **pattern_kwargs)
        
        # Calculate the pattern and add it to the result
        convolved_result += local_pattern(xx, yy)
    
    # If the original x, y were scalars, return a single value instead of an array
    if np.isscalar(x) and np.isscalar(y):
        return convolved_result[0, 0]
    
    return convolved_result


#### faster version to be added #####


def airy_kernel_shifted(kernel_size, airy_x, airy_y, pixel, shift_x, shift_y, I0=1.0):
    """Generate an Airy kernel shifted by (shift_x, shift_y) pixels."""
    half = kernel_size // 2
    x = np.linspace(-half, half, kernel_size) * pixel
    y = np.linspace(-half, half, kernel_size) * pixel
    xx, yy = np.meshgrid(x, y)
    r = np.sqrt((xx / airy_x)**2 + (yy / airy_y)**2) * np.pi
    with np.errstate(divide='ignore', invalid='ignore'):
        kernel = np.where(r == 0, I0, I0 * (2 * j1(r) / r)**2)
    kernel /= kernel.sum()
    return shift(kernel, shift=(shift_y, shift_x), order=1, mode='constant', cval=0.0)


def convolve_dots_subpixel(
    x, y, xdots, ydots, airy_x, airy_y, pixel, kernel_size=15, I0=1.0
):
    H, W = len(y), len(x)
    psf = np.zeros((H, W), dtype=float)
    k_half = kernel_size // 2

    # Generate 4 subpixel-shifted kernels
    K00 = airy_kernel_shifted(kernel_size, airy_x, airy_y, pixel, 0.0, 0.0, I0)
    K10 = airy_kernel_shifted(kernel_size, airy_x, airy_y, pixel, 0.5, 0.0, I0)
    K01 = airy_kernel_shifted(kernel_size, airy_x, airy_y, pixel, 0.0, 0.5, I0)
    K11 = airy_kernel_shifted(kernel_size, airy_x, airy_y, pixel, 0.5, 0.5, I0)

    x0 = x[0]
    y0 = y[0]

    for x_dot, y_dot in zip(xdots, ydots):
        fx = (x_dot - x0) / pixel
        fy = (y_dot - y0) / pixel

        j = int(np.floor(fx))
        i = int(np.floor(fy))

        dx = fx - j  # fractional part in x
        dy = fy - i  # fractional part in y

        # Bilinear weights
        w00 = (1 - dx) * (1 - dy)
        w10 = dx * (1 - dy)
        w01 = (1 - dx) * dy
        w11 = dx * dy

        i_min = max(i - k_half, 0)
        i_max = min(i + k_half + 1, H)
        j_min = max(j - k_half, 0)
        j_max = min(j + k_half + 1, W)

        ki_min = k_half - (i - i_min)
        ki_max = k_half + (i_max - i)
        kj_min = k_half - (j - j_min)
        kj_max = k_half + (j_max - j)

        # Composite kernel via weighted sum
        stamp = (
            w00 * K00[ki_min:ki_max, kj_min:kj_max] +
            w10 * K10[ki_min:ki_max, kj_min:kj_max] +
            w01 * K01[ki_min:ki_max, kj_min:kj_max] +
            w11 * K11[ki_min:ki_max, kj_min:kj_max]
        )

        psf[i_min:i_max, j_min:j_max] += stamp

    return psf

if __name__ == "__main__":
# Set grid
    grid_size = 500
    x = np.linspace(-5, 5, grid_size)
    y = np.linspace(-5, 5, grid_size)
    X, Y = np.meshgrid(x, y)


# Define elliptical parameters
    airy_x = 1.0
    airy_y = 0.6  # Elliptical shape: squished in y-direction
    x0=1
    y0=2
# Compute elliptical Airy pattern with pi factor (standardized scaling)
    elliptical_pattern = airy_pattern(X, Y, airy_x, airy_y, x0=x0, y0=y0)

# Plot
    fig = plt.subplots(figsize=(10, 5))
    ax = plt.subplot(121)
    pcm = ax.pcolormesh(X, Y, elliptical_pattern, shading='auto', cmap='viridis')
    ax.set_aspect('equal')
    ax.set_title('Elliptical Airy Pattern (airy_x=1.0, airy_y=0.6)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

# Overlay elliptical Airy disk first dark ring (approximate unit circle in elliptical coordinates)
    theta = np.linspace(0, 2 * np.pi, 500)
    ellipse_x = airy_x * np.cos(theta) + x0
    ellipse_y = airy_y * np.sin(theta) + y0
    pcm = ax.plot(ellipse_x, ellipse_y, 'r--', label='Approx. First Dark Ring')

    #plt.colorbar(pcm, ax=ax, label='Intensity')
    ax.legend()

    ax = plt.subplot(122)
    xdots = [0, 0.5, 0.8]  # x positions of dots
    ydots = [0, 1, 0.2]  # y positions of dots
    convolved = convolve_dots(x, y, xdots, ydots, airy_x=1.5, airy_y=1.0, I0=1.0)

    pcm = ax.pcolormesh(X, Y, convolved, shading='auto', cmap='viridis')
    
#plt.colorbar(pcm, ax=ax, label='Intensity')
    ax.set_aspect('equal')
    plt.show()
