import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j1
from scipy.interpolate import interp1d

# Compute theoretical diffraction-limited MTF
def analytic_mtf(f, fstop, wavelength):
    fc = 1 / (wavelength * fstop)
    f = np.asarray(f)
    mtf = np.zeros_like(f)
    mask = f < fc
    x = f[mask] / fc
    mtf[mask] = (2 / np.pi) * (np.arccos(x) - x * np.sqrt(1 - x**2))
    return mtf

# Generate Airy disk PSF
def airy_psf(grid_size, pixel_size_mm, wavelength_mm, fstop):
    N = grid_size
    x = (np.arange(N) - N // 2) * pixel_size_mm
    X, Y = np.meshgrid(x, x)
    R = np.sqrt(X**2 + Y**2)

    k = 2 * np.pi / wavelength_mm
    alpha = np.pi * R / (wavelength_mm * fstop)  # spatial frequency scaling
    alpha[alpha == 0] = 1e-10  # avoid divide-by-zero
    psf = (2 * j1(alpha) / alpha) ** 2
    return psf / psf.sum()

def airy_pattern(x, y, airy_x=1, airy_y=1, x0=0, y0=0, I0=1.0):
    r = np.sqrt(((x - x0) / airy_x) ** 2 + ((y - y0) / airy_y) ** 2)
    r *= np.pi
    with np.errstate(divide='ignore', invalid='ignore'):
        z = np.where(r == 0, I0, I0 * (2 * j1(r) / r) ** 2)
    return z


def airy_psf_with_limit(grid_size, s_limit_mm, wavelength_mm, fstop):
    N = grid_size
    pixel_size_mm = (2 * s_limit_mm) / N

    # Physical grid coordinates
    x = np.linspace(-s_limit_mm, s_limit_mm, N)
    y = np.linspace(-s_limit_mm, s_limit_mm, N)
    xx, yy = np.meshgrid(x, y)

    # Airy radius in mm
    airy_radius_mm = 1.22 * wavelength_mm * fstop

    # Use your function with Airy radius in mm
    psf = airy_pattern(xx, yy, airy_x=airy_radius_mm, airy_y=airy_radius_mm)
    return psf / psf.sum(), pixel_size_mm


# Compute 2D and radial MTF from PSF
def compute_mtf(psf, pixel_size_mm, radial=False):
    psf /= psf.sum()
    mtf2d = np.abs(np.fft.fftshift(np.fft.fft2(psf)))
    mtf2d /= mtf2d.max()

    N, M = psf.shape
    fx = np.fft.fftshift(np.fft.fftfreq(M, d=pixel_size_mm))
    fy = np.fft.fftshift(np.fft.fftfreq(N, d=pixel_size_mm))
    fx_grid, fy_grid = np.meshgrid(fx, fy)
    fr = np.sqrt(fx_grid**2 + fy_grid**2)

    # Radial average
    if radial:
        f_bins = np.linspace(0, fx.max(), 300)
        mtf_radial = np.zeros_like(f_bins)
        for i in range(len(f_bins)-1):
            mask = (fr >= f_bins[i]) & (fr < f_bins[i+1])
            if np.any(mask):
                mtf_radial[i] = mtf2d[mask].mean()
        f_centers = 0.5 * (f_bins[:-1] + f_bins[1:])
        return f_centers, mtf_radial[:-1]
    else:
        return fx, mtf2d[N//2, :]

# Main experiment
wavelength = 0.00055  # mm
pixel_size = 0.001    # mm (1 µm)
N = 1000
f_stops = [2.8, 4, 5.6, 8]

plt.figure(figsize=(10, 6))
for fnum in f_stops:

    psf, pixel_size = airy_psf_with_limit(N, s_limit_mm=0.1, wavelength_mm=wavelength, fstop=fnum)
    f_sim, mtf_sim = compute_mtf(psf, pixel_size, radial=True)

    f_theory = np.linspace(0, f_sim.max(), 1000)
    mtf_theory = analytic_mtf(f_theory, fnum, wavelength)

    plt.plot(f_sim, mtf_sim, label=f"f/{fnum} (sim)", lw=2)
    plt.plot(f_theory, mtf_theory, '--', label=f"f/{fnum} (theory)")

plt.xlim(0, 200)
plt.ylim(0, 1.05)
plt.grid(True)
plt.xlabel("Spatial Frequency [lp/mm]")
plt.ylabel("MTF")
plt.title("Diffraction-Limited MTF for Airy Disk (λ=550nm)")
plt.legend()
plt.tight_layout()
plt.show()

