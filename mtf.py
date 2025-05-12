import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from diffraction import *
from plottools import *
from lenspy3d import *

def compute_mtf(psf, pixel_size_mm=None, plot=False, radial=False):
    """
    Compute the 2D MTF, X/Y cuts, and optionally the radial profile.
    
    Parameters:
    - psf: 2D numpy array, point spread function
    - pixel_size_mm: float or None, size of one pixel in mm (for frequency units)
    - plot: bool, whether to show plots
    - radial: bool, whether to compute radial profile
    
    Returns:
    - mtf2d: 2D numpy array, MTF magnitude
    - fx, fy: 1D arrays of spatial frequencies (same size as axes)
    - mtf_x, mtf_y: 1D MTF cuts along X and Y through center
    - mtf_x_interp, mtf_y_interp: functions to get interpolated MTF along X and Y directions at any frequency
    - radial_profile: 1D radial MTF profile (or None)
    """
    #psf = psf.astype(np.float32)
    psf /= psf.sum()  # Normalize energy
    
    # Compute MTF
    mtf2d = np.abs(np.fft.fftshift(np.fft.fft2(psf)))
    mtf2d /= mtf2d.max()

    N, M = psf.shape
    if pixel_size_mm is None:
        # Frequencies in cycles/pixel
        fx = np.fft.fftshift(np.fft.fftfreq(M))
        fy = np.fft.fftshift(np.fft.fftfreq(N))
        freq_unit = "cycles/pixel"
    else:
        fx = np.fft.fftshift(np.fft.fftfreq(M, d=pixel_size_mm))
        fy = np.fft.fftshift(np.fft.fftfreq(N, d=pixel_size_mm))
        freq_unit = "cycles/mm"

    # Cuts along center row/column
    mtf_x = mtf2d[N//2, :]
    mtf_y = mtf2d[:, M//2]

    # Interpolate MTF for X and Y directions
    mtf_x_interp = interp1d(fx, mtf_x, bounds_error=False, fill_value=0)
    mtf_y_interp = interp1d(fy, mtf_y, bounds_error=False, fill_value=0)

    # Radial profile
    radial_profile = None
    if radial:
        y, x = np.indices(mtf2d.shape)
        center = np.array(mtf2d.shape) // 2
        r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
        r = r.astype(np.int32)
        tbin = np.bincount(r.ravel(), mtf2d.ravel())
        nr = np.bincount(r.ravel())
        radial_profile = tbin / np.maximum(nr, 1)

    if plot:
        fig, axs = plt.subplots(1, 3 if radial else 2, figsize=(15, 4))

        axs[0].imshow(mtf2d, extent=[fx[0], fx[-1], fy[0], fy[-1]], origin='lower', cmap='viridis')
        axs[0].set_title("2D MTF")
        axs[0].set_xlabel(freq_unit)
        axs[0].set_ylabel(freq_unit)

        axs[1].plot(fx, mtf_x, label="X direction")
        axs[1].plot(fy, mtf_y, label="Y direction")
        axs[1].set_title("MTF Cuts")
        axs[1].set_xlabel(freq_unit)
        axs[1].set_ylabel("MTF")
        axs[1].legend()
        axs[1].grid(True)

        if radial:
            radial_freqs = np.linspace(0, max(fx.max(), fy.max()), len(radial_profile))
            axs[2].plot(radial_freqs, radial_profile)
            axs[2].set_title("Radial MTF")
            axs[2].set_xlabel(freq_unit)
            axs[2].set_ylabel("MTF")
            axs[2].grid(True)

        plt.tight_layout()
        plt.show()

    return mtf2d, fx, fy, mtf_x, mtf_y, mtf_x_interp, mtf_y_interp, (radial_profile if radial else None)


import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline

def analyze_mtf_across_field(
    results,
    wls,
    weights,
    x, y,
    pixel,
    airy_radius_by_wl=None,
    smooth = False,
    freq_samples=[10, 20, 40, 80, 160],
    convolver_fn=convolve_dots,
    clist=clist0,
    lw=1.,
    facecolor='black',
    tick_color='white',
    font_color='white',
):

    grid_size = len(x)
    r_targets = sorted([float(k) for k in results])
    r_labels = [f"{r:.2f}" for r in r_targets]

    mtf_x_at_freqs = {f: [] for f in freq_samples}
    mtf_y_at_freqs = {f: [] for f in freq_samples}

    for k_str in r_labels:
        print(f'Processing r = {k_str} mm')
        psf_total = np.zeros((grid_size, grid_size), dtype=float)
        psf_dict = {}
        for wl in wls:
            coords = results[k_str]['points'][wl]
            xdots = coords[:, 0] - results[k_str]['xmean']
            ydots = coords[:, 1] - results[k_str]['ymean']
            
            if airy_radius_by_wl:
                airy_x = airy_radius_by_wl[wl]
                airy_y = airy_radius_by_wl[wl]
            else:
                airy_x = results[k_str]['airy'][wl][0]
                airy_y = results[k_str]['airy'][wl][1]
            
            psf_wl = convolver_fn(x, y, xdots, ydots,
                                  airy_x=airy_x, airy_y=airy_y,
                                  I0=1.0)
            psf_dict[wl] = psf_wl

            psf_total += weights[wl] * psf_wl

        psf_total /= psf_total.sum()
        
        psf_dict['white'] = psf_total

        # Store PSFs in results
        results[k_str]['psf'] = psf_dict

        # Compute MTF
        mtf2d, fx, fy, mtf_x, mtf_y, mtf_x_interp, mtf_y_interp, _ = compute_mtf(
            psf_total, pixel_size_mm=pixel, plot=False
        )

        for f in freq_samples:
            mtf_x_at_freqs[f].append(mtf_x_interp(f))
            mtf_y_at_freqs[f].append(mtf_y_interp(f))

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 5.5), facecolor=facecolor)
    ax.set_facecolor(facecolor)
    ax.tick_params(colors=tick_color)
    for spine in ax.spines.values():
        spine.set_color(tick_color)

    ax.set_xlabel('Field Radius (mm)', color=font_color)
    ax.set_ylabel('MTF', color=font_color)
    ax.set_xlim(0, max(r_targets))
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.grid(True, color='gray', alpha=0.3)

    if clist is None:
        clist = plt.cm.viridis(np.linspace(0, 1, len(freq_samples)))

    handles = []
    labels = []

    for idx, f in enumerate(freq_samples):
        color = clist[idx % len(clist)]

        # Plot MTF-X (dot-dashed)
        if smooth:
            cs = CubicSpline(r_targets, mtf_x_at_freqs[f])
            r_fine = np.linspace(0, max(r_targets), 1000)
            mtf_x_fine = cs(r_fine)
            line_x, = ax.plot(r_fine, mtf_x_fine, lw=lw, color=color, ls='-.')
        else:
            line_x, = ax.plot(r_targets, mtf_x_at_freqs[f], lw=lw, color=color, ls='-.')

        # Add "S" marker
        ax.text(r_targets[-1] + 0.5, mtf_x_at_freqs[f][-1], f"S{idx+1}", va='center', ha='left',
                color=color, fontsize=8)

        # Plot MTF-Y (solid)
        if smooth:
            cs = CubicSpline(r_targets, mtf_y_at_freqs[f])
            r_fine = np.linspace(0, max(r_targets), 1000)
            mtf_y_fine = cs(r_fine)
            line_y, = ax.plot(r_fine, mtf_y_fine, lw=lw, color=color, ls='-')
        else:
            line_y, = ax.plot(r_targets, mtf_y_at_freqs[f], lw=lw, color=color, ls='-', label=f'{f} lp/mm (X)')

        handles.append(line_y)
        labels.append(f'{f} lp/mm')
            
        s_y = mtf_x_at_freqs[f][-1]
        t_y = mtf_y_at_freqs[f][-1]

        # Check vertical distance
        if abs(s_y - t_y) < 0.03:  # Threshold to trigger offset
            offset = 0.015
            if s_y > t_y:
                s_y += offset
                t_y -= offset
            else:
                s_y -= offset
                t_y += offset

        # Add "S" marker (MTF-X)
        ax.text(r_targets[-1] + 0.5, s_y, f"S{idx+1}", va='center', ha='left',
                color=color, fontsize=8)


        # Add "T" marker (MTF-Y)
        ax.text(r_targets[-1] + 0.5, t_y, f"T{idx+1}", va='center', ha='left',
                color=color, fontsize=8)
    # Custom legend below the plot, only for MTF-X
    leg = ax.legend(handles, labels, 
                    #title="MTF-X Frequencies", title_fontsize=9,
                    bbox_to_anchor=(0.5, -0.1), loc="upper center", ncol=5,
                    framealpha=0.3, facecolor=facecolor, frameon=False)

    for text, line in zip(leg.get_texts(), handles):
        text.set_color(line.get_color())

    plt.tight_layout()
    plt.show()

    return r_targets, mtf_x_at_freqs, mtf_y_at_freqs

def plot_results_psfs_grid(results, wls, weights, colors, extent=None, normalize=True):
    """
    Plots additive colored and white PSFs for multiple results[key] entries in 2x3 grids.

    Parameters:
    - results: dict with structure results[key]['psf'][wl] and ['psf_white']
    - wls: list of wavelengths (e.g., ['g', 'F', 'e', 'd', 'C'])
    - weights: dict mapping wl -> weight
    - colors: dict mapping wl -> RGB color (e.g. [1,0,0] for red)
    - extent: [x0, x1, y0, y1] for image coordinates (optional)
    - normalize: whether to normalize each PSF before display
    """
    keys = sorted(results.keys(), key=lambda k: float(k))
    N = len(keys)
    if N > 6:
        indices = np.round(np.linspace(0, N - 1, 6)).astype(int)
        keys_to_plot = [keys[i] for i in indices]
    else:
        keys_to_plot = keys

    fig_color, axs_color = plt.subplots(2, 3, figsize=(12, 8))
    fig_white, axs_white = plt.subplots(2, 3, figsize=(12, 8))

    axs_color = axs_color.flatten()
    axs_white = axs_white.flatten()

    for i, key in enumerate(keys_to_plot):
        axc = axs_color[i]
        axw = axs_white[i]

        # --- Colored PSF ---
        canvas = AdditivePSFCanvas(axc)
        for wl in wls:
            psf = results[key]['psf'][wl]
            canvas.add_array(psf, color=colors[wl], extent=extent, normalize=normalize)
        canvas.render()
        axc.set_title(f'r = {key}')
        #axc.set_xticks([])
        #axc.set_yticks([])

        # --- White PSF ---
        psf_white = results[key]['psf']['white']
        if normalize:
            psf_white = psf_white / psf_white.max()
        if extent is None:
            h, w = psf_white.shape
            extent = [-w/2, w/2, -h/2, h/2]  # fallback
        axw.imshow(psf_white, cmap='gray', extent=extent, origin='lower')
        axw.set_title(f'r = {key}')
        #axw.set_xticks([])
        #axw.set_yticks([])

    # Hide unused subplots
    for i in range(len(keys_to_plot), 6):
        fig_color.delaxes(axs_color[i])
        fig_white.delaxes(axs_white[i])

    fig_color.suptitle("Additive Colored PSFs", color='white')
    fig_white.suptitle("White PSFs", color='black')

    fig_color.tight_layout()
    fig_white.tight_layout()

    return fig_color, fig_white


def get_weights(wls, plot=False, eye_only=False, d65_only=False):
    
    # Load CIE 1931 photopic luminosity function (Y values)
    cie_data = np.loadtxt('glass/CIE_sle_photopic.csv', delimiter=',', skiprows=0)  # Adjust skiprows if necessary
    cie_interp = interp1d(cie_data[:, 0], cie_data[:, 1], kind='linear', bounds_error=False, fill_value=0.0)

    # Load D65 spectral power distribution (SPD)
    d65_data = np.loadtxt('glass/CIE_std_illum_D65.csv', delimiter=',', skiprows=0)  # Adjust skiprows if necessary
    d65_interp = interp1d(d65_data[:, 0], d65_data[:, 1], kind='linear', bounds_error=False, fill_value=0.0)

    raw_weights = {}

    for line in wls:
        wl = FRAUNHOFER_WAVELENGTHS.get(line)

        if wl is None:
            wl = FRAUNHOFER_WAVELENGTHS.get(line)

        S = d65_interp(wl)
        V = cie_interp(wl)
        raw_weights[line] = S * V 

        if eye_only:
            raw_weights[line] = V
        if d65_only:
            raw_weights[line] = S

    total = sum(raw_weights.values())
    if total == 0:
        raise ValueError("Total weight is zero; check input/interpolators.")

    normalized_weights = {line: val / total for line, val in raw_weights.items()}


    if plot:
        wavelengths = np.linspace(380, 780, 1000)
        S_vals = d65_interp(wavelengths)
        V_vals = cie_interp(wavelengths)
        SV_vals = S_vals * V_vals

        # Normalize
        S_vals /= S_vals.max()
        V_vals /= V_vals.max()
        SV_vals /= SV_vals.max()

        plt.figure(figsize=(8, 4))
        plt.plot(wavelengths, S_vals, label='D65 (S)', color='blue')
        plt.plot(wavelengths, V_vals, label='CIE 1931 Y (V)', color='green')
        plt.plot(wavelengths, SV_vals, label='S * V', color='red')
        plt.title('Spectral Weight Components (Normalized)')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Relative Intensity')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    return normalized_weights
