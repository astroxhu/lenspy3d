import matplotlib.pyplot as plt
import numpy as np
import optictxt
from lenspy3d import *
from getglass import *
from plottools import *
#digtype=np.float64

numray=200
folder = 'samples'
filename = folder+"/"+"apo130f7.7.txt"

optic_data = optictxt.parse_optical_file(filename)

print(optic_data)



def build_system_and_trace(optical_data, catalog=None, wavelengths=('g', "F'", 'e', 'd', 'C'), num_rays=50, raygen_kwargs=None):
    system = SurfaceSystem(optical_data, catalog=catalog)
    #print('system')
    #for surf in system.surfaces:
        #print(surf)
    surf0 = system.surfaces[0]
    if raygen_kwargs is None:
        raygen_kwargs = {}
    base_rays = ray_gen(**raygen_kwargs, z1 = surf0.z0, aperture=surf0.diam, num_rays=num_rays, random=False)
    rays_by_wavelength = {}
    for wl in wavelengths:
        #print('wl',wl)
        rays = [Ray3D(ray.ps[0], ray.direc.copy(), color='C0', wavelength=wl) for ray in base_rays]
        traced = system.trace(rays)
        #for ray in traced:
        #    print('traced rays',ray.current_point(), ray.direc)
        rays_by_wavelength[wl] = traced
    return rays_by_wavelength

def focal_plane_scan(optic_data, build_system_and_trace, find_focus,catalog=None,
                     y_targets=[4, 8, 12, 17, 22], x0=0, z0=-1e8, num_rays=500,plotsize=15):
    #surf0 = optic_data
    # Step 1: Find focal plane using central ray
    center_kwargs = {'x0': x0, 'y0': 0, 'z0': z0}
    result_center = build_system_and_trace(optic_data, catalog=catalog,raygen_kwargs=center_kwargs, num_rays=num_rays)
    rays_center = result_center['e']
    z_focus = find_focus(rays_center, radius=65, plot=False, quick_focus=False)

    # Step 2: Try different y0 to match radius on focal plane
    y0_candidates = np.linspace(0, 5e6, 200)  # mm
    r_targets = np.array(y_targets)
    matched_y0s = []

    for r in r_targets:
        best_y0 = None
        min_err = np.inf
        for y0_try in y0_candidates:
            try_kwargs = {'x0': x0, 'y0': y0_try, 'z0': z0}
            result = build_system_and_trace(optic_data, catalog=catalog, raygen_kwargs=try_kwargs, num_rays=10)
            rays = result['e']
            points = [ray.point_at_z(z_focus) for ray in rays if ray.reach_z(z_focus)]
            if not points:
                continue
            coords = np.array(points)
            rs = np.sqrt(coords[:, 0]**2 + coords[:, 1]**2)
            median_r = np.median(rs)
            err = abs(median_r - r)
            if err < min_err:
                best_y0 = y0_try
                min_err = err
        if best_y0 is not None:
            matched_y0s.append(best_y0)
        else:
            matched_y0s.append(np.nan)
    # Prepend y0=0 (on-axis) case
    matched_y0s = [0.0] + matched_y0s
    r_targets = [0] + list(r_targets)

    # Step 3: Plot spot diagrams for each y0
    fig, axs = plt.subplots(2, 3, figsize=(9, 6))
    axs = axs.ravel()


    ray_types = ['g', "F'", 'e', 'd', 'C']  # This can be supplied as a variable
    #ray_sizes = [30,2]
    for i, (y0_val, r_target) in enumerate(zip(matched_y0s, r_targets)):
        ax = axs[i]
        ax.set_facecolor('black')
        if np.isnan(y0_val):
            ax.set_title(f"r={r_target} mm\n(no match)")
            ax.axis('off')
            continue
        result = build_system_and_trace(optic_data, catalog=catalog, raygen_kwargs={'x0': x0, 'y0': y0_val, 'z0': z0}, num_rays=num_rays)
        #for raytype, raysize in zip(ray_types, ray_sizes):
        for raytype in ray_types:

            rays = result[raytype]
            raycolor = rays[0].color
            #print('raycolor', raycolor)
            points = [ray.point_at_z(z_focus) for ray in rays if ray.reach_z(z_focus)]
            #print('points',points)
            coords = np.array(points)
            xmean = np.mean(coords[:, 0])
            ymean = np.mean(coords[:, 1])
            ax.scatter((coords[:, 0]-xmean)*1e3, (coords[:, 1]-ymean)*1e3, s=1,color=raycolor,alpha=0.6)
        #ax.set_title(f"r ≈ {r_target} mm\ny0 = {y0_val/1e6:.2f} Mm")
        ax.set_title(f"r ≈ {r_target} mm")
        ax.set_xlabel("x [um]")
        ax.set_ylabel("y [um]")
        ax.set_xlim(-plotsize/2,plotsize/2)
        ax.set_ylim(-plotsize/2,plotsize/2)
        # Compute nice spacing
        x_range=plotsize
        y_range=plotsize
        x_spacing = nice_tick_spacing(x_range)
        y_spacing = nice_tick_spacing(y_range)
        # Compute ticks centered around 0
        xticks = get_centered_ticks(*ax.get_xlim(), x_spacing)
        yticks = get_centered_ticks(*ax.get_ylim(), y_spacing)

        ax.set_xticks(xticks)
        ax.set_yticks(yticks)

        ax.set_aspect('equal')
        ax.grid(True) 

    if len(r_targets) < len(axs):
        for j in range(len(r_targets), len(axs)):
            axs[j].axis('off')

    fig.suptitle(f"Spot Diagrams at z = {z_focus:.2f} mm (Focal Plane)")
    plt.tight_layout()
    plt.show()




# Example usage:
glassdata = 'glass/OHARA_20250312_6.csv'
catalog = load_glass_catalog(glassdata)

print('catalog\n',catalog.loc['S-NPH7']['a4'])

focal_plane_scan(optic_data, build_system_and_trace, find_focus, y_targets=[1,2,3,4,5], catalog=catalog, num_rays=numray)

for label in catalog.index:
    # Access the row by its index (label) and print its content (all columns for that row)
    row = catalog.loc[label]
    #print(f"Label: {label}")
    #print(row)

#for col, value in catalog.loc['S-NPH 7'].items():
#    print(f"Column: {col}, Value: {value}")

raygen_kwargs = {
    'x0':0e6,
    'y0':0e6,
    'z0':-1e8,
}
results = build_system_and_trace(optic_data, catalog=catalog, num_rays=numray, raygen_kwargs=raygen_kwargs)

rays = results["e"]


z_plane0=find_focus(rays,radius=65,plot=True, quick_focus=False, metric='rms')
print('focus at', z_plane0)
z_plane=z_plane0
fig=plt.figure(figsize=(15,8))

ax=plt.subplot(251)

x,y=plot_focal_plane_hits(ax,rays, z_plane,aperture=130)
ax.set_title("focal="+str(z_plane))
cell_size=2e-3 #4.35e-3
ax=plt.subplot(256)
plot_density(ax, x, y, cell_size=cell_size)

plt.show()
