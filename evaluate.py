import matplotlib.pyplot as plt
import numpy as np
from constants import *
import optictxt

from lenspy3d import *
from getglass import *
from plottools import *
from diffraction import *
from mtf import *
#digtype=np.float64




numray=100
folder = 'samples'
wls = ["g", "F", "e", "d", "C"]
wl0 = "e"
use_catalog=True
focal_y=[-4,-8,-12,-17,-22]
focal_y=[0,4,8,12,17,22]
aperture_fac=0.99

fl0 = 780
fstop0 = 5.68
#filename = folder+"/"+"apo130f7.7.txt"
filename = '../lenspy/samples/Nikkor856eairdiam.txt' 
#filename = '../lenspy/samples/Nikkor640eairdiam.txt' 
#filename = '../lenspy/samples/sigma556airdiam.txt' 
optic_data = optictxt.parse_optical_file(filename,loc=0)

optic_data = dict(list(optic_data.items())[:])
#print(optic_data)

# Example usage:
glassdata = 'glass/OHARA_20250312_6merge.csv'
glassdata = 'glass/Optical Glass Lookup Table(Typecode Reference).csv'
catalog = load_glass_catalog(glassdata)
#print('catalog loaded')
#print(catalog['manufacturer'][10:30])
#print(catalog['vd'])
#print('catalog\n',catalog.loc['S-NPH7']['a4'])
#print('glass params',catalog.iloc[1])


for label in catalog.index:
    # Access the row by its index (label) and print its content (all columns for that row)
    row = catalog.loc[label]
    #print(f"Label: {label}")
    #print(row)

if use_catalog:
    assign_glasses_to_lens_data(optic_data, catalog, N=3)
    glass_list = []
    for key in optic_data:
        glass = optic_data[key]['glass']
        if glass != 'air' and glass != None:
            glass_list.append(glass)
    print(glass_list)
if not use_catalog:
    catalog = None

FRAUNHOFER_COLORS_NORM = {
    k: tuple(c / 255 for c in rgb)
    for k, rgb in FRAUNHOFER_COLORS.items()
}


weights = get_weights(wls, plot=True, d65_only=False)
#weights = {'g': 0.0,'F': 0.0,'e': 0.1,'d': 0.0,'C': 0.00}
print(weights)


system = SurfaceSystem(optic_data, catalog=catalog)

fig = plt.figure(figsize=(15,6),dpi=117)
ax = plt.subplot(111)

_, z_focus = ray_gen2d(system, ax)
system.draw_lens(ax, z_focus=z_focus)

ax.axis('off')  # Removes everything: ticks, labels, frame

# Set the figure and axes background to black
fig.patch.set_facecolor('black')  # Figure background

fig.tight_layout()
plt.show()



def build_system_and_trace(optical_data, catalog=None, wavelengths=wls, num_rays=20, raygen_kwargs=None, aperture=None):
    system = SurfaceSystem(optical_data, catalog=catalog)
    #print('system assembled')
    #for surf in system.surfaces:
    #    print(surf)
    surf0 = system.surfaces[0]
    if raygen_kwargs is None:
        raygen_kwargs = {}
    if aperture:
        aperture0=aperture
    else:
        aperture0=surf0.diam*aperture_fac
    base_rays = ray_gen(**raygen_kwargs, z1 = surf0.z0, aperture=aperture0, num_rays=num_rays, random=False)
    #print("num of base rays=",len(base_rays))
    rays_by_wavelength = {}
    for wl in wavelengths:
        #print('wl',wl)
        rays = [Ray3D(ray.ps[0], ray.direc.copy(), color='C0', wavelength=wl, n0=1.00277) for ray in base_rays]
        #for ray in rays:
        #    print('init rays',ray.current_point(), ray.direc)
        traced = system.trace(rays)
        #for ray in traced:
        #    print('traced rays',ray.current_point(), ray.direc)
        rays_by_wavelength[wl] = traced
    return rays_by_wavelength

def focal_plane_scan(optic_data, build_system_and_trace, find_focus,catalog=None,
                     y_targets=[4, 8, 12, 17, 22], x0=0, z0=-1e8, num_rays=500,plotsize=15, legend=False, wavelengths=wls):
    #surf0 = optic_data
    # Step 1: Find focal plane using central ray
    center_kwargs = {'x0': x0, 'y0': 0, 'z0': z0}
    result_center = build_system_and_trace(optic_data, catalog=catalog,raygen_kwargs=center_kwargs, num_rays=num_rays)
    rays_center = result_center[wl0]
    z_focus = find_focus(rays_center, radius=65, plot=False, quick_focus=False)
    print(f"the focal plane is at {z_focus:.1f} mm")
    # Step 2: Try different y0 to match radius on focal plane
    y0_candidates = np.linspace(0, -1.1*z0/z_focus*y_targets[-1], 50)  # mm
    r_targets = abs(np.array(y_targets))
    matched_y0s = []

    for r in r_targets:
        best_y0 = None
        min_err = np.inf
        for y0_try in y0_candidates:
            try_kwargs = {'x0': x0, 'y0': y0_try, 'z0': z0}
            result = build_system_and_trace(optic_data, catalog=catalog, raygen_kwargs=try_kwargs, num_rays=1,aperture=0.1)
            rays = result[wl0]
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
    print("Matched y0 are:", matched_y0s)

    # Step 3: Plot spot diagrams for each y0
    fig, axs = plt.subplots(2, 3, figsize=(9, 6))
    axs = axs.ravel()


    ray_types = wavelengths #["g", "F", "e", "d", "C"]  # This can be supplied as a variable
    colorlist=[]
    wl_list = []
    for raytype in ray_types:
         rays = result_center[raytype]
         raycolor = rays[0].color
         raywl    = rays[0].wavelength*1e6
         print('wl = ',raywl)
         label = f"{raytype} ({raywl:.1f} nm)"
         colorlist.append(raycolor)
         wl_list.append(label)

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
        rays = result[wl0]
        points = [ray.point_at_z(z_focus) for ray in rays if ray.reach_z(z_focus)]
        coords = np.array(points)
        xmean = np.mean(coords[:, 0])
        ymean = np.mean(coords[:, 1])

        ax.set_xlim(-plotsize/2,plotsize/2)
        ax.set_ylim(-plotsize/2,plotsize/2)

        for raytype in ray_types:

            rays = result[raytype]
            raycolor = rays[0].color
            #print('raycolor', raycolor, raytype, rays[0].nlist[-2])
            points = [ray.point_at_z(z_focus) for ray in rays if ray.reach_z(z_focus)]
            #print('points',points)
            coords = np.array(points)
            ax.scatter((coords[:, 0]-xmean)*1e3, (coords[:, 1]-ymean)*1e3, s=1,color=raycolor,alpha=1)
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
# Parameters for positioning
    x_start = 0.6
    y_start = 0.99
    x_spacing = 0.12  # horizontal distance between columns
    y_spacing = 0.03  # vertical distance between rows

# Split into two rows: 3 in the top row, 2 in the bottom
    n_per_row = [3, 2]
    row_indices = [0, 3]  # where each row starts
    if legend:
        for row, start_idx in enumerate(row_indices):
            n = n_per_row[row]
            for j in range(n):
                idx = start_idx + j
                color = colorlist[idx]
                label = wl_list[idx]
                fig.text(x_start + j * x_spacing, y_start - row * y_spacing,
                         label, color=color, fontsize=9, ha='left', va='top',fontweight='bold')

    fig.suptitle(f"Spot Diagrams at z = {z_focus:.2f} mm (Focal Plane)",x=0.3)
    plt.tight_layout()
    plt.show()


def focal_plane_scan_render(optic_data, build_system_and_trace, find_focus,catalog=None,
                     y_targets=[0, 4, 8, 12, 17, 22], x0=0, z0=-1e8, num_rays=500,plotsize=25, spotsize=30, plot=True, legend=False):
    #surf0 = optic_data
    # Step 1: Find focal plane using central ray
    system = SurfaceSystem(optic_data, catalog = catalog)
    surf0 = system.surfaces[0]
    center_kwargs = {'x0': x0, 'y0': y_targets[0], 'z0': z0}
    result_center = build_system_and_trace(optic_data, catalog=catalog,raygen_kwargs=center_kwargs, num_rays=num_rays)
    #rays_center = [ray for wl in result_center for ray in result_center[wl]]
    rays_center = result_center[wl0]
    z_focus = find_focus(rays_center, radius=surf0.rad, plot=False, quick_focus=False)
    z_focus = find_focus_psf(result_center, radius=surf0.rad, z_init=z_focus, xgrid=None, 
        weights = weights
        )
    print(f"the focal plane is at {z_focus:.1f} mm")
    # Step 2: Try different y0 to match radius on focal plane
    y0_candidates = np.linspace(0, 1.1*z0/z_focus*y_targets[-1], 10*len(y_targets))  # mm
    r_targets = abs(np.array(y_targets))
    matched_y0s = []

    for r in r_targets:
        best_y0 = None
        min_err = np.inf
        for y0_try in y0_candidates:
            try_kwargs = {'x0': x0, 'y0': y0_try, 'z0': z0}
            ps0 =[x0, y0_try, z0]
            y_apert = -y0_try/z0*surf0.rad/1.5
            direc0 = [0-x0, y_apert-y0_try, 0-z0]
            #result = build_system_and_trace(optic_data, catalog=catalog, raygen_kwargs=try_kwargs, num_rays=1,aperture=0.1)
            rays = [Ray3D(ps0, direc0, color='C0', wavelength=wl0, n0=n_air0)]
            system.trace(rays)
            #rays = result[wl0]
            points = [ray.point_at_z(z_focus) for ray in rays if ray.reach_z(z_focus)]
            #points = ray.point_at_z(z_focus)
            if not len(points)>0:
                continue
            coords = np.array(points)
            #rs = np.sqrt(coords[:, 0]**2 + coords[:, 1]**2)
            #median_r = np.median(rs)
            median_r =np.sqrt(coords[:,0]**2 + coords[:,1]**2)
            err = abs(median_r - r)
            if err < min_err:
                best_y0 = y0_try
                min_err = err
        if best_y0 is not None:
            matched_y0s.append(best_y0)
        else:
            matched_y0s.append(np.nan)

    
    # Prepend y0=0 (on-axis) case
    #matched_y0s = [0.0] + matched_y0s
    #r_targets = [0] + list(r_targets)
    print("Matched y0 are:", matched_y0s)

    # Step 3: Plot spot diagrams for each y0
    if plot:
        fig, axs = plt.subplots(2, 3, figsize=(9, 6),dpi=117)
        #axs = axs.flatten()
        axs = axs.ravel()

    #sc_multi = AdditiveScatterMulti(fig, axs, s=100)

    ray_types = wls  # This can be supplied as a variable
    colorlist=[]
    wl_list = []
    for raytype in ray_types:
         rays = result_center[raytype]
         raycolor = rays[0].color
         raywl    = rays[0].wavelength*1e6
         print('wl = ',raywl)
         label = f"{raytype} ({raywl:.1f} nm)"
         colorlist.append(raycolor)
         wl_list.append(label)

    #ray_sizes = [30,2]
    results_dict = {}

    for i, (y0_val, r_target) in enumerate(zip(matched_y0s, r_targets)):
        if plot:
            ax = axs[i]
            ax.set_facecolor('black')
        if np.isnan(y0_val):
            if plot:
                ax.set_title(f"r={r_target} mm\n(no match)")
                ax.axis('off')
            continue
        result = build_system_and_trace(optic_data, catalog=catalog, raygen_kwargs={'x0': x0, 'y0': y0_val, 'z0': z0}, num_rays=num_rays)
        #for raytype, raysize in zip(ray_types, ray_sizes):
        rays = result[wl0]
        points = [ray.point_at_z(z_focus) for ray in rays if ray.reach_z(z_focus)]
        coords = np.array(points)
        xmean = np.mean(coords[:, 0])
        ymean = np.mean(coords[:, 1])
        if plot:
            ax.set_xlim(-plotsize/2,plotsize/2)
            ax.set_ylim(-plotsize/2,plotsize/2)
            canvas = AdditiveScatterCanvas(ax, s=spotsize)
        #canvas = DynamicAdditiveScatterFig(ax, s=3)
        #canvas = DynamicAdditiveScatterMask(ax, s=3)
        
        points_by_type = {}
        airy_by_type = {}
        for raytype in ray_types:

            rays = result[raytype]
            raycolor = rays[0].color
            #print('raycolor', raycolor, raytype, 'wl',rays[0].wavelength)
            result_points = [(ray.point_at_z(z_focus), ray.ps[1]) for ray in rays if ray.reach_z(z_focus)]
            points, points_at_front = zip(*result_points) if result_points else ([], [])
            #print('points',points)
            coords = np.array(points)
            coords_front = np.array(points_at_front)
            points_by_type[raytype] = coords

            eff_ap = coords_front.max(axis=0) - coords_front.min(axis=0)

            fl=fl0
            fstop = fstop0
            eff_ap *= fl/fstop/eff_ap.max(axis=0) # temp fix for aperture
            wl_ray = rays[0].wavelength
            #wl_ray = 548e-6
            airy_x = 1.22*wl_ray/eff_ap[0]*fl
            airy_y = 1.22*wl_ray/eff_ap[1]*fl

            airy_by_type[raytype] = [airy_x,airy_y]

            #sc_multi.add_points((coords[:, 0]-xmean)*1e3, (coords[:, 1]-ymean)*1e3, color=raycolor, ax_index=i)
            if plot:
                canvas.add_points((coords[:, 0]-xmean)*1e3, (coords[:, 1]-ymean)*1e3, color=raycolor)
        #ax.set_title(f"r ≈ {r_target} mm\ny0 = {y0_val/1e6:.2f} Mm")
        

        results_dict[f"{r_target:.2f}"] = {
            'r': r_target,
            'y0_val': y0_val,
            'xmean': xmean,
            'ymean': ymean,
            'points': points_by_type,
            'airy': airy_by_type,
        }
        if plot:
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
        
            canvas.render()
    
    if plot and len(r_targets) < len(axs):
        for j in range(len(r_targets), len(axs)):
            axs[j].axis('off')
# Parameters for positioning
    x_start = 0.6
    y_start = 0.99
    x_spacing = 0.12  # horizontal distance between columns
    y_spacing = 0.03  # vertical distance between rows

# Split into two rows: 3 in the top row, 2 in the bottom
    n_per_row = [3, 2]
    row_indices = [0, 3]  # where each row starts
    if plot and legend:
        for row, start_idx in enumerate(row_indices):
            n = n_per_row[row]
            for j in range(n):
                idx = start_idx + j
                color = colorlist[idx]
                label = wl_list[idx]
                fig.text(x_start + j * x_spacing, y_start - row * y_spacing,
                         label, color=color, fontsize=9, ha='left', va='top',fontweight='bold')
    if plot:
        fig.suptitle(f"Spot Diagrams at z = {z_focus:.2f} mm (Focal Plane)",x=0.3)
        plt.tight_layout()
        plt.show()

    return results_dict



#focal_plane_scan_render(optic_data, build_system_and_trace, find_focus, y_targets=[1,2,3,4,5], catalog=catalog, num_rays=numray,legend=True)
results = focal_plane_scan_render(optic_data, build_system_and_trace, find_focus, y_targets=focal_y, catalog=catalog, num_rays=numray,legend=False)
#focal_plane_scan(optic_data, build_system_and_trace, find_focus, y_targets=[1,2,3,4,5], catalog=catalog, num_rays=numray,legend=True)


focal_y_mtf = [i*1.0 for i in range(23)]
focal_y_mtf = np.arange(0,22.0001,2)

print('y for mtf',focal_y_mtf)
#results = focal_plane_scan_render(optic_data, build_system_and_trace, find_focus, y_targets=focal_y_mtf, catalog=catalog, num_rays=numray,legend=False, plot=False)



airy_radius_by_wl = {
    'g': 3.0e-3,
    'F': 3.2e-3,
    'e': 3.5e-3,
    'd': 3.7e-3,
    'C': 4.0e-3
}

grid_size = 100
s_limit = 20e-3  # 20 um field size
pixel = s_limit / grid_size
x = np.linspace(-s_limit/2, s_limit/2, grid_size)
y = np.linspace(-s_limit/2, s_limit/2, grid_size)


r_targets, mtf_x_vals, mtf_y_vals = analyze_mtf_across_field(
    results=results,
    wls=wls,
    weights=weights,
    x=x, y=y,
    pixel=pixel,
    smooth = True,
    #airy_radius_by_wl=airy_radius_by_wl,
    freq_samples=[10, 20, 40, 80, 160],  # lp/mm
    convolver_fn=convolve_dots  # You can swap with convolve_dots_subpixel
)


plot_results_psfs_grid(
    results=results,
    wls=['g', 'F', 'e', 'd', 'C'],
    weights=weights,
    colors=FRAUNHOFER_COLORS_NORM,
    extent=np.array([-s_limit, s_limit, -s_limit, s_limit])*1e3,
    normalize=True
)


#for col, value in catalog.loc['S-NPH 7'].items():
#    print(f"Column: {col}, Value: {value}")

raygen_kwargs = {
    'x0':0e6,
    'y0':0e6,
    'z0':-1e8,
}

results = build_system_and_trace(optic_data, catalog=catalog, num_rays=numray, raygen_kwargs=raygen_kwargs)

rays = results[wl0]


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
