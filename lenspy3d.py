import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from constants import *
from getglass import *
from plottools import *
from diffraction import *
from mtf import *


def n_air(wavelength_mm):
    wl_um = wavelength_mm * 1e3
    # Convert to inverse microns
    sigma2 = (1 / wl_um)**2  # in μm⁻²

    # Edlén 1966 / Ciddor approximation for standard air (dry, 15°C, 1013.25 hPa, 450 ppm CO2)
    n_minus_1 = 1e-8 * (8342.13 + 2406030 / (130 - sigma2) + 15997 / (38.9 - sigma2))

    return n_air0
    return 1 + n_minus_1



#def n_air(wavelength_mm):
#    wl_nm = wavelength_mm * 1e6
#    known_wls = np.array(sorted(_AIR_REFRACTIVE_INDEX.keys()))
#    known_ns = np.array([_AIR_REFRACTIVE_INDEX[wl] for wl in known_wls])
#    if wl_nm < known_wls[0] or wl_nm > known_wls[-1]:
 #       print(f"⚠️ Wavelength {wl_nm:.1f} nm is outside air index data range ({known_wls[0]}–{known_wls[-1]} nm). Using nearest value.")
#        wl_nm = np.clip(wl_nm, known_wls[0], known_wls[-1])
#    return n_air0
    #return np.interp(wl_nm, known_wls, known_ns)


def get_surface_edge_z(surf, y):
    R = surf.R
    if abs(R) > 1e10:  # Flat surface
        return surf.z0
    else:
        try:
            return surf.z0 + R - np.sign(R) * np.sqrt(R**2 - y**2)
        except ValueError:
            return surf.z0  # fallback if y > R (shouldn't happen if diam is sane)

import pandas as pd
import numpy as np


def load_glass_catalog2(csv_path='glass_catalog.csv', encoding='shift_jis', fallback_encodings=None):
    """
    Load a glass catalog CSV file with configurable encoding.
    
    Parameters:
        csv_path (str): Path to the CSV file.
        encoding (str): Primary encoding to try (default: 'shift_jis').
        fallback_encodings (list[str]): Other encodings to try if the first fails.
        
    Returns:
        pd.DataFrame: Glass catalog with 'glass' (case-insensitive) as index.
    """
    encodings_to_try = [encoding]
    if fallback_encodings:
        encodings_to_try += fallback_encodings
    
    last_exception = None
    for enc in encodings_to_try:
        try:
            df = pd.read_csv(csv_path, encoding=enc, na_values=['inf'])
            df.columns = [col.lower() for col in df.columns]
            if 'glass' not in df.columns:
                raise KeyError("Expected column 'glass' (case-insensitive) not found.")
            df.set_index('glass', inplace=True)
            df.fillna(np.inf, inplace=True)
            return df
        except Exception as e:
            last_exception = e

    raise RuntimeError(f"Failed to load CSV with any of the tried encodings: {encodings_to_try}") from last_exception

def get_index_from_catalog(glass_name, wavelength_mm, catalog):
    wavelength_nm = wavelength_mm * 1e6
    wls = np.array([float(col[2:]) for col in catalog.columns if col.startswith('n_')])
    ns = np.array([catalog.loc[glass_name][f'n_{wl:.1f}'] for wl in wls])
    if wavelength_nm < wls.min() or wavelength_nm > wls.max():
        print(f"⚠️ Warning: Wavelength {wavelength_nm:.1f} nm is outside the glass catalog range ({wls.min()}–{wls.max()} nm)")
    return np.interp(wavelength_nm, wls, ns)

class Ray3D:
    def __init__(self, ps, direc, color='k', wavelength=550, n0=None):
        self.ps = np.array([ps])
        self.direc = np.array(direc)
        self.direc /= np.linalg.norm(direc)
        self.color = color
        self.nlist = []
        if n0 is not None:
            self.nlist = [n0]
        if isinstance(wavelength, str):
            if wavelength in FRAUNHOFER_WAVELENGTHS:
                self.wavelength = FRAUNHOFER_WAVELENGTHS[wavelength] * 1e-6  # nm → mm
                self.color = tuple(c/255 for c in FRAUNHOFER_COLORS[wavelength])  # normalized RGB
            else:
                raise ValueError(f"Unknown wavelength symbol: {wavelength}")
        elif isinstance(wavelength, (int, float)):
            if 100 < wavelength < 2000:
                self.wavelength = wavelength * 1e-6  # assume nm → mm
            else:
                self.wavelength = wavelength  # already in mm
        else:
            raise TypeError("Wavelength must be float (nm/mm) or Fraunhofer symbol")

    @property
    def wavelength_nm(self):
        return self.wavelength * 1e6
 
    def add_point(self, point):
        self.ps = np.vstack([self.ps, point])  # Append new point to the path

    def update_direction(self, new_dir):
        self.dir = new_dir / np.linalg.norm(new_dir)  # Update direction

    def current_point(self):
        return self.ps[-1]

    def add_n(self,new_n):
        return self.nlist.append(new_n)

    def current_n(self):
        return self.nlist[-1]
    
    def reach_z(self, z):
        """Return True if the ray's path or its forward extension reaches z."""
        # First: check extension of the last segment
        p_last = self.ps[-1]
        dz = self.direc[2]
        if dz != 0:
            t = (z - p_last[2]) / dz
            if t >= 0:
                return True

        # Second: check whether any segment already crosses z
        for i in range(len(self.ps) - 1):
            z1, z2 = self.ps[i][2], self.ps[i + 1][2]
            if (z1 - z) * (z2 - z) <= 0 and z1 != z2:
                return True

        return False

    def point_at_z(self, z):
        """Return the (x, y, z) point where the ray intersects the plane z=z."""
        # First: use extension from the last point
        p_last = self.ps[-1]
        dz = self.direc[2]
        if dz != 0:
            t = (z - p_last[2]) / dz
            if t >= 0:
                return p_last + t * self.direc

        # Second: search path segments only if extension doesn't work
        for i in range(len(self.ps) - 1):
            p1, p2 = self.ps[i], self.ps[i + 1]
            z1, z2 = p1[2], p2[2]
            if (z1 - z) * (z2 - z) <= 0 and z1 != z2:
                t = (z - z1) / (z2 - z1)
                return p1 + t * (p2 - p1)

        return None

class SphericalSurface:
    def __init__(self, R, diam, z0, x0=0.0, y0=0.0,
                 n_in=1.0, n_out=1.0,
                 glass_in=None, glass_out=None,
                 use_vacuum=False, formula=None, ncoeffs=None, manufacturer=None, debug=False, is_diaphragm = False, line_match=False):
        self.R = R
        self.diam = diam
        self.rad = diam / 2.0
        self.x0 = x0
        self.y0 = y0
        self.z0 = z0
        self.is_diaphragm = is_diaphragm
        self.is_flat = np.isinf(self.R)
        
        if not self.is_flat:
            self.center = np.array([x0, y0, z0 + R])
        else:
            self.center = None  # Not used for flat

        if self.is_flat: 
            self.z_top = self.z0
            self.z_bot = self.z0
        else:
            self.z_top = self.z0 + self.R - np.sign(self.R) * np.sqrt(self.R**2 - self.rad**2)
            self.z_bot = self.z0 + self.R - np.sign(self.R) * np.sqrt(self.R**2 - self.rad**2)

        

        self.default_n_in = n_in
        self.default_n_out = n_out
        self.glass_in = glass_in
        self.glass_out = glass_out
        self.use_vacuum = use_vacuum
        self.formula = formula
        self.ncoeffs = ncoeffs
        self.manufacturer = manufacturer
        self.line_match = line_match
        self.n_dict=dict()
        self.debug = debug

    def get_refractive_indices(self, wavelength_mm):
        """Return (n_in, n_out) at given wavelength (in mm), using the Sellmeier catalog if available."""

        def is_air(glass_name, default_n):
            return (glass_name is None or glass_name.lower() == 'air' or abs(default_n - 1.0) < 5e-4)

        def resolve(default_n):
            if self.line_match:
                wavelength_nm =  wavelength_mm*1e6
                # Wavelengths (nm) and corresponding catalog keys
                wl_nm = [435.8, 486.1, 546.1, 587.6, 656.3]

                n_keys = ["ng", "nF", "ne", "nd", "nC"]

                differences = [abs(wavelength_nm - wl) for wl in wl_nm]
                index = differences.index(min(differences))
                n_idx = n_keys[index]
                n_match = self.n_dict[n_idx]
                return n_match
            if self.ncoeffs is not None:
                n_compute = compute_refractive_index(self.formula, self.ncoeffs, wavelength_mm, self.manufacturer, self.glass_out)
                if abs(n_compute-default_n)>0.2:
                    print("n_compute",n_compute,default_n, self.manufacturer,self.glass_out)
                return n_compute
            return default_n  # fallback

        if self.use_vacuum:
            return 1.0, 1.0

        # Determine n_in
        #if is_air(self.glass_in, self.default_n_in):
        #    n1 = n_air(wavelength_mm)
        #else:
        #    n1 = resolve(self.default_n_in)

        # Determine n_out
        if is_air(self.glass_out, self.default_n_out):
            n2 = n_air(wavelength_mm)
        else:
            n2 = resolve(self.default_n_out)

        return n2


    def __str__(self):
        return (f"SphericalSurface(\n"
                f"  R={self.R},\n"
                f"  diam={self.diam}, rad={self.rad},\n"
                f"  x0={self.x0}, y0={self.y0}, z0={self.z0},\n"
                #f"  center={self.center.tolist()},\n"
                f"  center={self.center},\n"
                f"  n_in={self.default_n_in}, n_out={self.default_n_out},\n"
                f"  glass_in={self.glass_in}, glass_out={self.glass_out},\n"
                f")")
                #f"  catalog={self.catalog}, use_vacuum={self.use_vacuum}\n"


    def intersect(self, p0, d):
        if self.is_flat:
        # Ray-plane intersection at z = self.z0
            dz = d[2]
            if dz == 0:
                print('parallel')
                return [np.nan,np.nan,np.nan]  # Ray is parallel to the plane
            t = (self.z0 - p0[2]) / dz
            if t < 0:
                return [np.nan,np.nan,np.nan]  # Intersection behind ray origin
            hit = p0 + t * d
            # Check if within aperture
            if np.linalg.norm(hit[:2] - np.array([self.x0, self.y0])) > self.rad:
                if self.debug:
                    print('outside flat',hit,np.linalg.norm(hit[:2] - np.array([self.x0, self.y0])),self.rad)
                return [np.nan,np.nan,np.nan]
            return hit

        oc = p0 - self.center
        a = np.dot(d, d)
        b = 2.0 * np.dot(oc, d)
        c = np.dot(oc, oc) - self.R**2
        discriminant = b**2 - 4*a*c
        if discriminant < 0:
            if self.debug:
                print('outside sphere', oc, self.R)
            return [np.nan,np.nan,np.nan]
        sqrt_disc = np.sqrt(discriminant)
        t0 = (-b - sqrt_disc) / (2*a)
        t1 = (-b + sqrt_disc) / (2*a)
        t = t0 if t0 > 0 else t1
        if t < 0:
            if self.debug:
                print('intersect backward',t)
            return [np.nan,np.nan,np.nan]
        hit = p0 + t * d
        r = np.linalg.norm(hit[:2] - np.array([self.x0, self.y0]))
        if r > self.rad:
            if self.debug:
                print('outside aperture', hit, r, self.rad)
            return [np.nan,np.nan,np.nan]
        return hit

    def normal(self, pt):
        if self.is_flat:
            return np.array([0, 0, -1])  # Assumes surface is perpendicular to z-axis
        else:
            return (pt - self.center) / np.linalg.norm(pt - self.center)*np.sign(self.R)

    def refract(self, d_in, normal, n1, n2):
        cos_i = -np.dot(normal, d_in)
        sin2_t = (n1/n2)**2 * (1 - cos_i**2)
        if sin2_t > 1.0:
            return [np.nan,np.nan,np.nan]
        cos_t = np.sqrt(1 - sin2_t)
        return (n1/n2) * d_in + (n1/n2 * cos_i - cos_t) * normal

class SurfaceSystem:
    def __init__(self, optical_data_list, catalog=None, n_keys=["ng", "nF", "ne", "nd", "nC"], debug=False):
        self.surfaces = []
        self.catalog = catalog
        self.n_keys = n_keys
        cumulative_z = 0.0
        n_in = n_air0
        glass_in = 'air'
        self.debug=debug
        for i, data in optical_data_list.items():
            #if data.get('type', 'sphere') != 'sphere':
            #    continue
            
            if 'z' in data:
                z_pos = data['z']
                cumulative_z = z_pos  # reset base if explicit
            else:
                z_pos = cumulative_z
                cumulative_z += data.get('d', 0.0)

            # Retrieve glass information from catalog
            if self.catalog is not None and 'glass' in data and data['nd']>1.005:
                glass_type = data['glass']
                # Get the glass parameters from catalog, e.g., formula for n and glass properties
                
                glass_data = self.catalog.loc[glass_type]
                # Formula for refractive index calculation
                formula = glass_data['Formula']
                manufacturer = glass_data['manufacturer']
                # For example, As (a4) might be relevant for other glass properties
                ncoeffs = np.array([glass_data[f'a{i}'] for i in range(1, 7)])  # if you need to use it elsewhere
                n_dict=dict(zip(n_keys, glass_data[n_keys]))

            else:
                formula = None
                ncoeffs = None
                manufacturer = None

            surf = SphericalSurface(
                R=data['r'],
                diam=data['diam'],
                z0=z_pos,
                n_in=n_in,
                n_out=data['nd'] if data['nd'] > 1.0 else 1.0,
                glass_in = glass_in,
                glass_out=data.get('glass'),
                formula = formula,
                ncoeffs = ncoeffs,
                manufacturer = manufacturer,
                #is_diaphragm = is_diaphragm
                #catalog=catalog
            )
            surf.n_dict = n_dict
            n_in = surf.default_n_out
            glass_in = surf.glass_out
            self.surfaces.append(surf)

    def add_surface(self, R, diam, z0, x0=0.0, y0=0.0,
                 n_in=n_air0, n_out=n_air0,
                 glass_in=None, glass_out=None,
                 use_vacuum=False, formula=None, ncoeffs=None, manufacturer=None, debug=False, is_diaphragm = False, line_match=False):
        surf = SphericalSurface(
                R=R,
                diam=diam,
                z0=z0,
                n_in=n_in,
                n_out=n_out,
                glass_in = glass_in,
                glass_out = glass_out,
                formula = formula,
                ncoeffs = ncoeffs,
                manufacturer = manufacturer,
                #is_diaphragm = is_diaphragm
                #catalog=catalog
            )
        self.surfaces.append(surf)

 


    def trace(self, rays):
        for i_surf, surface in enumerate(self.surfaces):
            updated_rays = []
            for ray in rays:
                if np.isnan(ray.current_point()[2]):
                    continue

                pt = surface.intersect(ray.current_point(), ray.direc)
                ray.add_point(pt)
                if np.isnan(pt[2]):
                    if self.debug:
                        print('pt in trace', pt)
                    continue
                N = surface.normal(pt)

                # Use catalog to resolve actual indices for this wavelength
                if self.catalog is not None:
                    n1 = ray.current_n()
                    if i_surf == 0: # first surface
                        n1 = n_air(ray.wavelength)
                    n2 = surface.get_refractive_indices(ray.wavelength) ## will need to fix n1 

                    #print('wl', ray.wavelength, 'n1',n1,'n2',n2)
                else:
                    n1 = surface.default_n_in
                    n2 = surface.default_n_out
                    #print('default', 'n1',n1,'n2',n2)
                
                T = surface.refract(ray.direc, N, n1, n2)
                if T is None:
                    print('T',T)
                    continue
                ray.direc = T / np.linalg.norm(T)
                ray.add_n(n2)
                updated_rays.append(ray)
            rays = updated_rays
        return rays

    def draw_lens(self, ax=None,axcolor='k',color='w',lw=0.8, diaphragm_size=8, z_focus=None, r_focus=22):
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))

        ax.set_facecolor(axcolor)

        ax.set_aspect('equal')
        ax.set_xlabel('Optical axis (z)')
        ax.set_ylabel('Height (y)')
        y_max=0
        for i, surf in enumerate(self.surfaces):
            z = surf.z0
            R = surf.R
            diam = surf.diam
            rad = surf.rad
            if rad>y_max:
                y_max=rad

            # Diaphragm check
            is_diaphragm = (
                abs(surf.default_n_in - 1.0) < 0.0005 and
                abs(surf.default_n_out - 1.0) < 0.0005 and
                #(surf.glass_in == 'air' and surf.glass_out == 'air') and
                abs(R) > 1e10  # effectively flat
            )
            if is_diaphragm:
                ax.plot([z, z], [rad, rad+diaphragm_size], color, linewidth=lw)
                ax.plot([z, z], [-diaphragm_size-rad, -rad], color, linewidth=lw)
                ax.text(z, -rad - diaphragm_size*2, 'Diaphragm', ha='center', fontsize=10, va='top',color=color)
                continue

            # Draw curved or flat surface
            if abs(R) > 1e10:  # flat surface
                ax.plot([z, z], [-rad, rad], color, lw=lw)
            else:
                sag_sign = np.sign(R)
                center_z = z + R
                theta = np.linspace(-np.arcsin(rad / abs(R)), np.arcsin(rad / abs(R)), 200)
                y_arc = R * np.sin(theta)
                z_arc = center_z - R * np.cos(theta)
                ax.plot(z_arc, y_arc, color, lw=lw)

            # Optional: connect lens edges
            if i < len(self.surfaces) - 1:
                next_surf = self.surfaces[i + 1]
                if abs(surf.default_n_out) > 1.0005:  # optical interface
                    y1 = diam / 2
                    y2 = next_surf.diam / 2
                    if y1 >= y2:
                        ax.plot([surf.z_top, next_surf.z_top], [y1, y1], color,lw=lw)
                        ax.plot([next_surf.z_bot, next_surf.z_top], [y2, y1], color,lw=lw)
                        
                        ax.plot([surf.z_top, next_surf.z_top], [-y1, -y1], color, lw=lw)
                        ax.plot([next_surf.z_bot, next_surf.z_top], [-y2, -y1], color, lw=lw)
                    else:
                        ax.plot([surf.z_top, next_surf.z_top], [y2, y2], color, lw=lw)
                        ax.plot([surf.z_bot, surf.z_top], [y2, y1], color, lw=lw)
                        
                        ax.plot([surf.z_top, next_surf.z_top], [-y2, -y2], color, lw=lw)
                        ax.plot([surf.z_bot, surf.z_top], [-y2, -y1], color, lw=lw)


            # Draw lens body edge if next surface is part of same lens
        if z_focus:
            ax.plot([z_focus, z_focus], [-r_focus, r_focus], color, lw=lw)
            draw_scale(ax,start=0,end=z_focus, position = -y_max*1.1, color=color)

        ax.set_title("Lens System Layout")
        #ax.grid(True)
        return ax

def ray_gen(x0=0, y0=0, z0=-1e10, num_rays=50, R1=1.0, z1=0.0, aperture=10.0, random=True,n0=n_air0):
    rays = []
    aperture_radius = aperture / 2.0
    if random:
        for _ in range(num_rays):
            r = aperture_radius * np.sqrt(np.random.rand())
            theta = 2 * np.pi * np.random.rand()
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            dx = x - x0
            dy = y - y0
            dz = z1 - z0
            ray = Ray3D((x0, y0, z0), (dx, dy, dz),n0=n0)
            rays.append(ray)
    else:
        num_rings = int(np.sqrt(num_rays))
        for i in range(num_rings):
            r = aperture_radius * (i + 1) / num_rings
            num_points = int(2 * np.pi * r / (aperture / num_rings))
            for j in range(num_points):
                theta = 2 * np.pi * j / num_points
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                dx = x - x0
                dy = y - y0
                dz = z1 - z0
                ray = Ray3D((x0, y0, z0), (dx, dy, dz),n0=n0)
                rays.append(ray)
    return rays



def ray_gen2d(system, ax=None, z0=-1e9,num_rays=100,y_targets=[0,4,8,12,17,22],
    #clist=['C0','C1','C2','C3','C4','C5'],
    clist=clist0,
    lw=1):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    ax.set_aspect('equal')
    ax.set_xlabel('Optical axis (z)')
    ax.set_ylabel('Height (y)')

    rad0=system.surfaces[0].rad
    #center_kwargs = {'x0': x0, 'y0': 0, 'z0': -1e10}
    x0 = 0.
    rays = ray_gen(x0=0, y0=0, z0=z0, num_rays=num_rays,aperture=rad0/1.5*2)
    rays_center = system.trace(rays)
    z_focus = find_focus(rays_center, radius=rad0, plot=False, quick_focus=False)
    
    y0_candidates = np.linspace(0, 1.1*z0/z_focus*y_targets[-1], 50)  # mm
    r_targets = abs(np.array(y_targets))
    matched_y0s = []

    for r in r_targets:
        best_y0 = None
        min_err = np.inf
        for y0_try in y0_candidates:
            #try_kwargs = {'x0': x0, 'y0': y0_try, 'z0': z0}
            rays = ray_gen(x0=0, y0=y0_try, z0=z0, num_rays=1,aperture=rad0/10)
            rays = system.trace(rays)
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

    y_range_list=[]
    for y0 in matched_y0s:
        rays = []
        for i in range(num_rays):
            y = -rad0+i*2.0/num_rays*rad0
            x0 = 0.
            dx = 0 - x0
            dy = y - y0
            dz = 0 - z0
            ray = Ray3D((x0, y0, z0), (dx, dy, dz),n0=n_air0)
            rays.append(ray)
        traced_rays=system.trace(rays)
        y_max=-1e10
        y_min=1e10
        for ray in traced_rays:
            if not np.isnan(ray.current_point()[2]):
                if ray.ps[1,1]>y_max:
                    y_max = ray.ps[1,1]
                    if len(ray.ps)>22 and False:
                        print(f'y_max ray={y_max} ',ray.ps[22])
                if ray.ps[1,1]<y_min:
                    y_min = ray.ps[1,1]

        y_range_list.append({'y0':y0,'y_min':y_min,'y_max':y_max,'z0':z0})

    print('y_range',y_range_list)

    demo_rays = []
    print(matched_y0s)

    for i in range(len(matched_y0s)):
        y0 = y_range_list[i]['y0']
        x0 = 0.
        dx = 0.
        dy = y_range_list[i]['y_min']-y0
        dz = 0 - z0
        ray1 = Ray3D((x0, y0, z0), (dx, dy, dz),n0=n_air0,color=clist[i])
        demo_rays.append(ray1)

        y0 = y_range_list[i]['y0']
        x0 = 0.
        dx = 0.
        dy = y_range_list[i]['y_max']-y0
        dz = 0 - z0
        ray2 = Ray3D((x0, y0, z0), (dx, dy, dz),n0=n_air0,color=clist[i])
        demo_rays.append(ray2)
        
        y0 = y_range_list[i]['y0']
        x0 = 0.
        dx = 0.
        dy = (y_range_list[i]['y_max']+y_range_list[i]['y_min'])/2.-y0
        dz = 0 - z0
        ray3 = Ray3D((x0, y0, z0), (dx, dy, dz),n0=n_air0,color=clist[i])
        demo_rays.append(ray3)

    system.add_surface(R=np.inf, z0=z_focus, diam = 2*abs(y_targets[-1])+0.1, n_out=100)
    print('TRACE DEMO')
    system.trace(demo_rays)

    for ray in demo_rays:
        print('demo',ray.ps[-1],ray.ps[1])
        path = np.array(ray.ps)
        if not np.isnan(ray.current_point()[2]) or True:
            ax.plot(path[1:, 2], path[1:, 1], color=ray.color, alpha=1,lw=lw)

    return ax, z_focus



def plot_rays(rays, is_2d=False, max_rays=20, radius=60,return_ax=False):
    fig = plt.figure(figsize=(10, 6))
    if is_2d:
        ax = fig.add_subplot(111)
    else:
        ax = fig.add_subplot(111, projection='3d')
    #sample = np.random.choice(rays, size=max_rays, replace=False)
    if max_rays < len(rays):
    # Define N target positions evenly spaced along the diameter (x-axis)
        x_targets = np.linspace(-radius, radius, max_rays)
        y_targets = np.zeros(max_rays)
        target_positions = np.column_stack((x_targets, y_targets))

        # For each target, find the closest real point
        points = np.array([ray.ps[1] for ray in rays])
        from scipy.spatial import cKDTree
        tree = cKDTree(points[:,[0,1]])
        _, indices = tree.query(target_positions, k=1)  # closest point to each target

        sample = [rays[i] for i in indices]
    else:
        sample = rays
    for ray in sample:
        path = np.array(ray.ps)
        if is_2d:
            #ax.plot(path[:, 2], np.sqrt(path[:, 0]**2+path[:, 1]**2)*np.sign(path[:,0]), color=ray.color, alpha=0.6)
            ax.plot(path[:, 2], path[:, 0], color=ray.color, alpha=1,lw=0.5)
        else:
            ax.plot(path[:, 0], path[:, 1], path[:, 2], color=ray.color, alpha=0.6)


    if is_2d:
        ax.set_xlabel("Z")
        ax.set_ylabel("X")
    else:
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

    plt.tight_layout()

    if return_ax:
        return fig, ax
    else:
        plt.show()


def plot_focal_plane_hits(ax, rays, z_plane, plot=True, airy=True, wavelength=550e-6, aperture=100, field_center=np.array([0, 0])):
    hits = []
    directions = []
    raycolor = rays[0].color
    for ray in rays:
        p0 = ray.current_point()
        d = ray.direc
        t = (z_plane - p0[2]) / d[2]
        if t < 0:
            continue
        hit_point = p0 + d * t
        hits.append(hit_point)
        directions.append(d)

    hits = np.array(hits)
    if len(hits) == 0:
        print("No focal plane hits")
        return
    if plot:
        # Plot the ray intersections
        print('Number of hits=',len(hits[:,0]))
        ax.scatter(hits[:, 0], hits[:, 1], s=0.1, alpha=0.6,color=raycolor)
        ax.set_title(f"Focal Plane Hits at z={z_plane}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_aspect('equal')

        if airy:
            from matplotlib.patches import Ellipse

            # Step 1: find centroid
            centroid = np.mean(hits, axis=0)

            # Step 2: get field direction (from center to centroid)
            field_vec = centroid[:2] - field_center[:2]
            field_radius = np.linalg.norm(field_vec)

            # Angle from z axis: assume field angle = arctan(field radius / focal length)
            theta = np.arctan2(field_radius, z_plane)  # z_plane ~ focal length
            print('theta',theta, np.cos(theta),np.cos(theta),field_radius,z_plane)

            # Step 3: minor and major axes
            minor_axis = 2.44 * wavelength / aperture * abs(z_plane)
            major_axis = minor_axis / np.cos(theta)

            # Step 4: orientation
            angle_deg = np.degrees(np.arctan2(field_vec[1], field_vec[0]))

            # Step 5: draw ellipse
            ellipse = Ellipse(
                xy=(centroid[0], centroid[1]),
                width=major_axis,
                height=minor_axis,
                angle=angle_deg,
                edgecolor='red',
                facecolor='none',
                linestyle='--',
                linewidth=1,
                alpha=0.7,
            )
            ax.add_patch(ellipse)
    return hits[:, 0], hits[:, 1]

from scipy.stats import gaussian_kde

def plot_density(ax, x, y, plot= True, cell_size=2e-3, method='histogram', normalize=False, cmap='viridis'):
    """
    Plot 2D density on the given axis using fixed square cells.

    Parameters:
    - ax: matplotlib Axes object
    - x, y: 1D arrays of coordinates
    - cell_size: width/height of each grid cell
    - method: 'histogram' or 'kde'
    - normalize: if True, normalize histogram to show probability density
    - cmap: colormap for imshow
    """
    x = np.asarray(x)
    y = np.asarray(y)

    # Define grid edges based on cell size
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    x_edges = np.arange(x_min, x_max + cell_size, cell_size)
    y_edges = np.arange(y_min, y_max + cell_size, cell_size)

    if method == 'histogram':
        # Normalized histogram or raw counts
        H, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges], density=normalize)
        data = H.T  # Transpose so it matches meshgrid orientation
        label = 'Probability Density' if normalize else 'Raw Counts'
    elif method == 'kde':
        # Evaluate KDE on a fixed square grid (always normalized)
        xy = np.vstack([x, y])
        kde = gaussian_kde(xy)
        x_grid, y_grid = np.meshgrid(x_edges[:-1], y_edges[:-1])
        positions = np.vstack([x_grid.ravel(), y_grid.ravel()])
        data = kde(positions).reshape(x_grid.shape).T
        label = 'Estimated Density'
    else:
        raise ValueError("method must be 'histogram' or 'kde'")

    # Plot with fixed aspect ratio and proper extent
    if plot:
        im = ax.imshow(data, origin='lower', aspect='equal',
                       extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
                       cmap=cmap)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'2D {method.capitalize()} Density')
        plt.colorbar(im, ax=ax, label=label, orientation='horizontal')
    return data

def closest_approach_z(p1, d1, p2, d2):
    w0 = p1 - p2
    a = np.dot(d1, d1)
    b = np.dot(d1, d2)
    c = np.dot(d2, d2)
    d = np.dot(d1, w0)
    e = np.dot(d2, w0)
    denom = a * c - b * b

    if np.isclose(denom, 0):
        # Lines are parallel
        return None

    t = (b * e - c * d) / denom
    s = (a * e - b * d) / denom

    closest_point1 = p1 + t * d1
    closest_point2 = p2 + s * d2

    z = 0.5 * (closest_point1[2] + closest_point2[2])
    return z


def golden_section_minimize(f, a, b, tol=3e-4, max_iter=50):
    gr = (np.sqrt(5) + 1) / 2  # golden ratio
    c = b - (b - a) / gr
    d = a + (b - a) / gr

    fc = f(c)
    fd = f(d)
    history = [(c, fc), (d, fd)]

    for _ in range(max_iter):
        if abs(b - a) < tol:
            break
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - (b - a) / gr
            fc = f(c)
            history.append((c, fc))
        else:
            a, c, fc = c, d, fd
            d = a + (b - a) / gr
            fd = f(d)
            history.append((d, fd))

    x_min = (b + a) / 2
    f_min = f(x_min)
    history.append((x_min, f_min))
    return x_min, f_min, history

def spread_at_z(z, rays, metric='rms', center='median', **kwargs):
    x, y = plot_focal_plane_hits(None, rays, z, plot=False, **kwargs)
    xc = np.mean(x) if center == 'mean' else np.median(x)
    yc = np.mean(y) if center == 'mean' else np.median(y)
    r = np.sqrt((x - xc)**2 + (y - yc)**2)
    if metric == 'rms':
        return np.sqrt(np.mean(r ** 2))
    elif metric == 'std':
        return np.std(r)
    elif metric == 'hist':
        hist = plot_density(None, x, y, plot=False, cell_size=2e-3)
        return 1./np.max(hist)  # or use np.var(hist), np.sum(hist**2), etc.
    else:
        raise ValueError("Metric must be 'rms', 'std', or 'hist'")    

def white_light_psf(
    result_at_z,
    wls,
    weights,
    x, y,
    airy_radius_by_wl=None,
    convolver_fn=convolve_dots,
):
    """
    Render the white-light PSF from precomputed ray results at a specific focal plane z.

    Parameters
    ----------
    result_at_z : dict
        Dictionary like results[k_str] from your ray trace output at a given z.
    wls : list of float
        List of wavelengths in nm.
    weights : dict[float, float]
        Dict of wavelength weights (e.g. {550: 1.0, 480: 0.6, ...}).
    x, y : ndarray
        1D arrays representing the PSF grid.
    airy_radius_by_wl : dict[float, float] or None
        If provided, uses this for Airy disk size instead of what's in result_at_z['airy'].
    convolver_fn : callable
        Function to convolve dot cloud into PSF.

    Returns
    -------
    psf_total : 2D ndarray
        Normalized white-light PSF on the grid.
    psf_dict : dict[float|str, 2D ndarray]
        Individual PSFs per wavelength, and `'white'` for total.
    """
    grid_size = len(x)
    psf_total = np.zeros((grid_size, grid_size), dtype=float)
    psf_dict = {}

    for wl in wls:
        coords = result_at_z['points'][wl]
        xdots = coords[:, 0] - result_at_z['xmean']
        ydots = coords[:, 1] - result_at_z['ymean']

        if airy_radius_by_wl:
            airy_x = airy_radius_by_wl[wl]
            airy_y = airy_radius_by_wl[wl]
        else:
            airy_x = result_at_z['airy'][wl][0]
            airy_y = result_at_z['airy'][wl][1]

        psf_wl = convolver_fn(x, y, xdots, ydots, airy_x=airy_x, airy_y=airy_y, I0=1.0)
        psf_dict[wl] = psf_wl
        psf_total += weights[wl] * psf_wl

    psf_total /= psf_total.sum()
    psf_dict['white'] = psf_total
    return psf_total, psf_dict



def find_focus_psf(raysdict, radius, z_init=None, xgrid=None, weights=None, num_samples=31, metric = 'mtf', method = 'grid', plot=True, quick_focus=False, freq_m=160):
    
    wls = raysdict.keys()
    n_wl = len(wls)
    if 'e' in wls:
        wle = 'e'
    else:
        wle = wls[0]
    rays = raysdict[wle]

    if z_init is None:
        z_init = find_focus(rays,radius)

    z_min = z_init * 0.9998
    z_max = z_init * 1.0002

    if not xgrid:
        xgrid = np.linspace(-20e-3,20e-3,101)
    ygrid = xgrid
    pixel = xgrid[1]-xgrid[0]
    if not weights:
        weights = [1.]*n_wl

    if method == 'grid':
        z_values = np.linspace(z_min, z_max, num_samples)
        metrics = []
        for z in z_values:
            points_by_wl = {}
            airy_by_wl = {}
            for wl in wls:
                rays = raysdict[wl]
                raycolor = rays[0].color
                result_points = [(ray.point_at_z(z), ray.ps[1]) for ray in rays if ray.reach_z(z)]
                points, points_at_front = zip(*result_points) if result_points else ([], [])
                #print('points',points)
                coords = np.array(points)
                coords_front = np.array(points_at_front)
 
                points_by_wl[wl] = coords

                eff_ap = coords_front.max(axis=0) - coords_front.min(axis=0)

                fl=780 #fl0
                fstop = 5.68 #fstop0

                eff_ap *= fl/fstop/eff_ap.max(axis=0) # temp fix for aperture
                wl_ray = rays[0].wavelength
                #wl_ray = 548e-6
                airy_x = 1.22*wl_ray/eff_ap[0]*fl
                airy_y = 1.22*wl_ray/eff_ap[1]*fl

                airy_by_wl[wl] = [airy_x,airy_y]
            
            xmean = np.mean(points_by_wl[wle][:,0])
            ymean = np.mean(points_by_wl[wle][:,1])

            result_at_z = {
                "points":points_by_wl,
                "xmean": xmean,
                "ymean": ymean,
                'airy':airy_by_wl,
            }

            psf_total, _ = white_light_psf(
                result_at_z, wls, weights, xgrid, ygrid,
                airy_radius_by_wl=None,
                convolver_fn=convolve_dots,
            )

            #fig,ax = plt.subplots()
            #ax.imshow(psf_total)
            #ax.set_title(f"z={z}")
            #plt.show()
            if metric == 'mtf':
                mtf2d, fx, fy, mtf_x, mtf_y, mtf_x_interp, mtf_y_interp, _ = compute_mtf(
                    psf_total, pixel_size_mm=pixel, plot=False
                )
            
                metrics.append(mtf_x_interp(freq_m))
        z_best = z_values[np.argmax(metrics)]

        if plot:
            plt.figure(figsize=(6, 4))
            plt.plot(z_values, metrics, 'o-', label='Initial Search')
            #plt.plot(z_values_3, metrics_3, 'x-', label='Refined Search')
            plt.axvline(z_init, color='b', linestyle='--', label=f'Init Focus: z={z_init:.3f}')
            plt.axvline(z_best, color='g', linestyle='--', label=f'Best Focus: z={z_best:.3f}')
            #plt.axvline(z_best_3, color='r', linestyle='--', label=f'Best Focus: z={z_best_3:.3f}')
            plt.xlabel('z (focal plane)')
            plt.ylabel(f'{metric.upper()} Spread')
            plt.title('Focus Search')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        return z_best #, metrics_refined, z_values_refined



            

def find_focus(rays0, radius, num_samples=21, metric='rms', plot=False, quick_focus=False, method='grid', center='median', **kwargs):
    if isinstance(rays0, dict):
        raysdict=rays0
        wls = raysdict.keys()
        if 'e' in wls:
            wle = 'e'
        else:
            wle = wls[0]
        rays = raysdict[wle]
        
    else:
        rays = rays0

    for ray in rays:
        if abs(ray.ps[1,0]-radius*0.8)<radius*0.1:
            ray1=ray
            break
        else:
            ray1=ray
    p1 = ray1.current_point()
    d1 = ray1.direc
    for ray in rays:
        if abs(ray.ps[1,0]+radius*0.8)<radius*0.1:
            ray2=ray
            break
        else:
            ray2=rays[0]
    p2 = ray2.current_point()
    d2 = ray2.direc

    z_init = closest_approach_z(p1, d1, p2, d2)

    if quick_focus:
        return z_init

    z_min = z_init * 0.98
    z_max = z_init * 1.02

    if method == 'grid':
        z_values = np.linspace(z_min, z_max, num_samples)
        metrics = [spread_at_z(z, rays, metric, center=center, **kwargs) for z in z_values]
        z_best = z_values[np.argmin(metrics)]

        z_min_2 = z_best * 0.997
        z_max_2 = z_best * 1.003
        z_values_2 = np.linspace(z_min_2, z_max_2, num_samples)
        metrics_2 = [spread_at_z(z, rays, metric, center=center, **kwargs) for z in z_values_2]
        z_best_2 = z_values_2[np.argmin(metrics_2)]

        z_min_3 = z_best_2 * 0.9999
        z_max_3 = z_best_2 * 1.0001
        z_values_3 = np.linspace(z_min_3, z_max_3, num_samples)
        metrics_3 = [spread_at_z(z, rays, metric, center=center, **kwargs) for z in z_values_3]
        z_best_3 = z_values_3[np.argmin(metrics_3)]
        
        z_best_refined = z_best_3
        metrics_refined = metrics_3
        z_values_refined = z_values_3
        if plot:
            plt.figure(figsize=(6, 4))
            plt.plot(z_values, metrics, 'o-', label='Initial Search')
            plt.plot(z_values_3, metrics_3, 'x-', label='Refined Search')
            plt.axvline(z_init, color='b', linestyle='--', label=f'Init Focus: z={z_init:.3f}')
            plt.axvline(z_best_2, color='g', linestyle='--', label=f'Med Focus: z={z_best_2:.3f}')
            plt.axvline(z_best_3, color='r', linestyle='--', label=f'Best Focus: z={z_best_3:.3f}')
            plt.xlabel('z (focal plane)')
            plt.ylabel(f'{metric.upper()} Spread')
            plt.title('Focus Search')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        return z_best_refined #, metrics_refined, z_values_refined

    elif method == 'golden':
        f = lambda z: spread_at_z(z, rays, metric, center=center, **kwargs)
        z_best, _, history = golden_section_minimize(f, z_min, z_max)

        if plot:
            z_vals, spreads = zip(*history)
            plt.plot(z_vals, spreads, 'o-', label='Golden Search')
            plt.axvline(z_init, color='b', linestyle='--', label='Init Guess')
            plt.axvline(z_best, color='r', linestyle='--', label=f'Best Focus: z={z_best:.3f}')
            plt.xlabel('z (focal plane)')
            plt.ylabel(f'{metric.upper()} Spread')
            plt.title('Golden-Section Focus Search')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        return z_best, [s for _, s in history], [z for z, _ in history]

    elif method == 'brent':
        from scipy.optimize import minimize_scalar
        f = lambda z: spread_at_z(z, rays, metric, center=center, **kwargs)
        res = minimize_scalar(f, bracket=(z_min, z_max), method='brent')
        z_best = res.x

        if plot:
            z_grid = np.linspace(z_min, z_max, num_samples)
            metrics_grid = [f(z) for z in z_grid]
            plt.plot(z_grid, metrics_grid, '-', label='Brent Eval')
            plt.axvline(z_init, color='b', linestyle='--', label='Init Guess')
            plt.axvline(z_best, color='r', linestyle='--', label=f'Best Focus: z={z_best:.3f}')
            plt.xlabel('z (focal plane)')
            plt.ylabel(f'{metric.upper()} Spread')
            plt.title('Brent Focus Search')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        return z_best, None, None

    else:
        raise ValueError("method must be 'grid', 'golden', or 'brent'")

