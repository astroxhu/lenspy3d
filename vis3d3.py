import numpy as np
import matplotlib.pyplot as plt

n_air=1.01

def refract_ray(I, N, n1, n2):
    I = I / np.linalg.norm(I)
    N = N / np.linalg.norm(N)
    cos_theta_i = -np.dot(N, I)
    eta = n1 / n2
    k = 1 - eta**2 * (1 - cos_theta_i**2)
    if k < 0:
        return None
    T = eta * I + (eta * cos_theta_i - np.sqrt(k)) * N
    return T / np.linalg.norm(T)

def intersect_sphere(center, radius, origin, direction):
    # Returns nearest intersection point from origin toward direction
    oc = origin - center
    d = direction / np.linalg.norm(direction)
    b = 2 * np.dot(oc, d)
    c = np.dot(oc, oc) - radius**2
    disc = b**2 - 4 * c
    if disc < 0:
        #print('disc',disc)
        return None
    t1 = (-b - np.sqrt(disc)) / 2
    t2 = (-b + np.sqrt(disc)) / 2
    #print('b',b,'disc',np.sqrt(disc))
    t = min(t1, t2) if t1 > 1e-6 else max(t1, t2)
    if t < 1e-6:
        #print('t',t)
        return None
    #print('intersect at',origin + t * d)
    return origin + t * d

def ray_gen2d(x0, y0, z0, num_rays, R1, z1, aperture):
    rays = []

    for i in range(num_rays):
        x = 2*aperture * (i-num_rays//2)*1.0/num_rays
        print('x',x)
        y = 0.
        dx = x - x0
        dy = y - y0
        dz = z1 - z0
        ray = Ray3D((x0, y0, z0), (dx, dy, dz))
        rays.append(ray)
    return rays

def ray_gen(x0, y0, z0, num_rays, R1, z1, aperture, random=True):
    rays = []

    if random:
        # Random distribution inside a circle using rejection sampling
        for _ in range(num_rays):
            r = aperture * np.sqrt(np.random.rand())
            theta = 2 * np.pi * np.random.rand()
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            dx = x - x0
            dy = y - y0
            dz = z1 - z0
            ray = Ray3D((x0, y0, z0), (dx, dy, dz))
            rays.append(ray)
    else:
        # Even distribution using concentric rings
        num_rings = int(np.sqrt(num_rays))
        for i in range(num_rings):
            r = aperture * (i + 1) / num_rings
            num_points = int(2 * np.pi * r / (aperture / num_rings))
            for j in range(num_points):
                theta = 2 * np.pi * j / num_points
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                dx = x - x0
                dy = y - y0
                dz = z1 - z0
                ray = Ray3D((x0, y0, z0), (dx, dy, dz))
                rays.append(ray)

    return rays

class Ray3D:
    def __init__(self, ps, dir, color='k'):
        self.ps = np.array([ps])  # Initialize as an array to store multiple points
        self.dir = np.array(dir)
        self.dir /= np.linalg.norm(self.dir)
        self.color = color

    def add_point(self, point):
        self.ps = np.vstack([self.ps, point])  # Append new point to the path

    def update_direction(self, new_dir):
        self.dir = new_dir / np.linalg.norm(new_dir)  # Update direction

    def current_point(self):
        return self.ps[-1]  # Return the latest point


class LensElement:
    def __init__(self, R1, R2, thickness, aperture, n_lens, V_lens, z0=0):
        self.R1 = R1
        self.R2 = R2
        self.thickness = thickness
        self.aperture = aperture
        self.n_lens = n_lens
        self.V_lens = V_lens
        self.z0 = z0  # Starting z position of the lens


class LensSystem:
    def __init__(self, elements):
        self.elements = elements

    def trace(self, rays):
        traced_rays = []
        z_offset = 0
        for element in self.elements:
            R1, R2 = element.R1, element.R2
            thickness = element.thickness
            aperture = element.aperture
            n_lens = element.n_lens
            z_offset = element.z0  # Use z0 from the lens element

            center1 = np.array([0, 0, z_offset + R1])
            center2 = np.array([0, 0, z_offset + thickness + R2])

            updated_rays = []
            for ray in rays:
                pt1 = intersect_sphere(center1, abs(R1), ray.current_point(), ray.dir)
                if pt1 is None:
                    #print ('pt1 error')
                    continue
                ray.add_point(pt1)
                N1 = (pt1 - center1) / np.linalg.norm(pt1 - center1)
                N1 *=np.sign(R1)
                T1 = refract_ray(ray.dir, N1, 1.0, n_lens)
                if T1 is None:
                    #print ('T1 error')
                    continue

                pt2 = intersect_sphere(center2, abs(R2),  ray.current_point(), T1)
                if pt2 is None:
                    #print ('pt2 error')
                    continue
                ray.add_point(pt2)
                N2 = (pt2 - center2) / np.linalg.norm(pt2 - center2)
                N2 *= np.sign(R2)

                T2 = refract_ray(T1, N2, n_lens, 1.0)
                if T2 is None:
                    continue

                ray.update_direction(T2)
                #ray.add_point(pt2 + T2 * 700)
                updated_rays.append(ray)
            rays = updated_rays
        return rays


def draw_lens(ax, R1, R2, thickness, aperture, z_offset=0, n=100, is_2d=False):
    if is_2d:
        for R, z_shift in zip([R1, R2], [0, thickness]):
            x = np.linspace(-aperture, aperture, n)
            z = np.sqrt(np.clip(R**2 - x**2, 0, None))
            z = -z if R > 0 else z
            z += z_shift + R + z_offset
            ax.plot(z, x, color='lightblue', alpha=0.8)
    else:
        for R, z_shift in zip([R1, R2], [0, thickness]):
            phi = np.linspace(0, 2 * np.pi, n)
            theta = np.linspace(0, np.pi, n)
            phi, theta = np.meshgrid(phi, theta)
            x = aperture * np.sin(theta) * np.cos(phi)
            y = aperture * np.sin(theta) * np.sin(phi)
            z = np.sqrt(np.clip(R**2 - x**2 - y**2, 0, None))
            z = -z if R > 0 else z
            z += z_shift + R + z_offset
            ax.plot_surface(x, y, z, color='lightblue', alpha=0.2, linewidth=0)
            

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
        #print('path',path)
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

def plot_focal_plane_hitsold(ax, rays, z_plane, radius=50, airy=True,  wavelength=550e-9,):
    hits = []
    for ray in rays:
        p0 = ray.current_point()
        d = ray.dir
        t = (z_plane - p0[2]) / d[2]
        if t < 0:
            continue
        hit_point = p0 + d * t
        hits.append(hit_point)

    hits = np.array(hits)
    if len(hits) == 0:
        print("No focal plane hits")
        return

    ax.scatter(hits[:, 0], hits[:, 1], s=1, alpha=0.6)
    ax.set_title(f"Focal Plane Hits at z={z_plane}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect('equal')
    
    if airy:
        from matplotlib.patches import Ellipse
        xc=np.mean(hits[:,0])
        yc=np.mean(hits[:,1])



def plot_focal_plane_hits(ax, rays, z_plane, plot=True, airy=True, wavelength=550e-6, aperture=100, field_center=np.array([0, 0])):
    hits = []
    directions = []

    for ray in rays:
        p0 = ray.current_point()
        d = ray.dir
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
        ax.scatter(hits[:, 0], hits[:, 1], s=0.1, alpha=0.6)
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

def find_focusold(rays, radius, num_samples=20, metric='rms', plot=False, quick_focus=False, **kwargs):
    for ray in rays:
        if abs(ray.ps[1,0]-radius*0.8)<radius*0.1:
            ray1=ray
            break
    p1 = ray1.current_point()
    d1 = ray1.dir
    for ray in rays:
        if abs(ray.ps[1,0]+radius*0.8)<radius*0.1:
            ray2=ray
            break
    p2 = ray2.current_point()
    d2 = ray2.dir
    
    
    #z_pl = p0[2]-p0[0]*d[2]/d[0]
    z_init = closest_approach_z(p1, d1, p2, d2)
    
    if quick_focus:
        return z_init
    
    # Initial search window: +- 5% around z_init
    z_min = z_init * 0.95
    z_max = z_init * 1.05
    z_values = np.linspace(z_min, z_max, num_samples)
    metrics = []

    # Evaluate spread for each z
    for z in z_values:
        x, y = plot_focal_plane_hits(None, rays, z, plot=False, **kwargs)
        x_med = np.median(x)
        y_med = np.median(y)
        r = np.sqrt((x-x_med)**2 + (y-y_med)**2)
        if metric == 'rms':
            spread = np.sqrt(np.mean(r**2))  # RMS radius
        elif metric == 'std':
            spread = np.std(r)  # Standard deviation
        else:
            raise ValueError("Metric must be 'rms' or 'std'")
        metrics.append(spread)

    # Find the z with the minimum spread
    z_best = z_values[np.argmin(metrics)]

    # Refine by zooming in around the best z found
    # This zooms in a smaller window to further minimize the spread
    z_min_refined = z_best * 0.95
    z_max_refined = z_best * 1.05
    z_values_refined = np.linspace(z_min_refined, z_max_refined, num_samples)
    metrics_refined = []

    for z in z_values_refined:
        x, y = plot_focal_plane_hits(None, rays, z, plot=False, **kwargs)
        x_med = np.median(x)
        y_med = np.median(y)
        r = np.sqrt((x-x_med)**2 + (y-y_med)**2)
        if metric == 'rms':
            spread = np.sqrt(np.mean(r**2))  # RMS radius
        elif metric == 'std':
            spread = np.std(r)  # Standard deviation
        metrics_refined.append(spread)

    z_best_refined = z_values_refined[np.argmin(metrics_refined)]

    if plot:
        # Plot spread vs. z for both initial and refined search
        plt.figure(figsize=(6, 4))
        plt.plot(z_values, metrics, 'o-', label='Initial Search')
        plt.plot(z_values_refined, metrics_refined, 'x-', label='Refined Search')
        plt.axvline(z_init, color='b', linestyle='--', label=f'Init Focus: z={z_init:.3f}')
        plt.axvline(z_best, color='g', linestyle='--', label=f'Med Focus: z={z_best:.3f}')
        plt.axvline(z_best_refined, color='r', linestyle='--', label=f'Best Focus: z={z_best_refined:.3f}')
        plt.xlabel('z (focal plane)')
        plt.ylabel(f'{metric.upper()} Spread')
        plt.title('Focus Search')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return z_best_refined, metrics_refined, z_values_refined

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

def find_focus(rays, radius, num_samples=20, metric='rms', plot=False, quick_focus=False, method='grid', center='median', **kwargs):
    for ray in rays:
        if abs(ray.ps[1,0]-radius*0.8)<radius*0.1:
            ray1=ray
            break
    p1 = ray1.current_point()
    d1 = ray1.dir
    for ray in rays:
        if abs(ray.ps[1,0]+radius*0.8)<radius*0.1:
            ray2=ray
            break
    p2 = ray2.current_point()
    d2 = ray2.dir

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

        return z_best_refined, metrics_refined, z_values_refined

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


# Define the lens elements
lens1 = LensElement(R1=1750.669, R2=-535.138, thickness=5, aperture=50, n_lens=1.5168, V_lens=64.17, z0=0)
lens2 = LensElement(R1=-535.138, R2=535.138, thickness=5, aperture=50, n_lens=1.6202, V_lens=56.56, z0=5.01)
lens3 = LensElement(R1=535.138, R2=-535.138, thickness=5, aperture=50, n_lens=1.5168, V_lens=64.17, z0=10.02)

lens1 = LensElement(R1=448, R2=222, thickness=10, aperture=70, n_lens=1.51045, V_lens=60.98, z0=0) #N-ZK7
lens2 = LensElement(R1=222, R2=-486, thickness=19, aperture=70, n_lens=1.43985, V_lens=94.94, z0=10.000001) #S-FPL53
lens3 = LensElement(R1=-486, R2=-1880, thickness=10, aperture=70, n_lens=1.5045, V_lens=60.98, z0=29.000002) #N-ZK7

lens1 = LensElement(R1=2380, R2=-262.2, thickness=12, aperture=65, n_lens=1.43985, V_lens=94.94, z0=0) #S-FPL53
lens2 = LensElement(R1=-239.8, R2=2580, thickness=7, aperture=65, n_lens=1.518251, V_lens=63.93, z0=34.9) #S-BSL 7
lens3 = LensElement(R1=444, R2=-926, thickness=11, aperture=65, n_lens=1.43985, V_lens=94.94, z0=42.9) #S-FPL53

# Create the lens system
lens_system = LensSystem([lens1, lens2, lens3])

# Generate rays
x0=1e6
y0=0.4e6
z0=-1e8
num_rays=5000
rays = ray_gen(x0=x0, y0=y0, z0=z0, num_rays=num_rays, R1=1750.669, z1=0, aperture=68,random=False)

# Trace rays through the lens system
traced_rays = lens_system.trace(rays)


#for ray in rays:
#    print('ray', ray.current_point(), ray.dir)

#z_plane0=1061.68
#z_plane0=1004.36-0.00
z_plane0, zp1, zp2=find_focus(rays,radius=65,plot=True)

fig=plt.figure(figsize=(15,8))

ax=plt.subplot(251)

step=0.08
z_plane=z_plane0-2*step
x,y=plot_focal_plane_hits(ax,rays, z_plane)
ax.set_title("focal="+str(z_plane))
cell_size=2e-3 #4.35e-3
ax=plt.subplot(256)
plot_density(ax, x, y, cell_size=cell_size)

ax=plt.subplot(252)
z_plane=z_plane0-step
x,y=plot_focal_plane_hits(ax,rays, z_plane)
ax.set_title("focal="+str(z_plane))

ax=plt.subplot(257)
plot_density(ax, x, y, cell_size=cell_size)

ax=plt.subplot(253)
z_plane=z_plane0
x,y=plot_focal_plane_hits(ax,rays, z_plane)
ax.set_title("focal="+str(z_plane))

ax=plt.subplot(258)
plot_density(ax, x, y, cell_size=cell_size)

ax=plt.subplot(254)
z_plane=z_plane0+step
x,y=plot_focal_plane_hits(ax,rays, z_plane)
ax.set_title("focal="+str(z_plane))

ax=plt.subplot(259)
plot_density(ax, x, y, cell_size=cell_size)

ax=plt.subplot(255)
z_plane=z_plane0+2*step
x,y=plot_focal_plane_hits(ax,rays, z_plane)
ax.set_title("focal="+str(z_plane))

ax=plt.subplot(2,5,10)
plot_density(ax, x, y, cell_size=cell_size)

plt.tight_layout()
plt.show()



rays=ray_gen2d(x0=x0, y0=0, z0=z0, num_rays=40, R1=1750.669, z1=0, aperture=70)
#rays=ray_gen(x0=0, y0=0, z0=-1e8, num_rays=20, R1=1750.669, z1=0, aperture=50)

traced_rays = lens_system.trace(rays)

for ray in rays:
    p0 = ray.current_point()
    d = ray.dir
    t = (z_plane - p0[2]) / d[2]
    if t < 0:
        continue
    hit_point = p0 + d * t

    ray.add_point(hit_point)


# Plot the rays and lens system
fig, ax = plot_rays(rays, is_2d=True, return_ax=True, radius=80, max_rays=40)
z_offset = 0
for elem in lens_system.elements:
    draw_lens(ax, elem.R1, elem.R2, elem.thickness, elem.aperture, z_offset=elem.z0, is_2d=True)
    #z_offset += elem.thickness
ax.set_aspect('equal')
ax.set_xlim(-40,z_plane)
ax.set_ylim(-80,80)
plt.show()
