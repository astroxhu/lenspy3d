import numpy as np

def ray_sphere_intersect(ray_origin, ray_dir, sphere_center, radius):
    """
    Compute the first intersection of a ray with a sphere.
    
    Parameters:
        ray_origin (array-like): Origin of the ray, shape (3,)
        ray_dir (array-like): Direction of the ray (should be normalized), shape (3,)
        sphere_center (array-like): Center of the sphere, shape (3,)
        radius (float): Radius of the sphere
        
    Returns:
        tuple: (intersection point as np.array, t value) or (None, None) if no intersection
    """
    o = np.array(ray_origin, dtype=np.float128)
    d = np.array(ray_dir, dtype=np.float128)
    c = np.array(sphere_center, dtype=np.float128)
     
    m = o - c
    a = np.dot(d,d)
    b = 2 * np.dot(d, m)
    #c_val = np.dot(m, m) - radius**2
    norm_m = np.linalg.norm(m)
    c_val = (norm_m+radius)*(norm_m-radius) 
    discriminant = b**2 - 4 * a * c_val

    
    
    if discriminant < 0:
        return None, None  # No real intersection
    
    sqrt_disc = np.sqrt(discriminant)
    q = -0.5 * (b + np.copysign(sqrt_disc, b))
    t1 = q / a
    t2 = c_val / q
    #t1 = (-b - sqrt_disc) / 2
    #t2 = (-b + sqrt_disc) / 2
    
    # Choose the smallest positive t
    t = min(t for t in (t1, t2) if t >= 0) if any(t >= 0 for t in (t1, t2)) else None
    if t is None:
        return None, None
    
    intersection_point = o + t * d
    return intersection_point, t

from mpmath import mp, mpf, sqrt

# Set desired precision (e.g., 50 decimal digits)
mp.dps = 50

def ray_sphere_intersect_mp(ray_origin, ray_dir, sphere_center, radius):
    o = [mpf(x) for x in ray_origin]
    d = [mpf(x) for x in ray_dir]
    c = [mpf(x) for x in sphere_center]
    r = mpf(radius)

    m = [o[i] - c[i] for i in range(3)]
    a = sum(d[i] * d[i] for i in range(3))
    b = 2 * sum(d[i] * m[i] for i in range(3))
    c_val = sum(m[i] * m[i] for i in range(3)) - r * r

    disc = b**2 - 4 * a * c_val
    if disc < 0:
        return None, None

    sqrt_disc = sqrt(disc)
    q = -0.5 * (b + mp.sign(b) * sqrt_disc)
    t1 = q / a
    t2 = c_val / q

    ts = [t for t in (t1, t2) if t >= 0]
    if not ts:
        return None, None

    t = min(ts)
    intersection = [o[i] + t * d[i] for i in range(3)]
    return intersection, t

if __name__ == "__main__":
    ray_origin = [0.0, -2.7601939e8, -1.0e10]
    ray_dir = [0.0, 0.02759142, 0.99961928]
    sphere_center = [0.0, 0.0, 1200.3704]
    radius = 1200.3704

    point, t = ray_sphere_intersect(ray_origin, ray_dir, sphere_center, radius)
    print("Intersection point:", point)
    print("Ray parameter t:", t)

    point, t = ray_sphere_intersect_mp(ray_origin, ray_dir, sphere_center, radius)
    print("Intersection point mp:", point)
    print("Ray parameter t mp:", t)


