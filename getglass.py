import numpy as np
import re
import pandas as pd

def load_glass_catalog(csv_path='glass_catalog.csv', encoding='shift_jis'):
    def process_df(df):
        #df.columns = [col.strip().lower() for col in df.columns]
        df.columns = [col.strip().lower() if col.strip().lower() == 'glass' else col for col in df.columns]
        if 'glass' not in df.columns:
            return None
        df.set_index('glass', inplace=True)
        df.index = df.index.str.replace(' ', '', regex=False)  # Strip all spaces from index
        df.fillna(np.inf, inplace=True)
        return df

    # Try default header
    df = pd.read_csv(csv_path, encoding=encoding, na_values=['inf'])
    result = process_df(df)
    if result is not None:
        return result

    # Try skipping the first row (header=1)
    df = pd.read_csv(csv_path, encoding=encoding, na_values=['inf'], header=1)
    result = process_df(df)
    if result is not None:
        return result

    raise ValueError("Column 'glass' not found in any header row.")

class GlassCatalog:
    def __init__(self, df):
        self.df = df

    @classmethod
    def from_csv(cls, path, encoding='shift_jis'):
        df = load_glass_catalog(path, encoding)
        return cls(df)

    def get_index(self, glass_name, wavelength_nm):
        if glass_name not in self.df.index:
            raise ValueError(f"Glass '{glass_name}' not found in catalog")

        row = self.df.loc[glass_name]

        # Try different coefficient sets
        coeffs = self._extract_coefficients(row)
        if coeffs is None:
            raise ValueError(f"Sellmeier coefficients not found for '{glass_name}'")

        B, C = coeffs
        λ = wavelength_nm / 1000  # nm → μm
        λ2 = λ * λ
        n2 = 1 + sum(Bi * λ2 / (λ2 - Ci) for Bi, Ci in zip(B, C))
        return np.sqrt(n2)

    def _extract_coefficients(self, row):
        row_lower = {k.strip().lower(): v for k, v in row.items()}

        def get_coeffs(prefix_b, prefix_c):
            try:
                B = [float(row_lower[f"{prefix_b}{i}"]) for i in range(1, 4)]
                C = [float(row_lower[f"{prefix_c}{i}"]) for i in range(1, 4)]
                if all(np.isfinite(B)) and all(np.isfinite(C)):
                    return B, C
            except KeyError:
                pass
            return None

        # Try Sellmeier B1–B3 and C1–C3
        result = get_coeffs("b", "c")
        if result:
            return result

        # Try A1–A3 and A4–A6 (some catalogs use this)
        try:
            A = [float(row_lower[f"a{i}"]) for i in range(1, 4)]
            C = [float(row_lower[f"a{i}"]) for i in range(4, 7)]
            if all(np.isfinite(A)) and all(np.isfinite(C)):
                return A, C
        except KeyError:
            pass

        # Try Schott-style Sellmeier: A0–A5
        try:
            A0 = float(row_lower["a0"])
            A = [float(row_lower[f"a{i}"]) for i in range(1, 6)]
            if np.isfinite(A0) and all(np.isfinite(A)):
                # Schott: n² = A0 + A1 λ² + A2 λ⁴ + A3 λ⁶ + ...
                # You may handle this differently — here we skip it
                raise NotImplementedError("Schott-style A0–A5 not yet implemented")
        except KeyError:
            pass

        return None

    def match_glass_single(self, nd_lens, vd_lens, N=5):
        
        #Find the closest matching glass by nd and vd.
        
        df = self.df
        nd_diffs = np.abs(df['nd'] - nd_lens)
        closest_nd_indices = nd_diffs.nsmallest(N).index
        subset = df.loc[closest_nd_indices]

        vd_diffs = np.abs(subset['vd'] - vd_lens)
        best_idx = vd_diffs.idxmin()
        row = self.df.loc[best_idx]

        coeffs = self._extract_coefficients(row)
        return best_idx, coeffs

    def match_glass(self, lens_data: dict, N=5):
        
        #Match each lens entry to the closest glass in the catalog.
        #Returns {index: glass_name}
        
        matches = {}
        for i, lens in lens_data.items():
            if 'nd' not in lens or 'vd' not in lens:
                continue
            name, _ = self.match_glass_single(lens['nd'], lens['vd'], N)
            matches[i] = name
        return matches

def match_glass_single(df, nd_lens, vd_lens, N=5):
    
    #Find the closest matching glass by nd and vd.
    #:param df: pandas DataFrame with glass catalog
    #:param nd_lens: lens refractive index (nd)
    #:param vd_lens: lens Abbe number (νd)
    #:param N: Number of closest matches to consider for nd
    #:return: Best glass name and its coefficients (B, C)
    
    if 'νd' in df.columns:
        df['νd'] = pd.to_numeric(df['νd'].replace('#DIV/0!', np.nan), errors='coerce')
    else:
        df['vd'] = pd.to_numeric(df['vd'].replace('#DIV/0!', np.nan), errors='coerce')
    # Step 1: Find N closest nd values
    nd_diffs = np.abs(df['nd'] - nd_lens)
    closest_nd_indices = nd_diffs.nsmallest(N).index  # Corrected line: nsmallest() is directly called on the Series
    subset = df.loc[closest_nd_indices]
    # DEBUG: show candidate glass names and their nd
    #for name in subset.index:
    #    print(f"Candidate glass of nd: {name}, nd: {subset.loc[name, 'nd']}")
    # Step 2: Among those, find closest vd (or νd)
    if 'νd' in df.columns:
        vd_diffs = np.abs(subset['νd'] - vd_lens)
    else:
        #print('subset vd',subset['vd'])
        vd_diffs = np.abs(subset['vd'] - vd_lens)
    best_idx = vd_diffs.idxmin()
    row = df.loc[best_idx]

    # Extract coefficients
    #coeffs = _extract_coefficients(row)
    print('original nd vd',nd_lens, vd_lens)
    print('matched nd vd', df.loc[best_idx]['nd'], df.loc[best_idx]['νd'] if 'νd' in df.columns else df.loc[best_idx]['vd'], best_idx, df.loc[best_idx]['manufacturer'], df.loc[best_idx]['classification'])
    return best_idx #, coeffs

def assign_glasses_to_lens_data(lens_data: dict, catalog_df: pd.DataFrame, N=3):
    
    #Assign closest glass to each surface in the lens_data dict.
    #:param lens_data: Dictionary containing lens data with nd and νd values
    #:param catalog_df: DataFrame containing glass catalog data
    #:param N: Number of closest glasses to consider for matching
    #:return: Updated lens_data with 'glass' and coefficient information
    
    for surf in lens_data.values():
        if surf.get('type', 'lens') == 'air':
            continue
        if 'nd' not in surf or 'vd' not in surf:
            continue
        if surf['nd']<1.005:
            continue

        #glass_name, (B, C) = match_glass_single(catalog_df, surf['nd'], surf['νd'], N=N)
        glass_name = match_glass_single(catalog_df, surf['nd'], surf['vd'], N=N)
        surf['glass'] = glass_name
        #print(glass_name)
        #surf['B1'], surf['B2'], surf['B3'] = B
        #surf['C1'], surf['C2'], surf['C3'] = C

    return lens_data

def compute_refractive_index(model_str, coeffs, wavelength_mm, manufacturer=None,glass_name=None, line_match=False):
    wavelength = wavelength_mm*1e3
    if line_match:
        # Wavelengths (nm) and corresponding catalog keys
        wl_um = [0.4358, 0.4861, 0.5461, 0.5876, 0.6563]

        n_keys = ["ng", "nF", "ne", "nd", "nC"]

        differences = [abs(wavelength - wl) for wl in wl_um]
        index = differences.index(min(differences))
        n_idx = n_keys[index]
        


    if model_str.startswith("Sellmeier"):
        n_match = re.search(r"n\s*=\s*(\d+)", model_str)
        if not n_match:
            raise ValueError("Invalid Sellmeier format.")
        n_terms = int(n_match.group(1))

        if len(coeffs) < 2 * n_terms:
            raise ValueError(f"Expected {2*n_terms} coefficients for Sellmeier, got {len(coeffs)}.")

        B = coeffs[0:n_terms]
        C = coeffs[n_terms:]
        #A1 = coeffs[0]
        #A2 = coeffs[1]
        #A3 = coeffs[2]
        #A4 = coeffs[3]
        #A5 = coeffs[4]
        #A6 = coeffs[5]
        #wl2 = wavelength**2
        #n2 = 1 + (A1 * λ2 / (λ2 - B1**1)) + (A2 * λ2 / (λ2 - B2**1)) + (A3 * λ2 / (λ2 - B3**1)) 
        # Compute Sellmeier formula directly for the given wavelength
        n2 = 1 + sum(B[i] * wavelength**2 / (wavelength**2 - C[i]) for i in range(n_terms))
        n_squared = n2
        #print('manufacturer', manufacturer, manufacturer.strip().lower())
        if manufacturer is not None and manufacturer.strip().lower() == 'hikari':          
            B = coeffs[0::2]
            C = coeffs[1::2]
            n2 = 1 + sum(B[i] * wavelength**2 / (wavelength**2 - C[i]) for i in range(n_terms))
            n_squared = ((n2-1)*2+1)/(2-n2) 
        #n_squared = 1 + A1*wl2/(wl2 - A2) +  A3*wl2/(wl2 - A4) + A5*wl2/(wl2 - A6)
        if manufacturer is not None and manufacturer.strip().lower() == 'cdgm':          
            B = coeffs[0::2]
            C = coeffs[1::2]
            n_squared = 1 + sum(B[i] * wavelength**2 / (wavelength**2 - C[i]) for i in range(n_terms))
        n_val = np.sqrt(n_squared)
        if n_val<1.3 or n_val > 2.1:
            print('Sellmeier', manufacturer,glass_name, n_val, n_terms, coeffs)


    elif model_str.startswith("Schott"):
        n_match = re.search(r"n\s*=\s*(\d+)", model_str)
        if not n_match:
            raise ValueError("Invalid Schott format.")
        n = int(n_match.group(1))
        m_match = re.search(r"m\s*=\s*(\d+)", model_str)
        if not m_match:
            raise ValueError("Invalid Schott format.")
        m = int(m_match.group(1))

        if len(coeffs) != m + 2:  # A1, A2, ..., Am+1
            raise ValueError(f"Expected {m+2} coefficients for Schott 2x{m}, got {len(coeffs)}.")

        #A1 = coeffs[0]
        #A2 = coeffs[1]
        #A3 = coeffs[2]
        #A4 = coeffs[3]
        #A5 = coeffs[4]
        #A6 = coeffs[5]
        #wl=wavelength
        coeffs_p = coeffs[:n]
        inverse_coeffs = coeffs[n:]

        # Compute Schott formula directly for the given wavelength
        n_squared = sum(coeffs_p[i] * wavelength**(2*i) for i in range(n)) + sum(inverse_coeffs[i] * wavelength**(-2*(i+1)) for i in range(m))
        n_val = np.sqrt(n_squared)
        #n_val = A1+A2*wl**2+A3*wl**-2+A4*wl**-4+A5*wl**-6+A6*wl**-8
        if n_val<1.3 or n_val > 2.1:
            print('Schott',manufacturer, glass_name, n_val, n,m, coeffs)

    else:
        raise ValueError(f"Unknown model: {model_str}")
    return n_val

