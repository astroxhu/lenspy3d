import numpy as np

import pandas as pd

def load_glass_catalog(csv_path='glass_catalog.csv', encoding='shift_jis'):
    def process_df(df):
        df.columns = [col.strip().lower() for col in df.columns]
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
