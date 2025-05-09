import numpy as np
import matplotlib.pyplot as plt

def get_centered_ticks(start, end, spacing):
    # Shift range so it starts at the nearest multiple of spacing below or equal to start
    first_tick = np.floor(start / spacing) * spacing
    last_tick = np.ceil(end / spacing) * spacing
    return np.arange(first_tick, last_tick + spacing, spacing)

def nice_tick_spacing(total_range, target_ticks=5):
    """
    Returns a 'nice' tick spacing (1, 2, 5, 10, etc.)
    based on total range and desired number of ticks.
    """
    raw_spacing = total_range / target_ticks
    magnitude = 10 ** np.floor(np.log10(raw_spacing))
    residual = raw_spacing / magnitude

    if residual < 1.5:
        nice = 1
    elif residual < 3:
        nice = 2
    elif residual < 7:
        nice = 5
    else:
        nice = 10

    return nice * magnitude
