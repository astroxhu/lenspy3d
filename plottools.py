import numpy as np
import matplotlib.pyplot as plt


clist0 = [
    "#0ADCF9", #(  7, 222, 243), #cyan
    "#7DDE36", #(124, 223,  57), #green
    "#FE5B59", #(253,  90,  88), #orange-red
    "#FBF258", #(251, 241,  81), #gold-yellow
    "#A46DED", #(168, 103, 252), #purple
    "#56A1EB", #( 81, 160, 252), #blue
    ]

def draw_scale(ax, start, end, position=0, orientation='horizontal',
               tick_height=3, label_offset=0.05, fontsize=8, color='w'):
    """
    Draw a linear scale with ticks on a given axis.

    Parameters:
        ax          : matplotlib axis to draw on
        start       : start coordinate along the scale axis
        end         : end coordinate along the scale axis
        position    : constant coordinate orthogonal to the scale (e.g., y if horizontal)
        orientation : 'horizontal' or 'vertical'
        tick_height : length of tick marks
        label_offset: offset of tick labels from the ticks
        fontsize    : size of tick label font
        color       : color of the scale line, ticks, and labels
    """
    total_length = abs(end - start)
    axis_min = min(start, end)
    axis_max = max(start, end)

    # Decide tick step size
    if total_length > 400:
        tick_step = 100
    elif total_length > 150:
        tick_step = 50
    elif total_length > 80:
        tick_step = 20
    elif total_length > 30:
        tick_step = 10
    else:
        tick_step = 5

    ticks = np.arange(np.ceil(axis_min / tick_step) * tick_step,
                      axis_max + 1, tick_step)

    # Draw scale line and ticks
    if orientation == 'horizontal':
        ax.plot([start, end], [position, position], color=color, lw=1)
        for t in ticks:
            ax.plot([t, t], [position - tick_height/2, position + tick_height/2], color=color, lw=1)
            ax.text(t, position - tick_height/2 - label_offset, f"{t:.0f}",
                    ha='center', va='top', fontsize=fontsize, color=color)
    elif orientation == 'vertical':
        ax.plot([position, position], [start, end], color=color, lw=1)
        for t in ticks:
            ax.plot([position - tick_height/2, position + tick_height/2], [t, t], color=color, lw=1)
            ax.text(position - tick_height/2 - label_offset, t, f"{t:.0f}",
                    ha='right', va='center', fontsize=fontsize, color=color)
    else:
        raise ValueError("orientation must be 'horizontal' or 'vertical'")

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


import numpy as np
import matplotlib.pyplot as plt

class AdditiveScatterMulti:
    def __init__(self, fig, axes, s=100):
        """
        Additive scatter for multiple subplots in one figure space.
        Parameters:
        - fig: Matplotlib figure object
        - axes: List of Matplotlib axis objects
        - s: marker size in points^2 (like in plt.scatter)
        """
        self.fig = fig
        self.axes = axes
        self.s = s
        self.points = []  # list of (x, y, color, alpha, ax_index)
        self._cached_img = None

    def add_points(self, x, y, color, alpha=None, ax_index=0):
        """
        Add scatter points to a specific subplot.
        Parameters:
        - x, y: data points
        - color: color for each point
        - alpha: alpha transparency
        - ax_index: index of the subplot to add points
        """
        x = np.asarray(x)
        y = np.asarray(y)
        color = np.asarray(color)
        if alpha is None:
            alpha = np.ones_like(x)
        self.points.append((x, y, color, alpha, ax_index))

    def _make_disk(self, radius_px, feather=1.0):
        """Generates a disk mask with feathering effect."""
        r = int(np.ceil(radius_px + feather))
        y, x = np.mgrid[-r:r+1, -r:r+1]
        dist = np.sqrt(x**2 + y**2)
        mask = np.clip(1 - (dist - radius_px) / feather, 0, 1)
        return mask

    def render(self):
        W, H = self.fig.canvas.get_width_height()
        img = np.zeros((H, W, 3), dtype=np.float32)

        # Calculate pixel radius from point size in pt²
        radius_pt = np.sqrt(self.s / np.pi)
        radius_px = radius_pt * self.fig.dpi / 72
        mask = self._make_disk(radius_px, feather=1.0)

        mh, mw = mask.shape
        off_y, off_x = mh // 2, mw // 2

        for x_arr, y_arr, color_arr, alpha_arr, ax_idx in self.points:
            ax = self.axes[ax_idx]
            # Convert data coordinates to figure pixel coordinates
            xy_pix = ax.transData.transform(np.vstack([x_arr, y_arr]).T)
            x_pix = xy_pix[:, 0].astype(int)
            y_pix = (H - xy_pix[:, 1]).astype(int)  # flip y to match figure space

            # Get axis bounding box in figure pixels
            ax_bbox = ax.get_window_extent().transformed(self.fig.dpi_scale_trans.inverted())
            ax_x0, ax_x1 = ax_bbox.xmin * W, ax_bbox.xmax * W
            ax_y0, ax_y1 = ax_bbox.ymin * H, ax_bbox.ymax * H

            # Render each point in the correct subplot region
            for xi, yi, ci in zip(x_pix, y_pix, color_arr):
                if not (ax_x0 <= xi < ax_x1 and ax_y0 <= yi < ax_y1):
                    continue  # Skip points outside the subplot bounds

                # Calculate offset for the current subplot
                xmin = xi - off_x
                xmax = xi + off_x + 1
                ymin = yi - off_y
                ymax = yi + off_y + 1

                if xmax < 0 or ymax < 0 or xmin >= W or ymin >= H:
                    continue  # out of bounds

                xslice = slice(max(xmin, 0), min(xmax, W))
                yslice = slice(max(ymin, 0), min(ymax, H))

                mx0 = max(0, -xmin)
                my0 = max(0, -ymin)
                mx1 = mw - max(0, xmax - W)
                my1 = mh - max(0, ymax - H)

                # Apply the color mask to the image
                img[yslice, xslice] += ci * mask[my0:my1, mx0:mx1, None]

        # Normalize and clip the image
        img = np.clip(img, 0, 1)[::-1, :, :]  # origin upper

        # Display the image in the figure space
        self.fig.figimage(img, origin='upper', zorder=5)
        self.fig.canvas.draw_idle()


class DynamicAdditiveScatterFig:
    def __init__(self, ax, s=100):
        """
        Fixed-size additive scatter (in screen pixels).
        Parameters:
        - ax: Matplotlib Axes
        - s: size in points^2 (like scatter)
        """
        self.ax = ax
        self.fig = ax.figure
        self.s = s
        self.points = []  # List of (x, y, color, alpha)
        self.canvas = ax.figure.canvas
        self.img_artist = None

        self.ax.callbacks.connect('xlim_changed', self._on_change)
        self.ax.callbacks.connect('ylim_changed', self._on_change)
        self._needs_redraw = True

    def add_points(self, x, y, color, alpha=None):
        x = np.asarray(x)
        y = np.asarray(y)
        n = len(x)
        color = np.asarray(color).reshape(1, 3)
        color = np.tile(color, (n, 1))
        if alpha is None:
            alpha = np.ones(n)
        else:
            alpha = np.asarray(alpha)
        color = color * alpha[:, None]
        self.points.append((x, y, color))
        self._needs_redraw = True

    def _on_change(self, event):
        self._needs_redraw = True
        self.render()

    def render(self):
        if not self._needs_redraw:
            return

        self.fig.canvas.draw()  # ensure correct size
        W, H = self.fig.canvas.get_width_height()
        img = np.zeros((H, W, 3), dtype=np.float32)
        print('H',H,'W',W)

        # Get the bounding box of the axes in display (pixel) coords
        bbox = self.ax.get_window_extent()

        # Convert bbox to integers
        left, bottom, width, height = map(int, bbox.bounds)
        print('height',height,'width',width)


        # Calculate pixel radius from scatter size
        radius_pt = np.sqrt(self.s / np.pi)
        radius_px = int(np.ceil(radius_pt * self.fig.dpi / 72))
        feather = 0.0  # 1 screen pixel feather

        # Create feathered disk mask
        rr = np.arange(-radius_px - 1, radius_px + 2)
        dx, dy = np.meshgrid(rr, rr)
        dist = np.sqrt(dx**2 + dy**2)
        #mask = np.clip(1 - (dist - radius_px) / feather, 0, 1)
        if feather <= 0:
            mask = (dist <= radius_px).astype(float)
        else:
            mask = np.clip(1 - (dist - radius_px) / feather, 0, 1)
        if s<1e-4:
            radius_px = 0  # just one pixel
            mask = np.ones((1, 1), dtype=np.float32)  # no feather

        for x_arr, y_arr, color_arr in self.points:
            xy_pixels = self.ax.transData.transform(np.vstack([x_arr, y_arr]).T)
            x_pix = xy_pixels[:, 0].astype(int)
            y_pix = (H - xy_pixels[:, 1]).astype(int)  # flip y for image coords

            for xi, yi, ci in zip(x_pix, y_pix, color_arr):
                for i in range(mask.shape[0]):
                    for j in range(mask.shape[1]):
                        xx = xi + j - radius_px - 1
                        yy = yi + i - radius_px - 1
                        if 0 <= xx < W and 0 <= yy < H:
                            img[yy, xx] += ci * mask[i, j]

        if self.img_artist:
            self.img_artist.set_data(np.clip(img, 0, 1))
        else:
            self.img_artist = self.fig.figimage(np.clip(img, 0, 1), origin='upper')

        self._needs_redraw = False
        self.fig.canvas.draw_idle()


class DynamicAdditiveScatter:
    def __init__(self, ax, s=10):
        """
        Additive scatter renderer with fixed-size (in screen pixels) feathered disks.
        Parameters:
        - ax: Matplotlib Axes
        - s: marker size in points^2 (like in plt.scatter)
        """
        self.ax = ax
        self.fig = ax.figure
        self.canvas = self.fig.canvas
        self.s = s
        self.points = []  # list of (x, y, color, alpha)
        self._img_artist = None
        self._cached_img = None

        # Attach callbacks to auto-update on zoom/pan
        ax.callbacks.connect('xlim_changed', self._on_view_change)
        ax.callbacks.connect('ylim_changed', self._on_view_change)

    def add_points(self, x, y, color, alpha=None):
        x = np.asarray(x)
        y = np.asarray(y)
        n = len(x)
        color = np.asarray(color).reshape(1, 3)
        color = np.tile(color, (n, 1))
        if alpha is None:
            alpha = np.ones(n)
        else:
            alpha = np.asarray(alpha)
        color = color * alpha[:, None]
        self.points.append((x, y, color))

    def _on_view_change(self, event):
        self.render()  # refresh on zoom/pan

    def _make_disk(self, radius_px, feather=1.0):
        r = int(np.ceil(radius_px + feather))
        y, x = np.mgrid[-r:r+1, -r:r+1]
        dist = np.sqrt(x**2 + y**2)
        mask = np.clip(1 - (dist - radius_px) / feather, 0, 1)
        return mask

    def render(self):
        if not self.points:
            return

        # Get screen dimensions
        self.canvas.draw()
        W, H = self.canvas.get_width_height()
        img = np.zeros((H, W, 3), dtype=np.float32)
        print('H',H,'W',W)

        # Get the bounding box of the axes in display (pixel) coords
        bbox = self.ax.get_window_extent()

        # Convert bbox to integers
        left, bottom, width, height = map(int, bbox.bounds)
        print('height',height,'width',width)

        # Compute pixel radius from point size in pt²
        radius_pt = np.sqrt(self.s / np.pi)
        radius_px = radius_pt * self.fig.dpi / 72
        mask = self._make_disk(radius_px, feather=1.0)

        mh, mw = mask.shape
        off_y, off_x = mh // 2, mw // 2

        for x_arr, y_arr, color_arr in self.points:
            xy_pix = self.ax.transData.transform(np.vstack([x_arr, y_arr]).T)
            x_pix = xy_pix[:, 0].astype(int)
            y_pix = (H - xy_pix[:, 1]).astype(int)  # flip y

            for xi, yi, ci in zip(x_pix, y_pix, color_arr):
                xmin = xi - off_x
                xmax = xi + off_x + 1
                ymin = yi - off_y
                ymax = yi + off_y + 1

                if xmax < 0 or ymax < 0 or xmin >= W or ymin >= H:
                    continue  # out of bounds

                xslice = slice(max(xmin, 0), min(xmax, W))
                yslice = slice(max(ymin, 0), min(ymax, H))

                mx0 = max(0, -xmin)
                my0 = max(0, -ymin)
                mx1 = mw - max(0, xmax - W)
                my1 = mh - max(0, ymax - H)

                img[yslice, xslice] += ci * mask[my0:my1, mx0:mx1, None]

        # Normalize and flip vertically
        img = np.clip(img, 0, 1)[::-1, :, :]  # origin upper

        # Define extent based on current view limits
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        extent = (x0, x1, y0, y1)

        if self._img_artist is None:
            self._img_artist = self.ax.imshow(img, extent=extent, origin='upper',
                                              transform=self.ax.transData, interpolation='none', zorder=5)
        else:
            self._img_artist.set_data(img)
            self._img_artist.set_extent(extent)

        self.canvas.draw_idle()


class DynamicAdditiveScatterMask:
    def __init__(self, ax, s=100):
        """
        Fixed-size additive scatter (in screen pixels).
        Parameters:
        - ax: Matplotlib Axes
        - s: size in points^2 (like scatter)
        """
        self.ax = ax
        self.fig = ax.figure
        self.s = s
        self.points = []  # List of (x, y, color, alpha)
        self.canvas = ax.figure.canvas
        self.img_artist = None

        # Connect callbacks for zoom and pan events
        self.ax.callbacks.connect('xlim_changed', self._on_change)
        self.ax.callbacks.connect('ylim_changed', self._on_change)
        self._needs_redraw = True

    def add_points(self, x, y, color, alpha=None):
        """
        Add points to the scatter plot with color and alpha.
        """
        x = np.asarray(x)
        y = np.asarray(y)
        n = len(x)
        color = np.asarray(color).reshape(1, 3)
        color = np.tile(color, (n, 1))
        if alpha is None:
            alpha = np.ones(n)
        else:
            alpha = np.asarray(alpha)
        color = color * alpha[:, None]
        self.points.append((x, y, color))
        self._needs_redraw = True

    def _on_change(self, event):
        """
        Triggered on zoom/pan change to mark the figure as needing a redraw.
        """
        self._needs_redraw = True
        self.render()

    def render(self):
        """
        Renders the additive scatter points only when required.
        """
        if not self._needs_redraw:
            return

        self.fig.canvas.draw()  # ensure correct size
        W, H = self.fig.canvas.get_width_height()
        img = np.zeros((H, W, 3), dtype=np.float32)

        # Calculate pixel radius from scatter size
        radius_pt = np.sqrt(self.s / np.pi)
        radius_px = int(np.ceil(radius_pt * self.fig.dpi / 72))
        feather = 1.0  # 1 screen pixel feather

        # Create feathered disk mask for each point
        rr = np.arange(-radius_px - 1, radius_px + 2)
        dx, dy = np.meshgrid(rr, rr)
        dist = np.sqrt(dx**2 + dy**2)
        mask = np.clip(1 - (dist - radius_px) / feather, 0, 1)

        # Get the subplot bounds in figure coordinates
        subplot_bbox = self.ax.get_position().bounds
        subplot_x0, subplot_y0, subplot_width, subplot_height = subplot_bbox

        # Loop over all the points and their respective colors
        for x_arr, y_arr, color_arr in self.points:
            xy_pixels = self.ax.transData.transform(np.vstack([x_arr, y_arr]).T)
            x_pix = xy_pixels[:, 0].astype(int)
            y_pix = (H - xy_pixels[:, 1]).astype(int)  # flip y for image coords

            # Loop over each point and apply the feathered mask
            for xi, yi, ci in zip(x_pix, y_pix, color_arr):
                # Mask out points that fall outside the current subplot area
                if not (subplot_x0 * W <= xi <= (subplot_x0 + subplot_width) * W and
                        subplot_y0 * H <= yi <= (subplot_y0 + subplot_height) * H):
                    continue

                for i in range(mask.shape[0]):
                    for j in range(mask.shape[1]):
                        xx = xi + j - radius_px - 1
                        yy = yi + i - radius_px - 1
                        if 0 <= xx < W and 0 <= yy < H:
                            img[yy, xx] += ci * mask[i, j]

        # Set image data and show it using figimage
        if self.img_artist:
            self.img_artist.set_data(np.clip(img, 0, 1))
        else:
            self.img_artist = self.fig.figimage(np.clip(img, 0, 1), origin='upper')

        self._needs_redraw = False
        self.fig.canvas.draw_idle()


class AdditiveScatterCanvas:
    def __init__(self, ax, s=100, edge_feather=1.0):
        """
        Initializes the AdditiveScatterCanvas.
        
        Parameters:
        - ax: matplotlib axis
        - s: Size of the scatter points (area in points², like scatter's 's')
        - edge_feather: Feathering width in pixels for anti-aliasing (always 1 for zoom-in consistency)
        """
        self.ax = ax
        self.s = s  # size in points², as in scatter
        self.edge_feather = edge_feather  # always 1 pixel
        self.points = []  # list of (x, y, color) tuples

    def add_points(self, x, y, color, alpha=None):
        """
        Adds points to the canvas.

        Parameters:
        - x, y: arrays of coordinates
        - color: single RGB list like [1, 0, 0]
        - alpha: optional array of brightness multipliers (same length as x/y)
        """
        x = np.asarray(x)
        y = np.asarray(y)
        color = np.asarray(color)

        if color.ndim == 1:
            color = np.broadcast_to(color, (len(x), 3))
        
        if alpha is None:
            alpha = np.ones_like(x)
        
        alpha = np.asarray(alpha).reshape(-1, 1)
        color = color * alpha  # intensity scaled per point

        self.points.append((x, y, color))

    def render(self):
        """
        Renders the accumulated scatter points onto the canvas with additive blending.
        """
        # Get canvas pixel resolution
        fig = self.ax.figure
        renderer = fig.canvas.get_renderer()
        bbox = self.ax.get_window_extent(renderer=renderer)
        W, H = int(bbox.width), int(bbox.height)
        W, H = fig.canvas.get_width_height()
        print('H',H,'W',W)

        # Get the bounding box of the axes in display (pixel) coords
        bbox = self.ax.get_window_extent()

        # Convert bbox to integers
        left, bottom, width, height = map(int, bbox.bounds)
        print('height',height,'width',width)

        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        # Blank canvas
        img = np.zeros((H, W, 3), dtype=np.float32)

        # Size conversion: s is in pt², like scatter
        dpi = self.ax.figure.dpi
        area_pts = self.s
        radius_pts = np.sqrt(area_pts / np.pi)  # radius in points
        radius_px = radius_pts * dpi / 72       # radius in pixels
        r_pix = int(np.ceil(radius_px))
        feather_px = 0.0  # fixed 1 pixel feathering

        # Prepare 1-pixel feathered disk mask in pixel units
        rr = np.arange(-r_pix - 1, r_pix + 2)
        dx, dy = np.meshgrid(rr, rr)
        dist = np.sqrt(dx**2 + dy**2)
        if feather_px <= 0:
            mask = (dist <= radius_px).astype(float)
        else:
            mask = np.clip(1 - (dist - radius_px) / feather_px, 0, 1)
        
        if self.s<1e-4:
            radius_px = 0  # just one pixel
            mask = np.ones((1, 1), dtype=np.float32)  # no feather


        for x_arr, y_arr, color_arr in self.points:
            # Convert data coordinates to pixel space
            x_pix = ((x_arr - xlim[0]) / (xlim[1] - xlim[0]) * (W - 1)).astype(int)
            y_pix = ((ylim[1] - y_arr) / (ylim[1] - ylim[0]) * (H - 1)).astype(int)

            for xi, yi, ci in zip(x_pix, y_pix, color_arr):
                for i in range(mask.shape[0]):
                    for j in range(mask.shape[1]):
                        xx = xi + j - r_pix - 1
                        yy = yi + i - r_pix - 1
                        if 0 <= xx < W and 0 <= yy < H:
                            img[yy, xx] += ci * mask[i, j]

        # Render final image
        self.ax.imshow(np.clip(img, 0, 1), extent=(*xlim, *ylim), origin='lower', interpolation='bilinear')


class AdditiveScatterCanvasSquare:
    def __init__(self, ax, s=100, edge_feather=1.0):
        """
        Initializes the AdditiveScatterCanvas.
        
        Parameters:
        - ax: matplotlib axis
        - s: Size of the scatter points (area in points², like scatter's 's')
        - edge_feather: Feathering width in pixels for anti-aliasing (always 1 for zoom-in consistency)
        """
        self.ax = ax
        self.s = s  # size in points², as in scatter
        self.edge_feather = edge_feather  # always 1 pixel
        self.points = []  # list of (x, y, color) tuples

    def add_points(self, x, y, color, alpha=None):
        """
        Adds points to the canvas.

        Parameters:
        - x, y: arrays of coordinates
        - color: single RGB list like [1, 0, 0]
        - alpha: optional array of brightness multipliers (same length as x/y)
        """
        x = np.asarray(x)
        y = np.asarray(y)
        color = np.asarray(color)

        if color.ndim == 1:
            color = np.broadcast_to(color, (len(x), 3))
        
        if alpha is None:
            alpha = np.ones_like(x)
        
        alpha = np.asarray(alpha).reshape(-1, 1)
        color = color * alpha  # intensity scaled per point

        self.points.append((x, y, color))

    def render(self):
        """
        Renders the accumulated scatter points onto the canvas with additive blending.
        Fixes feather width based on zoom.
        """
        # Get canvas pixel resolution
        fig = self.ax.figure
        renderer = fig.canvas.get_renderer()
        bbox = self.ax.get_window_extent(renderer=renderer)
        W, H = int(bbox.width), int(bbox.height)

        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        # Calculate zoom factor based on axis limits
        x_range = xlim[1] - xlim[0]
        y_range = ylim[1] - ylim[0]
        
        # Here, we use the average zoom factor to scale feather width
        zoom_factor = max(W / x_range, H / y_range)

        # Blank canvas
        img = np.zeros((H, W, 3), dtype=np.float32)

        # Size conversion: s is in pt², like scatter
        dpi = self.ax.figure.dpi
        area_pts = self.s
        radius_pts = np.sqrt(area_pts / np.pi)  # radius in points
        radius_px = radius_pts * dpi / 72       # radius in pixels
        r_pix = int(np.ceil(radius_px))
        
        # Adjust the feather width based on zoom factor: constant 1px
        feather_px = 1.0 / zoom_factor  # Feather width scales inversely with zoom

        # Prepare 1-pixel feathered disk mask in pixel units
        rr = np.arange(-r_pix - 1, r_pix + 2)
        dx, dy = np.meshgrid(rr, rr)
        dist = np.sqrt(dx**2 + dy**2)
        mask = np.clip(1 - (dist - r_pix) / feather_px, 0, 1)

        for x_arr, y_arr, color_arr in self.points:
            # Convert data coordinates to pixel space
            x_pix = ((x_arr - xlim[0]) / (xlim[1] - xlim[0]) * (W - 1)).astype(int)
            y_pix = ((ylim[1] - y_arr) / (ylim[1] - ylim[0]) * (H - 1)).astype(int)

            for xi, yi, ci in zip(x_pix, y_pix, color_arr):
                for i in range(mask.shape[0]):
                    for j in range(mask.shape[1]):
                        xx = xi + j - r_pix - 1
                        yy = yi + i - r_pix - 1
                        if 0 <= xx < W and 0 <= yy < H:
                            img[yy, xx] += ci * mask[i, j]

        # Render final image
        self.ax.imshow(np.clip(img, 0, 1), extent=(*xlim, *ylim), origin='lower', interpolation='bilinear')



def scatter_additive_disk_buffer(img, x, y, colors, size=1, ax=None, edge_feather=1.0):
    """
    Draws anti-aliased additive disks into an existing image buffer.

    Parameters:
    - img: (H, W, 3) numpy array to draw on
    - x, y: arrays of data coordinates
    - colors: RGB colors (list of [r, g, b])
    - size: disk radius in pixels
    - edge_feather: controls softness of the edge
    - ax: matplotlib Axes for coordinate mapping
    """
    H, W, _ = img.shape
    x = np.array(x)
    y = np.array(y)
    colors = np.array(colors)

    # Axis limits
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Map to pixel coordinates
    x_pix = ((x - xlim[0]) / (xlim[1] - xlim[0]) * (W - 1)).astype(int)
    y_pix = ((ylim[1] - y) / (ylim[1] - ylim[0]) * (H - 1)).astype(int)

    # Precompute circular kernel with anti-aliasing
    r_range = np.arange(-size-1, size+2)
    dx, dy = np.meshgrid(r_range, r_range)
    dist = np.sqrt(dx**2 + dy**2)
    # Full intensity inside, fade near edge
    mask = np.clip(1 - (dist - size) / edge_feather, 0, 1)

    for xi, yi, ci in zip(x_pix, y_pix, colors):
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                xx = xi + j - size - 1
                yy = yi + i - size - 1
                if 0 <= xx < W and 0 <= yy < H:
                    img[yy, xx] += ci * mask[i, j]

def show_additive_image(ax, img):
    """Displays the accumulated RGB image on the given axes, after clipping."""
    img_clipped = np.clip(img, 0, 1)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.imshow(img_clipped, extent=(*xlim, *ylim), origin='lower')
