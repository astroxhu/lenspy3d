import cv2
import numpy as np
import matplotlib.pyplot as plt

class ImageMeasurer:
    def __init__(self, image_path):
        self.img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        self.scale_set = False
        self.scale_mm_per_px = None
        self.points = []
        self.measurements = []

        self.fig, self.ax = plt.subplots()
        self.crosshair_v = self.ax.axvline(color='gray', linestyle='--')
        self.crosshair_h = self.ax.axhline(color='gray', linestyle='--')

        self.cid_click = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.cid_move = self.fig.canvas.mpl_connect('motion_notify_event', self.onmove)
        self.cid_key = self.fig.canvas.mpl_connect('key_press_event', self.onkey)
        
        self.inset_ax = None
        self.inset_zoom = 10     # Magnification factor
        self.inset_size = 20     # Half-size of zoomed region in pixels (will be 40x40 box)


    def to_image_coords(self, event):
        # Get bounding box of the axes in display coordinates
        bbox = self.ax.get_window_extent()

        # Normalize click position within axes (0–1)
        x_norm = (event.x - bbox.x0) / bbox.width
        y_norm = (event.y - bbox.y0) / bbox.height

        # Flip Y (image origin is top-left)
        y_norm = 1 - y_norm

        h, w, _ = self.img.shape
        x_px = x_norm * w
        y_px = y_norm * h

        return x_px, y_px

    def onclick(self, event):
        if event.button != 1 or event.inaxes != self.ax:
            return

        xdata, ydata = self.to_image_coords(event)

        h, w, _ = self.img.shape
        if not (0 <= xdata <= w and 0 <= ydata <= h):
            return

        self.points.append((xdata, ydata))
        self.ax.plot(xdata, ydata, 'ro')
        self.ax.text(xdata + 5, ydata + 5, f"P{len(self.points)}", color='red', fontsize=8)

        if not self.scale_set and len(self.points) == 2:
            self.set_scale()
        elif self.scale_set and len(self.points) % 2 == 0:
            self.measure_distance()

        self.fig.canvas.draw()


    def set_scale(self):
        p1, p2 = self.points[:2]
        px_dist = np.hypot(p1[0] - p2[0], p1[1] - p2[1])
        known_mm = float(input("Enter known distance in mm (e.g. 100): "))
        self.scale_mm_per_px = known_mm / px_dist
        self.scale_set = True
        print(f"[INFO] Scale set: 1 pixel = {self.scale_mm_per_px:.4f} mm")

    def measure_distance(self):
        p1, p2 = self.points[-2:]
        px_dist = np.hypot(p1[0] - p2[0], p1[1] - p2[1])
        mm_dist = px_dist * self.scale_mm_per_px
        self.measurements.append(mm_dist)
        print(f"[MEASURED] Distance = {mm_dist:.2f} mm")

        self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'g-')
        mid = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
        self.ax.text(mid[0], mid[1], f"{mm_dist:.2f} mm", color='blue', fontsize=9)
        self.fig.canvas.draw()

    def onmove(self, event):
        if not event.inaxes:
            return

        xdata, ydata = self.to_image_coords(event)

        # Update crosshairs
        self.crosshair_v.set_xdata(xdata)
        self.crosshair_h.set_ydata(ydata)

        # Draw magnifier
        self.draw_inset(xdata, ydata)

        self.fig.canvas.draw_idle()

    def onkey(self, event):
        if event.key == 'u':
            self.undo_last_point()
        elif event.key == 'r':
            self.reset()

    def undo_last_point(self):
        if not self.points:
            return
        self.points.pop()
        self.scale_set = False if len(self.points) < 2 else self.scale_set
        self.redraw()
        print("[UNDO] Last point removed")

    def reset(self):
        self.points.clear()
        self.measurements.clear()
        self.scale_set = False
        self.ax.clear()
        self.ax.imshow(self.img)
        self.crosshair_v = self.ax.axvline(color='gray', linestyle='--')
        self.crosshair_h = self.ax.axhline(color='gray', linestyle='--')
        self.fig.canvas.draw()
        print("[RESET] All points and measurements cleared")

    def run(self):
        self.ax.imshow(self.img)
        self.ax.set_title("Click to set scale (2 pts), then measure (2 pts per distance)\nKeys: 'u' undo, 'r' reset")
        plt.show()
    def draw_inset(self, x, y):
        h, w, _ = self.img.shape
        x, y = int(x), int(y)

        hs = self.inset_size
        zoom = self.inset_zoom

        x0, x1 = max(0, x - hs), min(w, x + hs)
        y0, y1 = max(0, y - hs), min(h, y + hs)

        patch = self.img[y0:y1, x0:x1]

        if patch.size == 0:
            return

        if self.inset_ax is None:
            from mpl_toolkits.axes_grid1.inset_locator import inset_axes
            self.inset_ax = inset_axes(self.ax, width="20%", height="20%", loc='upper right', borderpad=1)
            self.inset_ax.set_xticks([])
            self.inset_ax.set_yticks([])
            self.inset_ax.set_title("Zoom")

        self.inset_ax.clear()
        self.inset_ax.imshow(patch, interpolation='none', extent=[0, patch.shape[1], patch.shape[0], 0])
        self.inset_ax.set_xlim(0, patch.shape[1])
        self.inset_ax.set_ylim(patch.shape[0], 0)

        cx, cy = patch.shape[1] // 2, patch.shape[0] // 2
        self.inset_ax.plot([cx], [cy], 'r+', markersize=10)



# ---- Run the tool ----
if __name__ == "__main__":
    image_path = '../lens_diagrams/sigma556_dewey.png'  # Replace with your actual image path
    tool = ImageMeasurer(image_path)
    tool.run()

