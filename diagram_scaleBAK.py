import cv2
import matplotlib.pyplot as plt
import numpy as np

class ImageMeasurer:
    def __init__(self, image_path):
        self.img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        self.scale_set = False
        self.scale_mm_per_px = None
        self.points = []
        self.measurements = []
        self.fig, self.ax = plt.subplots()
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)

    def onclick(self, event):
        # Only proceed on left mouse button within axes
        if event.button != 1 or event.inaxes != self.ax:
            return

        if event.xdata is None or event.ydata is None:
            return  # Clicked outside the image

        self.points.append((event.xdata, event.ydata))
        self.ax.plot(event.xdata, event.ydata, 'ro')
        self.fig.canvas.draw()

        if not self.scale_set and len(self.points) == 2:
            self.set_scale()
        elif self.scale_set and len(self.points) % 2 == 0:
            self.measure_distance()

    def set_scale(self):
        p1, p2 = self.points[:2]
        px_dist = np.hypot(p1[0] - p2[0], p1[1] - p2[1])
        known_mm = float(input("Enter known distance in mm (e.g. 100 for the scale bar): "))
        self.scale_mm_per_px = known_mm / px_dist
        self.scale_set = True
        print(f"[INFO] Scale set: 1 pixel = {self.scale_mm_per_px:.4f} mm")

    def measure_distance(self):
        p1, p2 = self.points[-2:]
        px_dist = np.hypot(p1[0] - p2[0], p1[1] - p2[1])
        mm_dist = px_dist * self.scale_mm_per_px
        self.measurements.append(mm_dist)
        print(f"[MEASURED] Distance = {mm_dist:.2f} mm")

        # Draw line and label
        self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'g-')
        mid = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
        self.ax.text(mid[0], mid[1], f"{mm_dist:.2f} mm", color='blue', fontsize=8)
        self.fig.canvas.draw()

    def run(self):
        self.ax.imshow(self.img)
        self.ax.set_title("Click to set scale (2 points), then measure (2 points per segment)")
        plt.show()

# ---- Run the tool ----
if __name__ == "__main__":
    image_path = '../lens_diagrams/sigma556_dewey.png'  # Replace with your actual image path
    tool = ImageMeasurer(image_path)
    tool.run()

