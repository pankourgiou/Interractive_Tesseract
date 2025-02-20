import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider, Button

# Function to create the 16 vertices of a tesseract (4D hypercube)
def create_tesseract_vertices():
    vertices = []
    for x in [-1, 1]:
        for y in [-1, 1]:
            for z in [-1, 1]:
                for w in [-1, 1]:
                    vertices.append([x, y, z, w])
    return np.array(vertices)

# Function to create edges: two vertices are connected if they differ in exactly one coordinate.
def create_tesseract_edges(vertices):
    edges = []
    n = len(vertices)
    for i in range(n):
        for j in range(i + 1, n):
            if np.sum(np.abs(vertices[i] - vertices[j])) == 2:
                edges.append((i, j))
    return edges

# 4D rotation in the xw plane
def rotate_4d_xw(vertices, theta):
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([
        [c, 0, 0, -s],
        [0, 1, 0,  0],
        [0, 0, 1,  0],
        [s, 0, 0,  c]
    ])
    return vertices.dot(R.T)

# 4D rotation in the yw plane
def rotate_4d_yw(vertices, theta):
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([
        [1,  0, 0,  0],
        [0,  c, 0, -s],
        [0,  0, 1,  0],
        [0,  s, 0,  c]
    ])
    return vertices.dot(R.T)

# Perspective projection from 4D to 3D
def project_4d_to_3d(vertices, distance=4):
    projected = []
    for v in vertices:
        factor = distance / (distance - v[3])
        projected.append([v[0] * factor, v[1] * factor, v[2] * factor])
    return np.array(projected)

# Initialize tesseract vertices and edges
vertices = create_tesseract_vertices()
edges = create_tesseract_edges(vertices)

# Set up the figure and a 3D axes for the visualization
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(left=0.1, bottom=0.3)  # Reserve space at the bottom for sliders

def draw_tesseract(ax, theta_xw, theta_yw, distance):
    ax.cla()  # Clear the current axes
    ax.set_xlim([-4, 4])
    ax.set_ylim([-4, 4])
    ax.set_zlim([-4, 4])
    ax.set_box_aspect([1, 1, 1])
    ax.set_title("Interactive Folding Tesseract")

    # Apply 4D rotations based on the slider values
    rotated = rotate_4d_xw(vertices, theta_xw)
    rotated = rotate_4d_yw(rotated, theta_yw)
    
    # Project the rotated 4D vertices into 3D space
    projected = project_4d_to_3d(rotated, distance)

    # Draw each edge of the tesseract
    for edge in edges:
        start, end = edge
        xs = [projected[start, 0], projected[end, 0]]
        ys = [projected[start, 1], projected[end, 1]]
        zs = [projected[start, 2], projected[end, 2]]
        ax.plot(xs, ys, zs, color='blue', linewidth=1)

    # Optionally, plot the vertices as red dots
    ax.scatter(projected[:, 0], projected[:, 1], projected[:, 2], color='red')
    plt.draw()

# Set initial parameters for the rotations and perspective distance
initial_theta_xw = 0
initial_theta_yw = 0
initial_distance = 4

# Draw the initial tesseract
draw_tesseract(ax, initial_theta_xw, initial_theta_yw, initial_distance)

# Create sliders for interactive control
ax_theta_xw = plt.axes([0.1, 0.2, 0.8, 0.03])
slider_theta_xw = Slider(ax_theta_xw, 'xw Rotation', -np.pi, np.pi, valinit=initial_theta_xw)

ax_theta_yw = plt.axes([0.1, 0.15, 0.8, 0.03])
slider_theta_yw = Slider(ax_theta_yw, 'yw Rotation', -np.pi, np.pi, valinit=initial_theta_yw)

ax_distance = plt.axes([0.1, 0.1, 0.8, 0.03])
slider_distance = Slider(ax_distance, 'Distance', 2, 10, valinit=initial_distance)

# Update the drawing when any slider value changes
def update(val):
    theta_xw = slider_theta_xw.val
    theta_yw = slider_theta_yw.val
    distance = slider_distance.val
    draw_tesseract(ax, theta_xw, theta_yw, distance)

slider_theta_xw.on_changed(update)
slider_theta_yw.on_changed(update)
slider_distance.on_changed(update)

# Add a reset button to restore initial slider values
reset_ax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(reset_ax, 'Reset', hovercolor='0.975')
def reset(event):
    slider_theta_xw.reset()
    slider_theta_yw.reset()
    slider_distance.reset()
button.on_clicked(reset)

plt.show()
