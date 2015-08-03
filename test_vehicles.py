__author__ = 'Bo'

"""test_vehicles.py

A few tests are conducted to the class defined in REMUS_2D.py

"""
from REMUS_2D import REMUS
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.transforms as transforms


# ------------------------------------------------------------
# create instances of REMUS vehicles
vehicle_1 = REMUS((1.0, 0.2, 0.))
vehicle_2 = REMUS((1.5, -0.2, 0.), (5, 0, 0))
# ------------------------------------------------------------
# set up figure and animation
fig = plt.figure(figsize=(12, 8), dpi=100)
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False, xlim=(-20, 20), ylim=(-20, 20))
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

if np.linalg.norm(REMUS.ocean_current):
    ocean_text = ax.text(0.80, 0.1, 'Ocean Current', transform=ax.transAxes, color='b',
                         bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
    ocean_arrow = plt.Arrow(0.90, 0.05, 0.2*REMUS.ocean_current[1], 0.2*REMUS.ocean_current[0],
                            facecolor='b', edgecolor='b', width=0.05, transform=ax.transAxes)
    ax.add_patch(ocean_arrow)
else:
    ocean_text = ax.text(0.75, 0.1, 'No Ocean Current', transform=ax.transAxes, color='b',
                         bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})

ax.grid()
ax.set_xlabel(r"Longitude (m)")
ax.set_ylabel(r"Latitude (m)")
ax.set_title(r"The trajectory of the vehicle")

t_start = ax.transData

center_1, = ax.plot([], [], 'rx', markersize=6)
trajectory_1, = ax.plot([], [], 'b-', lw=1)
coords_1 = t_start.transform([vehicle_1.position[1], vehicle_1.position[0]])
arrow_1 = plt.Arrow(vehicle_1.position[1], vehicle_1.position[0], 0, 2, 
                    facecolor='r', edgecolor='b', width=1.5, animated=True)				   


center_2, = ax.plot([], [], 'rx', markersize=6)
trajectory_2, = ax.plot([], [], 'b-', lw=1)
coords_2 = t_start.transform([vehicle_2.position[1], vehicle_2.position[0]])
arrow_2 = plt.Arrow(vehicle_2.position[1], vehicle_2.position[0], 0, 2, 
                    facecolor='r', edgecolor='b', width=1.5, animated=True)


def animation_init():
    """initialize animation"""
    time_text.set_text('')

    def init(center, trajectory, arrow):
        center.set_data([], [])
        trajectory.set_data([], [])
        ax.add_patch(arrow)
        return center, trajectory, arrow

    return init(center_1, trajectory_1, arrow_1) + init(center_2, trajectory_2, arrow_2) + (time_text,)


def animate(frame_number):
    """perform animation step"""
    vehicle_1.step()
    vehicle_2.step()
    # update the time displayed
    # time_text.set_text('time = %.3f sec' % vehicle_1.time_elapsed)

    def update(vehicle, coords, center, trajectory, arrow):
        # update the center and trajectory of the vehicle
        center.set_data(vehicle.position[1], vehicle.position[0])
        trajectory.set_data(vehicle.position_history[:vehicle.step_counter, 1],
                            vehicle.position_history[:vehicle.step_counter, 0])
        # update the arrow which represents the vehicle
        angle = -vehicle.position[2] * 180/np.pi
        tr = transforms.Affine2D().rotate_deg_around(coords[0], coords[1], angle)
        trans = t_start.transform([vehicle.position[1], vehicle.position[0]]) - coords
        tr = tr.translate(trans[0], trans[1])
        t_end = t_start + tr

        arrow.set_transform(t_end)
        return center, trajectory, arrow

    return (update(vehicle_1, coords_1, center_1, trajectory_1, arrow_1)
            + update(vehicle_2, coords_2, center_2, trajectory_2, arrow_2)) + (time_text,)

interval = 900 * vehicle_1.step_size
ani = animation.FuncAnimation(fig, animate, interval=interval, blit=True,
                              init_func=animation_init)
plt.show()
