__author__ = 'Bo'

"""REMUS_2D.py

Class REMUS is defined in the module. The vehicle moves on a horizontal plane in a uniform and steady flow.
No control system is involved.

"""

import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.transforms as transforms
import shelve


class REMUS:
    """REMUS Class

    Ocean current is considered in the model, whose velocity is steady and uniform.

    """

    # define the parameters of the vehicle, which are class attributes
    mass = 3.05e1  # total mass of the vehicle (unit: kg)
    weight = 2.99e2  # total weight of the vehicle (unit: N)
    buoyancy = 3.06e2  # the buoyancy of the vehicle (unit: N)
    cog = (0, 0, 1.96e-2)  # the center of gravity of the vehicle (unit: m)
    actuation = np.asarray([9, 0, 0], dtype='float')  # the constant control force on the vehicle
    Izz = 3.45
    a11, a22, a26, a66 = (9.3e-1, 3.55e1, -1.93, 4.88)
    Xuu, Yvv, Yrr, Nvv, Nrr = (-3.87, -1e2, 6.32e-1, 7.38, -9.4e1)
    Yuv, Yur, Nuv, Nur = (-1.79e1, 8.62, 4.57, -6.4)

    # define the parameter of the ocean current
    ocean_current = np.asarray([0, 0., 0], dtype='float')  # the velocity in the earth-fixed frame

    def __init__(self, init_velocity=(1.0, 0, 0), init_position=(1, 0, 0), step_size = 0.05, step_number = 6000,
                 animation_on = False):

        # set the initial states of the vehicle
        self.init_velocity = np.asarray(init_velocity, dtype='float')
        self.init_position = np.asarray(init_position, dtype='float') * np.asarray((1, 1, np.pi/180.0), dtype='float')

        # define the states of the vehicle
        self.time_elapsed = 0.0
        self.velocity = self.init_velocity  # the velocity vector of the vehicle in the body-fixed frame
        self.position = self.init_position  # the position vector of the vehicle in earth-fixed frame
        self.relative_velocity = self.velocity \
            - np.linalg.inv(self.transformation_matrix(self.position)).dot(REMUS.ocean_current)

        # define the parameters of the simulation
        self.step_counter = 0
        self.step_size = float(step_size)
        self.step_number = step_number
        self.time_vec = 0.1 * np.asarray(range(self.step_number+1))

        # define the variables that store the histories of the position and velocity
        self.position_history = np.vstack((np.asarray([self.init_position]), np.zeros((self.step_number, 3))))
        self.velocity_history = np.vstack((np.asarray([self.init_velocity]), np.zeros((self.step_number, 3))))
        self.relative_velocity_history = np.vstack((np.asarray([self.relative_velocity]),
                                                    np.zeros((self.step_number, 3))))

        # define the figure of the animation
        self.animation_on = animation_on
        if self.animation_on:
            self.fig = plt.figure(figsize=(12, 8), dpi=100)
            self.ax = self.fig.add_subplot(111, aspect='equal', autoscale_on=False, xlim=(-20, 20), ylim=(-20, 20))
            self.center, = self.ax.plot([], [], 'rx', markersize=6)
            self.trajectory, = self.ax.plot([], [], 'b-', lw=1)
            self.time_text = self.ax.text(0.02, 0.95, '', transform=self.ax.transAxes)

            if np.linalg.norm(REMUS.ocean_current):
                self.ocean_text = self.ax.text(0.80, 0.1, 'Ocean Current', transform=self.ax.transAxes, color='b',
                                               bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
                self.ocean_arrow = plt.Arrow(0.90, 0.05, 0.2*REMUS.ocean_current[1], 0.2*REMUS.ocean_current[0],
                                             facecolor='b', edgecolor='b', width=0.05, transform=self.ax.transAxes)
            else:
                self.ocean_text = self.ax.text(0.75, 0.1, 'No Ocean Current', transform=self.ax.transAxes, color='b',
                                               bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})

            self.ax.grid()
            self.ax.set_xlabel(r"Longitude (m)")
            self.ax.set_ylabel(r"Latitude (m)")
            self.ax.set_title(r"The trajectory of the vehicle")

            self.t_start = self.ax.transData
            self.coords = self.t_start.transform([self.position[1], self.position[0]])
            self.arrow = plt.Arrow(self.position[1], self.position[0], 0, 2,
                                   facecolor='r', edgecolor='b', width=1.5, animated=True)

    @staticmethod
    def transformation_matrix(position):
        """compute the transformation matrix of the vehicle"""
        return np.asarray([[np.cos(position[2]), -np.sin(position[2]), 0],
                           [np.sin(position[2]), np.cos(position[2]), 0],
                           [0, 0, 1]], dtype='float')

    @staticmethod
    def rigid_body_mass_matrix():
        """compute the mass matrix of the rigid body"""
        return np.asarray([[REMUS.mass, 0, -REMUS.mass * REMUS.cog[1]],
                           [0, REMUS.mass, REMUS.mass * REMUS.cog[0]],
                           [-REMUS.mass * REMUS.cog[1], REMUS.mass * REMUS.cog[0], REMUS.Izz]],
                          dtype='float')

    @staticmethod
    def rigid_body_cc_matrix(velocity):
        """compute the centripetal-coriolis matrix of the rigid body"""
        return np.asarray([[0, -REMUS.mass * velocity[2], -REMUS.mass * velocity[2] * REMUS.cog[0]],
                           [REMUS.mass * velocity[2], 0, -REMUS.mass * velocity[2] * REMUS.cog[1]],
                           [REMUS.mass * velocity[2] * REMUS.cog[0],
                            REMUS.mass * velocity[2] * REMUS.cog[1], 0]], dtype='float')

    @staticmethod
    def added_mass_matrix():
        """compute the added mass matrix of the rigid body"""
        return np.asarray([[REMUS.a11, 0, 0],
                           [0, REMUS.a22, REMUS.a26],
                           [0, REMUS.a26, REMUS.a66]],
                          dtype='float')

    @staticmethod
    def added_mass_cc_matrix(velocity):
        """compute the centripetal-coriolis matrix of the added mass"""
        return np.asarray([[0, 0, -REMUS.a26 * velocity[2] - REMUS.a22 * velocity[1]],
                           [0, 0, REMUS.a11 * velocity[0]],
                           [REMUS.a26 * velocity[2] + REMUS.a22 * velocity[1],
                            -REMUS.a11 * velocity[0], 0]],
                          dtype='float')

    @staticmethod
    def viscous_force(velocity):
        """compute the viscous hydrodynamic loads on the vehicle"""
        return np.asarray([REMUS.Xuu * velocity[0] * np.absolute(velocity[0]),
                           REMUS.Yvv * velocity[1] * np.absolute(velocity[1]) +
                           REMUS.Yrr * velocity[2] * np.absolute(velocity[2]) +
                           REMUS.Yuv * velocity[0] * velocity[1] +
                           REMUS.Yur * velocity[0] * velocity[2],
                           REMUS.Nvv * velocity[1] * np.absolute(velocity[1]) +
                           REMUS.Nrr * velocity[2] * np.absolute(velocity[2]) +
                           REMUS.Nuv * velocity[0] * velocity[1] +
                           REMUS.Nur * velocity[0] * velocity[2]], dtype='float')

    def dvelocity_dt(self, velocity, t):
        """compute the derivative of the given velocity"""
        return -np.linalg.inv(self.added_mass_matrix() + self.rigid_body_mass_matrix()).dot(
            self.rigid_body_cc_matrix(velocity).dot(velocity) + self.added_mass_cc_matrix(velocity).dot(velocity) -
            self.viscous_force(velocity) - REMUS.actuation
        )

    def dposition_dt(self, position, t):
        """compute the derivative of the given position"""
        return self.transformation_matrix(position).dot(self.velocity)

    def step(self):
        """execute one time step of length dt and update the velocity and position of the vehicle"""
        self.position = integrate.odeint(self.dposition_dt, self.position, [0, self.step_size])[1]
        self.relative_velocity = integrate.odeint(self.dvelocity_dt, self.relative_velocity, [0, self.step_size])[1]
        self.velocity = self.relative_velocity \
            + np.linalg.inv(self.transformation_matrix(self.position)).dot(REMUS.ocean_current)
        self.time_elapsed += self.step_size
        self.step_counter += 1
        self.position_history[self.step_counter] = self.position
        self.velocity_history[self.step_counter] = self.velocity
        self.relative_velocity_history[self.step_counter] = self.relative_velocity

    def animation_init(self):
        """initialize animation"""
        self.center.set_data([], [])
        self.trajectory.set_data([], [])
        self.time_text.set_text('')
        self.ax.add_patch(self.arrow)
        if np.linalg.norm(REMUS.ocean_current):
            self.ax.add_patch(self.ocean_arrow)
        return self.center, self.trajectory, self.time_text, self.arrow

    def animate(self, frame_number):
        """perform animation step"""
        self.step()

        # update the center and trajectory of the vehicle
        self.center.set_data(self.position[1], self.position[0])

        self.trajectory.set_data(self.position_history[:self.step_counter, 1],
                                 self.position_history[:self.step_counter, 0])

        # update the time displayed
        self.time_text.set_text('time = %.3f sec' % self.time_elapsed)

        # update the arrow which represents the vehicle
        angle = -self.position[2] * 180/np.pi

        tr = transforms.Affine2D().rotate_deg_around(self.coords[0], self.coords[1], angle)

        trans = self.t_start.transform([self.position[1], self.position[0]]) - self.coords
        tr = tr.translate(trans[0], trans[1])

        t_end = self.t_start + tr

        self.arrow.set_transform(t_end)

        return self.center, self.trajectory, self.time_text, self.arrow

# -------------------------------------------------------------------------------
# The following codes are used to test the class defined above
if __name__ == "__main__":

#     # define an instance of REMUS
#     vehicle = REMUS((1.0, 0.05, 0))
#
# # -------------------------------------------------------------------------------
# # simulate the motion of the vehicle
#     for step in xrange(0, vehicle.step_number):
#         vehicle.step()
#
# # -------------------------------------------------------------------------------
# # visualize the velocity history
#     plt.figure(figsize=[10, 6])
#     plt.plot(vehicle.time_vec, vehicle.velocity_history[:, 0],
#              color="blue", linewidth=2.5, linestyle="-", label=r"$u$")
#     plt.plot(vehicle.time_vec, vehicle.velocity_history[:, 1],
#              color="green", linewidth=2.5, linestyle="-", label=r"$v$")
#     plt.plot(vehicle.time_vec, vehicle.velocity_history[:, 2],
#              color="red", linewidth=2.5, linestyle="-", label=r"$r$")
#     plt.xlabel(r"time (sec)")
#     plt.ylabel(r"velocity")
#     plt.legend(loc='lower right')
#     plt.grid(True)
#
# # visualize the heading angle history
#     plt.figure(figsize=[10, 6])
#     plt.plot(vehicle.time_vec, np.fmod(vehicle.position_history[:, 2] * 180/np.pi, np.asarray([360])),
#              color="blue", linewidth=2.5, linestyle="-")
#     plt.xlabel(r"time (sec)")
#     plt.ylabel(r"heading angle (deg)")
#     plt.title(r"The heading angle of the vehicle")
#     plt.grid(True)
#
# # visualize the trajectory
#     plt.figure(figsize=[10, 10])
#     plt.plot(vehicle.position_history[:, 1], vehicle.position_history[:, 0], color="blue", linewidth=2.5, linestyle="-")
#     plt.xlabel(r"Longitude (m)")
#     plt.ylabel(r"Latitude (m)")
#     plt.title(r"The trajectory of the vehicle on the horizontal plane")
#     plt.grid(True)
#     plt.show()
#
# # -------------------------------------------------------------------------------
# # save the instance into a file
#     dbase = shelve.open('vehicle_14.db')
#     try:
#         dbase['vehicle'] = vehicle
#     finally:
#         dbase.close()

# -------------------------------------------------------------------------------
    # define an instance of REMUS
    vehicle = REMUS((1.0, 0.1, 0.2), animation_on=True)
    # display the animation of the vehicle's trajectory
    if vehicle.animation_on:
        interval = 900 * vehicle.step_size

        ani = animation.FuncAnimation(vehicle.fig, vehicle.animate, interval=interval, blit=True,
                                      init_func=vehicle.animation_init)
        plt.show()
    else:
        print("Error: Animation not available, please turn on the animation when initializing the instance!")
