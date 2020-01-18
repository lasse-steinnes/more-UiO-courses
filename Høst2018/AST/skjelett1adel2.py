import numpy as np
import matplotlib.pyplot as plt
import sys
import random as random
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation


class part1:
    def __init__(self):
        self.H2gPrMol = 2.01588                          # Hydrogen gass weight [g/mol]
        self.N_a = 6.02214179e23                         # Avogadros number

        self.mass_sat = 1100                             # Satellite/Rocket mass [kg]
        self.k = 1.380648e-23                            #Boltzmann's constant [J/K]
        self.mass1Particle = self.H2gPrMol/self.N_a*1e-3 # [kg]

    def simulate_box(self, n, Nt, dt, box_dim, T):

        """
        Calculates the motion of the particles used for fuel calculations
        variables:


        n  =        number of particles
        Nt =        Number of time steps

        dt =        Time step
        box_dim =   Dimension of the box
        """


        sigma = #...     # Standard deviation for initial gaussian velocity distribution
        mean = 0                                         # Mean for initial gaussian distribution (no bulk flow)

        pos = np.zeros((n, 3, Nt))                       # Position vector

        vel = np.zeros((n, 3))                           # Velocity vector


        """ Setting initial positions and velocities. Positions are selected from a uniform distribution while
        velocities are selected from a gaussian distribution with a mean and deviation described in part1. """

        for j in range(n):
            for k in range(3):
                pos[j, k, 0] = #... uniform dist.
                vel[j, k] = #... Gauss dist.


        x_momentum = 0.0                            # Momentum value scalar in x-direction
        no_escapes = 0                              # Counter for number of particles escaped


        for i in range(Nt-1): #Time loop
            pos[:,:,i+1] = pos[:,:,i] + vel[:,:]*dt #Move ALL particles one step

            #If you are really skilled with vectorization,
            #It's possible to vectorize the particle loop too (hint: masking)
            for j in range(n): # Particle loop
                """ Here you correct for collisions with the walls.

                    You might need som if-else tests to make sure you cover all possible collisions """


                #Potential solution
                """

                #Check if particle is outside of any boundary
                if pos[j, 0, i+1] >= box_dim: # we are outside the box in x-direction
                    if vel[j, 0] > 0:         # the velocity is pointing outwards
                        #Here turn the direction of the velocity (bounce particle)

                        #This wall is special, it has a hole! Check if we hit it.
                        if (((pos[j,1,i+1] > hole_min) and (pos[j,1,i+1] < hole_max)) # within hole in y-direction
                        and (...) ): #particle is within hole in z-directin
                            no_escapes += 1   #One particle escaped
                            x_momentum += ... #Here calculate additional momentum
                    else:
                        continue         #particle already bounced, don't turn it.

                elif pos[j,0, i+1] <= 0:     #Similar test for negative x-dir
                    if vel[j, 0] < 0:         # the velocity is pointing outwards
                        #Turn the direction of the velocity (bounce particle)
                    else:
                        continue         #particle already bounced!

                elif pos[j,1, i+1] >= box_dim: # positive y-dir
                    ...
                ...
                """




        return x_momentum, pos


    def launch_sim(self, no_boxes, v_end, A, no_particles_sec):
        """
        Simulates the launch.

        Variables:
        no_boxes:           the number of fuel boxes you need
        v_end:              the desired velocity (relative to surface) at the end of launch
        A:                  dp/dt for the box described in first method
        no_particles_sec:   number of particles pr second leaving the box in first method

        returns:    How much time it takes to achieve the boost and how much fuel you have used up.
        """

        A = A*no_boxes                                   # (dp/dt) for whole engine
        B = 0. #(insert value)  # dm/dt for the box described in first method
        v = 0.                  # initial velocity relative to surface of planet
        time = 0.               # initialize time variable

        T = 20.*60                                       # Total time, 20 minutes
        Nt = 10000                                       # Number of time steps

        dt = float(T)/Nt                                 # Time step

        initial_fuel_mass = 100                          # Calculate/set fuel mass
        M = self.mass_sat + initial_fuel_mass            # Total mass


        for i in xrange(Nt):
            """ Here you need to update the new value of the velocity aswell as the new total mass value M """
            #v += boost*dt - gravity ...
            #M -= ...
            time += dt
            if M < self.mass_sat:
                """ Some sort of error message telling you you're out of fuel """
            elif v < 0:
                """ Gravity stronger than engine """
            elif v >= v_end:
                """ Boost is successful. Save values and end method """
                #fuel_needed = ...
                print "Boost Succesful"
                return fuel_needed, time

        return 0., 0. #returns 0 because the boost was not successful.


    def plot(self, n, Nt, dt, box_dim, T):
        """ A method for plotting and animating a simulation of particles in box """


        def update_lines(num, dataLines, lines) :
            for line, data in zip(lines, dataLines) :
                line.set_data(data[0:2, num-1:num])
                line.set_3d_properties(data[2,num-1:num])
            return lines

        # Attach 3D axis to the figure
        fig = plt.figure()
        ax = p3.Axes3D(fig)

        m = 100

        #Run the actual simulation
        x_momentum, datax = self.simulate_box(n, Nt, dt, box_dim, T)
        lines = []
        data = []
        for i in range(n):
            data.append( [datax[i]]) #wrap data inside another layer of [], needed for animation!
            lines.append([ax.plot(data[i][0][0,0:1], data[i][0][1,0:1], data[i][0][2,0:1], 'o')[0]])

        # Set the axes properties
        ax.set_xlim3d([0.0, box_dim])
        ax.set_xlabel('X')

        ax.set_ylim3d([0.0, box_dim])
        ax.set_ylabel('Y')

        ax.set_zlim3d([0.0, box_dim])
        ax.set_zlabel('Z')


        ax.set_title('Particle Animation')

        # Creating the Animation object
        ani = [i for i in range(n)] #"Initialize" a list of length n
        for i in range(n):
            #This is the method that needs the elements of data and lines to be wrapped in two layers of []

            ani[i] = animation.FuncAnimation(fig, update_lines, m, fargs=(data[i], lines[i]),
                                      interval=50, blit=False)
        plt.show()



instance = part1()
instance.plot(box_dim = 1e-6, n = 100, Nt = 1000, dt = 1e-12, T = 10000)
