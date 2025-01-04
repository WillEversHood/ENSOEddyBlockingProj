import numpy as np
import math as math
import scipy.integrate
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import KernelDensity
from RDOscilator import RDOscilator
import seaborn as sns
import pandas as pd
import sympy as sp

class Coupled():
    def __init__(self, aRD, a84, bRD, b84, c, T, h, F_0_RD, F_0_84, F_1_84, G_0_84, G_1_84, a_nx, a_ny, X, Y, Z, t, resolution):
        self.aRD = aRD
        self.a84 = a84
        self.bRD = bRD
        self.b84 = b84
        self.c = c
        self.T = T
        self.h = h
        self.F_0_RD = F_0_RD
        self.F_0_84 = F_0_84
        self.F_1_84 = F_1_84
        self.G_0_84 = G_0_84
        self.G_1_84 = G_1_84
        self.a_nx = a_nx
        self.a_ny = a_ny
        self.X = X
        self.Y = Y
        self.Z = Z
        self.t = t
        self.tprime = t # number of months
        self.reso = resolution # number of locations that will be
        self.model1 = RDOscilator(self.t, self.reso, self.T, self.h, self.aRD, self.bRD, self.c, self.F_0_RD)
        self.Th = self.model1._Solver()
        # import and call rd-escilator to generate T series

        '''
        d/dx = -Y^2 -Z^2 -aX + a(F_0 + F_1*cos(w_a*t) + a_nx*T_old
        d/dy = XY -bXZ - Y + G_0 + G_1cos(w_a*t) + a_ny*T
        d/dz = bXY + XZ - Z 
        '''
        #self.vector = [-Y^2 -Z^2 -aX + a(F_0 + F_1*cos(t) + a_nx*T_old, XY -bXZ - Y + G_0 + G_1cos(t) + a_ny*T, bXY + XZ - Z]
        return

    def CoupledL84 (self, t, xyz, a, b, F_0, F_1, G_0, G_1, a_nx, a_ny):
        X, Y, Z = xyz

        vector = [-Y**2 -Z**2 -a*X + a*(F_0 + F_1*math.cos(t) + a_nx*self.T_old(t*(self.reso/self.t))), X*Y -b*X*Z - Y + G_0 + G_1*math.cos(t) + a_ny*self.T_new((self.reso/self.t)), b*X*Y + X*Z - Z]
        return vector


    def T_new(self, t):
        #print(t)
        return self.Th[0,int(t+2)]

    def T_old(self, t):

        return  self.Th[0,int(t)]


    def _Solver(self):
        tspace = np.linspace(0,self.t-3,self.reso)
        tspan = (0, self.tprime-3)
        y = (self.X, self.Y, self.Z)
        oscilator = scipy.integrate.solve_ivp(
            self.CoupledL84,
            tspan,
            y,
            args=(self.a84, self.b84, self.F_0_84, self.F_1_84, self.G_0_84, self.G_1_84, self.a_nx, self.a_ny),
            method='RK45',
            t_eval=(tspace)
        )
        return (oscilator.y, oscilator.t)

    def PDF_Solver(self):
        kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(Coupled.vector)
        return (kde)

    def Time_Series(self, vector, time):
        x,y,z = vector
        t = time
        plt.plot(t, x, label = 'Jet Stream')
        plt.plot(t, y, label = 'Cos of Eddies')
        plt.plot(t, z, label = 'Sin of Eddies')
        plt.xlabel("Time in Months")
        plt.ylabel("Magnitude of Jet Stream and Eddies")
        plt.legend()
        plt.show()
        return

    def Time_SeriesEddyNorm(selfself, vector, time):
        x, y, z = vector
        t = time
        eddy = y + z
        plt.plot(t, x, label='Jet Stream')
        plt.plot(t, eddy, label='Eddy Energy')
        plt.xlabel("Time in Months")
        plt.ylabel("Magnitude of Jet Stream and Eddies")
        plt.legend()
        plt.show()

    def XYZ_Space(self, vector):
        x,y,z = vector
        # above_threshold = np.where(T > 1.5, T, np.nan)  # Values above the threshold
        # below_threshold = np.where(T <= 1.5, T, np.nan)
        #plt.plot(T, h, below_threshold, color='black')
        # plt.plot(T, h, above_threshold, color = 'red')
        '''
        plot in a 3D phase space
        with colorations to indicate blocking thresholds
        '''
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(x, y, z, label="l_84 for 1968/69", linewidth=0.5)
        ax.set_xlabel('Strength Jet Stream')
        ax.set_ylabel('Strength Cos Eddie')
        ax.set_zlabel('Strength Sin Eddie')
        plt.show()
        return

    def XYZ_SpaceArrows(self, vector):
        #implement threshold line
        '''implement deterministic arrows by taking grid XYZ labels and calculating a vector from each point'''
        ax = plt.figure().add_subplot(projection='3d')

        # Make the grid
        x, y, z = np.meshgrid(np.arange(-1, 3.25, 1),
                              np.arange(-3.25, 3.25, 1),
                              np.arange(-3.25, 3.25, 1))

        u = -y ** 2 - z ** 2 - self.a84 * x
        v = x * y - self.b84 * x * z - y
        w = self.b84 * x * y + x * z - z

        ax.quiver(x, y, z, u, v, w, length=0.35, normalize=True)
        X, Y, Z = vector

        ax.plot(X, Y, Z, label="l_84 for 1968/69", linewidth=0.5, color='r')
        ax.set_xlabel('Strength Jet Stream')
        ax.set_ylabel('Strength of sin eddy')
        ax.set_zlabel('Strength of cos eddy')

        # plot plane
        #x = 1.5
        #ax.plot_surface(x, y, z, color = 'm', alpha=0.5)
        plt.show()
        '''
        u = -y**2 -z**2 - self.a*x
        v = x*y -self.b*x*z - y
        w = self.b*x*y + x*z - z
        ax.quiver(x, y, z, u, v, w, length=0.1, normalize=True)
        plt.show()
        '''
        return

    def XEddyEnergy_SpaceArrows(self, vector):
        x = np.linspace(-1, 3.25, 10)
        y = np.linspace(-3.5, 3.25, 10)
        z = np.linspace(-3.5, 3.25, 10)
        #u = -y ** 2 - z ** 2 - self.a84 * x
        #v = (x * y - self.b84 * x * z - y) +(self.b84 * x * y + x * z - z)
        eddy = []
        x_coordinate = []
        for j in range(len(x)):
            counter = 0
            total_eddy = 0
            total = 0
            for i in range(len(z)):
                total_eddy += math.sqrt(((x[j] * y[i] - self.b84 * x[j] * z[i] - y[i])**2 + (self.b84 * x[j] * y[i] + x[j] * z[i] - z[i]))**2)
                total += (-y[i] ** 2 - z[i] ** 2 - self.b84 * x[j])
                counter += 1
            eddy.append(total_eddy/counter)
            x_coordinate.append(total/counter)
        x_coordinate = np.array(x_coordinate)
        x_coordinate = ((x_coordinate - x_coordinate.min()) / (x_coordinate.max() - x_coordinate.min())) * 0.25
        eddy = np.array(eddy)
        eddy = ((eddy - eddy.min())/(eddy.max() - eddy.min())) * 0.25
        X, Y, Z = vector
        eddy = Y + Z
        for j in range(len(x)):
            for i in range(len(y)):
                plt.arrow(x[j], y[i], x_coordinate[j], eddy[i])
        plt.plot(X, eddy, label="l_84 for 1968/69", linewidth=0.5, color='r')
        plt.xlabel('Strength Jet Stream')
        plt.ylabel('Strength of Eddy Energy')
        plt.show()

    def JPDF (self, vector):
        X, Y, Z = vector
        eddy = np.sqrt(np.square(Y) + np.square(Z))
        kde = KernelDensity(kernel='gaussian', bandwidth=0.5)
        '''
        data = np.vstack((X, eddy))
        columns = ['Jet Strength', 'Eddy Strength']
        df = pd.DataFrame(data, columns)
        '''
        sns.kdeplot(x=X, y=eddy, cmap='binary', fill=False, bw_adjust=0.8, n_levels=10, alpha=1.0, label='Joint pdf of Jet and Eddy Strenght',  thresh=0.5)
        plt.xlabel('Jet Strength')  # Label for x-axis
        plt.ylabel('Eddy Strength')  # Label for y-axis
        plt.title('Bivariate pdf of Jet & Eddy Strength')  # Title for the plot

        plt.show()
    # should be able to find create a runout span variable
    def NLLE (self, vector, runout):
        '''
        need some way figure out what is a year in the time series, what is a month, and how many
        values in the time series fall into each of those catgories
        '''
        months = self.tprime # number of months in the span of the vector
        resolution = self.reso # total number of data points
        m = math.floor(resolution/months) # number of points per month
        years = math.floor(months/12)
        og_vector = vector
        #pertub initial conditions
        self.X = self.X + 0.0001
        self.Y = self.Y + 0.0001
        self.Z = self.Z + 0.0001
        perturbed_vector, time = biggin._Solver()

        # split vector into chunks by month in order to
        # not og_years is poorly named it is a list of month arrays
        og_years = np.array_split(vector, months, axis = 1)

        '''
        clip all arrays in og_years to same length
        '''

        # 1. finding the minimum length of all arrays in og_years
        min_length = 10000
        for i in range(len(og_years)):
            length = (og_years[i].shape[1])

            if (i != 1) & length < min_length:
                min_length = length
        # clipping the arrays to the appropriate length
        for i in range(len(og_years)):
            array = og_years[i]
            og_years[i] = array[:,:min_length]


        # rinse and repeat for the perturbed vector
        perturbed_years = np.array_split(perturbed_vector, months, axis = 1)
        min_length = 10000
        for i in range(len(perturbed_years)):
            length = (perturbed_years[i].shape[1])

            if (i != 1) & length < min_length:
                min_length = length
        for i in range(len(perturbed_years)):
            array = perturbed_years[i]
            perturbed_years[i] = array[:,:min_length]

        '''
        this needs to be debugged 'sliding window' will likely have many syntactivcal and shape error
        be ready for annoying linear algebra (stuck on line 253) 1/1/2025
        12/30/24'''

        '''
        connect relevant chunks according to runout (sliding window method)
        '''

        og = np.stack(og_years)

        #print(og.shape)
        #print('split by month equal length array')
        perturbed = np.stack(perturbed_years)
        i = 1
        final_og = []
        final_perturbed = []
        for a in range((len(og[:,0]))):
            if(a+runout+1 < len(og[:,0])):
                slice_og = og[a:a+runout,:, :]
                slice_perturbed = perturbed[a:a + runout, :]
            else:
                break

            '''
            else:
                slice_og = og[a:a+runout - i, :, 0]
                slice_purturbed = perturbed[a:a + runout - i, :]
                i+=1     
            '''

            list_og = []
            list_perturbed = []
            '''
            kinda lost about the functionality of this block
            '''
            #print(slice_og.shape)
            start_og = np.empty((3,16))
            start_perturbed = np.empty((3,16))
            for i in range(slice_og.shape[0]):
                single_og = slice_og[i,:,:]
                single_perturbed = slice_perturbed[i, :, :]
                list_og.append(single_og)
                list_perturbed.append(single_perturbed)
            start_og = np.concatenate(list_og, axis = 1)
            #print(start_og.shape)
            #print('start_og shape')
            start_perturbed = np.concatenate(list_perturbed, axis = 1)
            final_og.append(start_og)
            final_perturbed.append(start_perturbed)
        og = np.stack(final_og, axis = 0)
        perturbed = np.stack(final_perturbed, axis = 0)
        og = og.squeeze()
        perturbed = perturbed.squeeze()
        lles = np.empty((years, 12))
        final_lles = np.empty((12,1))
        #print(og.shape)
        #print('og')
        for i in range(years):
            lles[0,i] = i
            for j in range(12):
                print(j)
                local_og = og[i*12 + j, :]
                local_perturbed = perturbed[i*12+j, :]
                #print(og.shape)
                #print('pre-calculation shape')
                local_og = local_og.squeeze()
                local_perturbed = local_perturbed.squeeze()
                # subtract
                difference = (local_og - local_perturbed)

                print(difference.shape)
                print('difference shape')
                # norm
                norm = np.empty(m*runout)

                '''
                nothing here should change
                this should all work the same
                '''

                for k in range(m*runout):
                    #print(difference.shape)
                    norm[k] = np.linalg.norm(difference[:,k])

                # divide and sum
                summation = np.sum(norm * (math.sqrt(0.0001**2*3)) * (1/m))
                #print('summation')
                #print(summation)
                lles[i, j] = summation
                #lles[j+1, i] = (1/m) * (math.sqrt(((og_years[:,(12*i)+j] - (perturbed_years[:,(12*i)+j]))**2))/(sqrt(0.0001**2*3))).sum
                #og_years[:, (12 * i) + j] - (perturbed_years[:, (12 * i) + j])
                # is it significantly problematic for me to take the norm after I subract
                # this is not what I ended up doing I found a simple and straight foreward way to take the norm
                # I am simply curious
                x = 0
                #final_lles[j] = np.avg(lles[j+1,:])
        print('finished')
        return final_lles

# time t is put in in months and resolution per month would be resolution/t
biggin = Coupled(-0.076,0.25,0.236, 4 , -0.125,1 , 1,0.17,7, 1.5625,0,1.5625,0.00513 , -0.184, 1, 1, 1, 30, 500)

vector, time = biggin._Solver()
#biggin.XYZ_Space(vector)
#biggin.XYZ_SpaceArrows(vector)
#biggin.Time_SeriesEddyNorm(vector, time)
#biggin.XEddyEnergy_SpaceArrows(vector)
#biggin.JPDF(vector)
biggin.NLLE(vector, 3)