import numpy as np
import math as math
import scipy.integrate
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import KernelDensity

class RDOscilator():
    def __init__(self, t, resolution, T, h, a, b, c, F_0):
        self.tprime = t
        self.t = t
        self.reso = resolution
        self.T = T
        self.h = h
        self.a = a
        self.b = b
        self.c = c
        self.F_0 = F_0
        return

    def RDO_scilator (self, t, Th, a, b, c, F_0):
        T, h = Th
        vector = ((self.a_nl(T, self.a) * T + b * h) + F_0 * self._epsilon(t), (c * T) + (F_0 * self._epsilon(t)))
        return vector

    def _epsilon(self, t):
        return np.random.randn() * 40

    def a_nl(self, T, a):
        if T > 1.5:
            return 0
        else:
            return a

    def _Solver(self):
        print('!!!!!!!!!!!!!!!!!!')
        #tend = self.t
        t = np.linspace(0,self.t,self.reso)
        t_span = (0, self.tprime)
        y = [self.T, self.h]
        oscilator = scipy.integrate.solve_ivp(
            self.RDO_scilator,
            t_span,
            y,
            args=(self.a, self.b, self.c, self.F_0),
            method='RK45',
            t_eval=(t)
                                              )
        return (oscilator.y)

    def PDF_Solver(self):
        # t = np.linspace(0, t, 300)
        # y = [T, h]
        kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(vector)
        return (kde)

    def Time_Series(self, vector, t):
        T, h = vector
        t = np.linspace(0, t, t)
        plt.plot(t, T)
        plt.plot(t, h)
        plt.show()
        return

    def T_h_Space(self, vector, t):
        T, h = vector
        # above_threshold = np.where(T > 1.5, T, np.nan)  # Values above the threshold
        # below_threshold = np.where(T <= 1.5, T, np.nan)
        # plt.plot(T, h, below_threshold, color='black')
        # plt.plot(T, h, above_threshold, color = 'red')
        plt.plot(T, h)
        # plt.xlabel = 'T in __'
        # plt.ylabel = 'h in __'
        plt.axvline(x=1.5, color='r')
        plt.show()


'''
test
'''

'''
finalVector = RD._Solver()
print('purple')
print(finalVector)
'''
