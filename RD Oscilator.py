import numpy as np
import math as math
import scipy.integrate
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import KernelDensity
'''
d  (T) = (a_nl b)(T)  + F_0(_epsilon(t))
dt (h)   (c    0)(h)       (_delta(t))

a_nl = a, T<=T_c, 0, T> T_c
'''

'''
inputs are _epsilon & _delta 
 and b, c
 '''

'''
Execution:
1. 
    create a function where a,b, maybe epsilon and delta are inputs;
    though epsilon and delta are state independent white noise
2. 
    Solve using a runga kutta just as in the paper
3.
    replicate results in the paper. Try to develop a better understanding
    of the PDFs via this strategy
'''
'''
a = -0.076 month^-1
b = 0.236 K dam^-1 month^-1
c = -0.125 dam K^-1 month^-1
F_0 = 0.17 K/month
'''

'''
 ( 𝜀 and 𝜁 ) are given by monthly gaussian
white-noise series, and are state-independent
'''
def _epsilon(t):
    return np.random.randn() * 40
'''
def _lambda():
    return randn()
'''
def a_nl(T, a):
    if T > 1.5:
        return 0
    else:
        return a

'''
for the following function the parameters are as follows:

t: time is in months
Th, is the T, h initial conditions
a, b, c: arguments that I pulled from the paper !ask prof about these!
F_0: another argument'''
def RD_Oscilator(t, Th, a, b, c, F_0):
    T,h = Th
    vector = ((a_nl(T, a) * T + b * h) + F_0 * _epsilon(t), (c * T) + (F_0 * _epsilon(t)))
    return vector


def _Solver(a, b, c, T, h, F_0, t):
    t = np.linspace(0, t, 300)
    y = [T, h]
    print(1)
    oscilator = scipy.integrate.solve_ivp(RD_Oscilator,(0, 840), y, args=(a, b, c, F_0), method = 'RK45', t_eval = (t))
    return (oscilator.y, oscilator.t)


'''Find a Joint PDF via the following integration technique
        P((X,Y)∈A)=∬AfXY(x,y)dxdy = 1
        ? joint PDF for 2 unique;y continues variables in 1 variable.  
        A: Potentially integrate twice and treat t as t_x, t_y ??
        A: More Likely integrate one time and plot in terms of T, h simulataneously
        where both axis are temp in Kelvin  
'''
def PDF_Solver(vector):
    #t = np.linspace(0, t, 300)
    #y = [T, h]
    kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(vector)
    return (kde)

    '''
    − a_nl𝜌 − (daNL/dT)(T𝜌) − (a_nlT + bh)(𝜕𝜌/𝜕T) − (cT)(𝜕𝜌/𝜕h) + (F^2_0)(𝜕^2𝜌/𝜕𝜙^2)
'''
def Foker_Planck_Solver (vector, a_nl, T, h, f):
    return
def Time_Series(vector, t):
    T, h = vector
    t = np.linspace(0, t, 840)
    plt.plot(t, T)
    plt.plot(t, h)
    plt.show()
    return

def T_h_Space(vector, t):
    T, h = vector
   # above_threshold = np.where(T > 1.5, T, np.nan)  # Values above the threshold
   # below_threshold = np.where(T <= 1.5, T, np.nan)
   # plt.plot(T, h, below_threshold, color='black')
   # plt.plot(T, h, above_threshold, color = 'red')
    plt.plot(T, h)
    #plt.xlabel = 'T in __'
    #plt.ylabel = 'h in __'
    plt.axvline(x = 1.5, color = 'r')
    plt.show()




# note make the span dictated by t as well to minimize work later

'''
write so that whole time series is generated and then use t as a iterator
'''
def _epsilon_integrated (t):
    #t = np.linspace(0, t, 300)
    Brownian = np.random.randint(-1, 1, 840)*40
    '''
    not brownian motion but sort of an approximation
    '''
    return Brownian[int(t-1)]
# error says this method is not picking up t

'''
    − a_nl𝜌 − (daNL/dT)(T𝜌) − (a_nlT + bh)(𝜕𝜌/𝜕T) − (cT)(𝜕𝜌/𝜕h) + (F^2_0)(𝜕^2𝜌/𝜕𝜙^2)
'''
def Fokker_Planck_Oscillator (a, b, c, T, h, F_0, t):

    return
'''
Time Series Test
finalVector, time  = _Solver(-0.076, 0.236, -0.125, 1, 1, 0.17, 840)
Time_Series(finalVector, time)
'''

'''
PDF Test:

finalVector, time  = _Solver(-0.076, 0.236, -0.125, 1, 1, 0.17, 840)
#Time_Series(finalVector, t)
T_h_Space(finalVector, time)
'''
import matplotlib.pyplot as plt
import numpy as np

ax = plt.figure().add_subplot(projection='3d')

# Make the grid
x, y, z = np.meshgrid(np.arange(-0.8, 1, 0.2),
                      np.arange(-0.8, 1, 0.2),
                      np.arange(-0.8, 1, 0.8))

# Make the direction data for the arrows
u = np.sin(np.pi * x) * np.cos(np.pi * y) * np.cos(np.pi * z)
v = -np.cos(np.pi * x) * np.sin(np.pi * y) * np.cos(np.pi * z)
w = (np.sqrt(2.0 / 3.0) * np.cos(np.pi * x) * np.cos(np.pi * y) *
     np.sin(np.pi * z))

ax.quiver(x, y, z, u, v, w, length=0.1, normalize=True)

plt.show()





