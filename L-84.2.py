import numpy as np
import math as math
import scipy.integrate
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#F0, F_1 (and similarly G0,G1) are the average values and the magnitudes of the annual cycle for EPG and OLC.
# You set the winter (t=0) and summer (t=T/2 halfway through the year)
# values for F and G to the nominal Lorenz values (see in the paper),
# and so you have 4 equations with 4 unknowns and this is how you find the F0,F1,G0,G1

# Nominal Values
#Conditions       Fw (EPG)      Gw (OLC)      Fs (EPG)     Gs (OLC)        F0    F1       G0     G1
#Typical (mean)    8 (-0.81)    1 (14.04)     6 (-0.35)    21 (-3.49)      7     1.5625   0      1.5625
#1968/69           8.3 (-0.84)  1.16 (16.32)  6.3 (-0.37)  20.98 (-3.43)   7.3   1.5625   0.09   1.675
#1988/89           7.7 (-0.78)  0.86 (12.14)  6 (-0.35)    21.04 (-3.65)   6.85  1.3281  -0.088  1.488
#  The values proposed by Lorenz (1984) of a = 0.25 and b = 4.0

# F and G calculations for l_84 forced by EPG & OLC for ENSO year 1968/69
def F(w):
    # for 1968/69
    return 7.3 + 1.5625*math.cos(w)

def G(w):
    # for 1968/69
    return 0.09 + 1.675*math.cos(w)

# define the vector decribing d(X,Y,Z)
'''
X - relative strength of the Jet Stream
Y, Z - sin and cos elements of the '''
def l_84(xyz, w, a, b):
  X,Y,Z = xyz
  vector = [-Y**2 - Z**2 - a*X + a*F(w), X*Y - b*X*Z - Y + G(w), b*X*Y + X*Z - Z]
  return vector

# time
w_values = np.linspace(0,50*math.pi, 7300)
#initial Conditions
y0 = [0,0,0]
# constants
a = 0.25
b = 4
# solve
swirly = scipy.integrate.odeint(l_84, y0, w_values, args=(a,b))
#unpack swirly
x, y, z = swirly.T
# plot
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.plot(x, y, z, label = "l_84 for 1968/69", linewidth=0.2)
plt.show()

