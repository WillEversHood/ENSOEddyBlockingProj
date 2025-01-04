import scipy.integrate
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# pX =  -Y^2 - Z^2 -ax + aF

# pY = XY -bXZ -Y +G

# pZ = bXY + XZ -Z

# F = F_0 + F_1cos(w_a * t)
# G = G_0 + G_1cos(w_a * t)
# Define the time scales associated with this model
#t = np.arrange()
def l_84(xyz,a = 0.25, b = 4):
  X,Y,Z = xyz
  Xp = -Y**2 - Z**2 - a*X + a*F
  Yp = X*Y - b*X*Z - Y + G
  Zp = b*X*Y + X*Z - Z
  return Xp, Yp, Zp



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
def l_84_Conditions(t):
#complete this infrastructure before presentation to prof
  #F = F_0 + F_1cos(w * t)
  #G = G_0 + G_1cos(w_a * t)
  return 0
# Solve For F, G

# Time Scales
# reflects the 5-day timesclae used in 2012
t_1 = np.linspace(0,73,73)

#scipy odeint to approximate solution
# 1968/69
swirly = scipy.integrate.odeint(l_84, #[4,1,4.05], t_1)
print(swirly)
xs, xy, xt = swirly.T

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.plot(xs, xy, xt, label = "l_84 for 1968/69")
plt.show()


# plot 3D