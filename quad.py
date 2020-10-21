import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg


def lqr(A,B,Q,R):
    S = linalg.solve_continuous_are(A,B,Q,R)
    K = linalg.inv(R) @ B.transpose() @ S
    return K


x = 0
y = 0
theta = 1.5
x_dot = 0
y_dot = 0
theta_dot = 0

g = 10
m = 1
I_moment = 1
r = 1

u1 = g / 2
u2 = g / 2

X = np.array([x, y, theta, x_dot, y_dot, theta_dot]).reshape((6,1))

# jacobian for 2-D quadcopter with respect to state
A = np.array([[0, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 1, 0],
              [0, 0, 0, 0, 0, 1], 
              [0, 0, -1*g, 0, 0, 0],
              [0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0]])

# jacobian for 2-D quadcopter with respect to control inputs
B = np.array([[0,0],[0,0],[0,0], [0,0], [1/m, 1/m],[r/I_moment,-1*r/I_moment]])

Q = np.eye(6) * 1000
R = np.eye(2)

K = lqr(A,B,Q,R)

dt = 0.001

current_time = 0
time_vals = []
x_vals = []
y_vals = []
theta_vals = []

stable_state = np.array([0,1,0,0,0,0]).reshape((6,1))

# runge kutta
for i in range(10000):
    
    time_vals.append(current_time)
    x_vals.append(X[0,0])
    y_vals.append(X[1,0])
    theta_vals.append(X[2,0])
    
    u = np.array([m*g/2, m*g/2]).reshape((2,1)) - K @ (X - stable_state)

    k1 = dt * (A @ X + B @ u)
    k2 = dt * (A @ (X + k1/2) + B @ u)
    k3 = dt * (A @ (X + k2/2) + B @ u)
    k4 = dt * (A @ (X + k3) + B @u)

    X = X + (1/6)*(k1 + 2*k2 + 2*k3 + k4)
    current_time = current_time + dt

plt.subplot(411)
plt.plot(time_vals, x_vals)
plt.subplot(412)
plt.plot(time_vals, y_vals)
plt.subplot(413)
plt.plot(time_vals, theta_vals)
plt.subplot(414)
plt.plot(x_vals, y_vals)
plt.show()
