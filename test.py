import mathutils
import numpy as np
import math

omega_x = 0     # theta
omega_y = 45
omega_z = 0

r_org = mathutils.Euler((0, math.radians(45), 0))
print(r_org)
R = r_org.to_matrix()
print(R)
R.resize_4x4()
print(R)

theta = np.sqrt(omega_x ** 2 + omega_y ** 2 + omega_z ** 2)
omega_cross = np.array([0.0, -omega_z, omega_y, omega_z, 0.0, -omega_x, -omega_y, omega_x, 0.0]).reshape(3, 3)

A = np.sin(theta) / theta
B = (1.0 - np.cos(theta)) / (theta ** 2)

R = np.eye(3, 3) + A * omega_cross + B * np.matmul(omega_cross, omega_cross)
print(R)