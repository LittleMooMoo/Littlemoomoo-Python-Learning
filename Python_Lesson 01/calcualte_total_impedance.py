R = 2700
w = 3354.82
C = 230e-9
L = 603.6e-3
V_max = 15


r = (R ** 2) + (w**2) * (L**2)
a = R * (w**2) * (L**2)
b = (R**2) * w * L
Z_c = 1/(w * C)

real_Z = a/r

imag_Z = b/r - Z_c

total_Z = ((real_Z ** 2) + (imag_Z ** 2)) ** 0.5

I_max = V_max / total_Z

V_c = I_max * Z_c

Q_max = V_c * C

print("The total impedance is: ", total_Z)
print("The maximum charge on the capacitor is: ", Q_max)
