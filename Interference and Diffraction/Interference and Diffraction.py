#Interference and Diffraction
#Author: Ayush Pandhi (1003227457)
#Date: November 29, 2018

#Importing required modules
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#Defining functions to compute theta and slit width
def get_theta(l, b):
    theta = np.arctan(l/b)
    return theta

def get_slitwidth(m, theta, wavelength):
    slitwidth = (m*wavelength)/np.sin(theta)
    return slitwidth

def get_slitseparation(m, theta, wavelength):
    slitseparation = (m*wavelength)/np.sin(theta)
    return slitseparation

def get_intensityratio(I_0, I_i):
    intensityratio = I_i/I_0
    return intensityratio

#Loading Data Set for single slit 0.04mm
position = np.loadtxt('Single Slit Exercise 1 (0.04mm).txt', skiprows=2, usecols=(0,)) + 0.01228
intensity = np.loadtxt('Single Slit Exercise 1 (0.04mm).txt', skiprows=2, usecols=(1,))

#Loading Data Set for single slit 0.08mm
position2 = np.loadtxt('Single Slit Exercise 1 (0.08mm).txt', skiprows=2, usecols=(0,)) + 0.00950
intensity2 = np.loadtxt('Single Slit Exercise 1 (0.08mm).txt', skiprows=2, usecols=(1,))

#Loading Data Set for single slit 0.16mm
position3 = np.loadtxt('Single Slit Exercise 1 (0.16mm).txt', skiprows=2, usecols=(0,)) - 0.01078
intensity3 = np.loadtxt('Single Slit Exercise 1 (0.16mm).txt', skiprows=2, usecols=(1,))

#Loading Data Set for double slit 0.25, 0.04mm
position4 = np.loadtxt('Double Slit Exercise 2 (0.25, 0.04mm).txt', skiprows=2, usecols=(0,)) - 0.01350
intensity4 = np.loadtxt('Double Slit Exercise 2 (0.25, 0.04mm).txt', skiprows=2, usecols=(1,))

#Loading Data Set for double slit 0.25, 0.08mm
position5 = np.loadtxt('Double Slit Exercise 2 (0.25, 0.08mm).txt', skiprows=2, usecols=(0,)) + 0.00989
intensity5 = np.loadtxt('Double Slit Exercise 2 (0.25, 0.08mm).txt', skiprows=2, usecols=(1,))

#Loading Data Set for double slit 0.50, 0.04mm
position6 = np.loadtxt('Double Slit Exercise 2 (0.50, 0.04mm).txt', skiprows=2, usecols=(0,)) + 0.01283
intensity6 = np.loadtxt('Double Slit Exercise 2 (0.50, 0.04mm).txt', skiprows=2, usecols=(1,))

#Loading Data Set for double slit 0.50, 0.08mm
position7 = np.loadtxt('Double Slit Exercise 2 (0.50, 0.08mm).txt', skiprows=2, usecols=(0,)) - 0.00994
intensity7 = np.loadtxt('Double Slit Exercise 2 (0.50, 0.08mm).txt', skiprows=2, usecols=(1,))

#Estimated errors in intensity and position
I_error = np.empty(len(intensity))
for i in range(len(intensity)):
    I_error[i] = 0.00001

x_error = np.empty(len(position))
for i in range(len(position)):
    x_error[i] = 0.00001
    
#Estimated errors in intensity and position
I_error2 = np.empty(len(intensity2))
for i in range(len(intensity2)):
    I_error2[i] = 0.00001
    
x_error2 = np.empty(len(position2))
for i in range(len(position2)):
    x_error2[i] = 0.00001
    
#Estimated errors in intensity and position
I_error3 = np.empty(len(intensity3))
for i in range(len(intensity3)):
    I_error3[i] = 0.00001
    
x_error3 = np.empty(len(position3))
for i in range(len(position3)):
    x_error3[i] = 0.00001
    
#Estimated errors in intensity and position
I_error4 = np.empty(len(intensity4))
for i in range(len(intensity4)):
    I_error4[i] = 0.00001
    
x_error4 = np.empty(len(position4))
for i in range(len(position4)):
    x_error4[i] = 0.00001
    
#Estimated errors in intensity and position
I_error5 = np.empty(len(intensity5))
for i in range(len(intensity5)):
    I_error5[i] = 0.00001

x_error5 = np.empty(len(position5))
for i in range(len(position5)):
    x_error5[i] = 0.00001
    
#Estimated errors in intensity and position
I_error6 = np.empty(len(intensity6))
for i in range(len(intensity6)):
    I_error6[i] = 0.00001
    
x_error6 = np.empty(len(position6))
for i in range(len(position6)):
    x_error6[i] = 0.00001
    
#Estimated errors in intensity and position
I_error7 = np.empty(len(intensity7))
for i in range(len(intensity7)):
    I_error7[i] = 0.00001
    
x_error7 = np.empty(len(position7))
for i in range(len(position7)):
    x_error7[i] = 0.00001
    
#Single Slit 0.04mm plot
plt.figure(figsize=(10,8))
plt.plot(position, intensity, '-')
plt.title('Single Slit 0.04mm Diffraction Pattern')
plt.xlabel('Position (m)')
plt.ylabel('Intensity (V)')
plt.show()

#Single Slit 0.08mm plot
plt.figure(figsize=(10,8))
plt.plot(position2, intensity2, '-')
plt.title('Single Slit 0.08mm Diffraction Pattern')
plt.xlabel('Position (m)')
plt.ylabel('Intensity (V)')
plt.show()

#Single Slit 0.16mm plot
plt.figure(figsize=(10,8))
plt.plot(position3, intensity3, '-')
plt.title('Single Slit 0.16mm Diffraction Pattern')
plt.xlabel('Position (m)')
plt.ylabel('Intensity (V)')
plt.show()

#Double Slit 0.25, 0.04mm plot
plt.figure(figsize=(10,8))
plt.plot(position4, intensity4, '-')
plt.title('Double Slit 0.25, 0.04mm Diffraction and Interference Pattern')
plt.xlabel('Position (m)')
plt.ylabel('Intensity (V)')
plt.show()

#Double Slit 0.25, 0.08mm plot
plt.figure(figsize=(10,8))
plt.plot(position5, intensity5, '-')
plt.title('Double Slit 0.25, 0.08mm Diffraction and Interference Pattern')
plt.xlabel('Position (m)')
plt.ylabel('Intensity (V)')
plt.show()

#Double Slit 0.50, 0.04mm plot
plt.figure(figsize=(10,8))
plt.plot(position6, intensity6, '-')
plt.title('Double Slit 0.50, 0.04mm Diffraction and Interference Pattern')
plt.xlabel('Position (m)')
plt.ylabel('Intensity (V)')
plt.show()

#Double Slit 0.50, 0.08mm plot
plt.figure(figsize=(10,8))
plt.plot(position7, intensity7, '-')
plt.title('Double Slit 0.50, 0.08mm Diffraction and Interference Pattern')
plt.xlabel('Position (m)')
plt.ylabel('Intensity (V)')
plt.show()

#Single Slit 0.04mm plot with errors
plt.figure(figsize=(10,8))
plt.plot(position, intensity, '-')
plt.title('Single Slit 0.04mm Diffraction Pattern')
plt.xlabel('Position (m)')
plt.ylabel('Intensity (V)')
plt.errorbar(position, intensity, xerr=0, yerr=I_error, linestyle='none', ecolor='g', label='Error', capsize=2)
plt.show()

#Single Slit 0.08mm plot with errors
plt.figure(figsize=(10,8))
plt.plot(position2, intensity2, '-')
plt.title('Single Slit 0.08mm Diffraction Pattern')
plt.xlabel('Position (m)')
plt.ylabel('Intensity (V)')
plt.errorbar(position2, intensity2, xerr=0, yerr=I_error2, linestyle='none', ecolor='g', label='Error', capsize=2)
plt.show()

#Single Slit 0.16mm plot with errors
plt.figure(figsize=(10,8))
plt.plot(position3, intensity3, '-')
plt.title('Single Slit 0.16mm Diffraction Pattern')
plt.xlabel('Position (m)')
plt.ylabel('Intensity (V)')
plt.errorbar(position3, intensity3, xerr=0, yerr=I_error3, linestyle='none', ecolor='g', label='Error', capsize=2)
plt.show()

#Double Slit 0.25, 0.04mm plot with errors
plt.figure(figsize=(10,8))
plt.plot(position4, intensity4, '-')
plt.title('Double Slit 0.25, 0.04mm Diffraction and Interference Pattern')
plt.xlabel('Position (m)')
plt.ylabel('Intensity (V)')
plt.errorbar(position4, intensity4, xerr=0, yerr=I_error4, linestyle='none', ecolor='g', label='Error', capsize=2)
plt.show()

#Double Slit 0.25, 0.08mm plot with errors
plt.figure(figsize=(10,8))
plt.plot(position5, intensity5, '-')
plt.title('Double Slit 0.25, 0.08mm Diffraction and Interference Pattern')
plt.xlabel('Position (m)')
plt.ylabel('Intensity (V)')
plt.errorbar(position5, intensity5, xerr=0, yerr=I_error5, linestyle='none', ecolor='g', label='Error', capsize=2)
plt.show()

#Double Slit 0.50, 0.04mm plot with errors
plt.figure(figsize=(10,8))
plt.plot(position6, intensity6, '-')
plt.title('Double Slit 0.50, 0.04mm Diffraction and Interference Pattern')
plt.xlabel('Position (m)')
plt.ylabel('Intensity (V)')
plt.errorbar(position6, intensity6, xerr=0, yerr=I_error6, linestyle='none', ecolor='g', label='Error', capsize=2)
plt.show()

#Double Slit 0.50, 0.08mm plot with errors
plt.figure(figsize=(10,8))
plt.plot(position7, intensity7, '-')
plt.title('Double Slit 0.50, 0.08mm Diffraction and Interference Pattern')
plt.xlabel('Position (m)')
plt.ylabel('Intensity (V)')
plt.errorbar(position7, intensity7, xerr=0, yerr=I_error7, linestyle='none', ecolor='g', label='Error', capsize=2)
plt.show()

#Defining parametes for single slit 0.04mm
l1 = np.array([0.004, 0.00856, 0.00439, 0.00772])
b = 0.915
wavelength = 6.5*10**(-7)
m1 = np.array([1, 2, 1, 2])

#Computing theta for single slit 0.04mm
theta1 = get_theta(l1, b)
print('Computed Theta (radians): ', theta1)
print('Mean Theta (radians): ', np.mean(theta1))

#Computing experimental slit width for single slit 0.04mm
a_computed = get_slitwidth(m1, theta1, wavelength)
print('Computed Slit Width (m): ', a_computed)
print('Mean Slit Width (m): ', np.mean(a_computed))

#Defining parametes for single slit 0.08mm
l2 = np.array([0.00206, 0.00361, 0.002, 0.00383])
m2 = np.array([1, 2, 1, 2])

#Computing theta for single slit 0.08mm
theta2 = get_theta(l2, b)
print('Computed Theta (radians): ', theta2)
print('Mean Theta (radians): ', np.mean(theta2))

#Computing experimental slit width for single slit 0.08mm
a_computed2 = get_slitwidth(m2, theta2, wavelength)
print('Computed Slit Width (m): ', a_computed2)
print('Mean Slit Width (m): ', np.mean(a_computed2))

#Defining parametes for single slit 0.16mm
l3 = np.array([0.001, 0.00194, 0.00289, 0.00095, 0.00184, 0.00284])
m3 = np.array([1, 2, 3, 1, 2, 3])

#Computing theta for single slit 0.16mm
theta3 = get_theta(l3, b)
print('Computed Theta (radians): ', theta3)
print('Mean Theta (radians): ', np.mean(theta3))

#Computing experimental slit width for single slit 0.16mm
a_computed3 = get_slitwidth(m3, theta3, wavelength)
print('Computed Slit Width (m): ', a_computed3)
print('Mean Slit Width (m): ', np.mean(a_computed3))

#Defining parametes for double slit 0.25, 0.004mm
k4 = np.array([0.00122, 0.00183, 0.00117, 0.00178])
l4 = np.array([0.00428, 0.00872, 0.00433, 0.00817])
m4 = np.array([1, 2, 1, 2])

#Computing theta for double slit 0.25, 0.004mm (diffraction)
theta4d = get_theta(l4, b)
print('Computed Theta (radians): ', theta4d)
print('Mean Theta (radians): ', np.mean(theta4d))

#Computing experimental slit width for double slit 0.25, 0.004mm
a_computed4 = get_slitwidth(m4, theta4d, wavelength)
print('Computed Slit Width (m): ', a_computed4)
print('Mean Slit Width (m): ', np.mean(a_computed4))

#Computing theta for double slit 0.25, 0.004mm (interference)
theta4i = get_theta(k4, b)
print('Computed Theta (radians): ', theta4i)
print('Mean Theta (radians): ', np.mean(theta4i))

#Computing experimental slit separation for double slit 0.25, 0.004mm
d_computed4 = get_slitseparation(m4, theta4i, wavelength)
print('Computed Slit Separation (m): ', d_computed4)
print('Mean Slit Separation (m): ', np.mean(d_computed4))

#Defining parametes for double slit 0.50, 0.004mm
k5 = np.array([0.00061, 0.00089, 0.00061, 0.00089])
l5 = np.array([0.00427, 0.00794, 0.00406, 0.00795])
m5 = np.array([1, 2, 1, 2])

#Computing theta for double slit 0.50, 0.004mm (diffraction)
theta5d = get_theta(l5, b)
print('Computed Theta (radians): ', theta5d)
print('Mean Theta (radians): ', np.mean(theta5d))

#Computing experimental slit width for double slit 0.50, 0.004mm
a_computed5 = get_slitwidth(m5, theta5d, wavelength)
print('Computed Slit Width (m): ', a_computed5)
print('Mean Slit Width (m): ', np.mean(a_computed5))

#Computing theta for double slit 0.50, 0.004mm (interference)
theta5i = get_theta(k5, b)
print('Computed Theta (radians): ', theta5i)
print('Mean Theta (radians): ', np.mean(theta5i))

#Computing experimental slit separation for double slit 0.50, 0.004mm
d_computed5 = get_slitseparation(m5, theta5i, wavelength)
print('Computed Slit Separation (m): ', d_computed5)
print('Mean Slit Separation (m): ', np.mean(d_computed5))

#Defining parametes for double slit 0.50, 0.008mm
k6 = np.array([0.00062, 0.00089, 0.00061, 0.00088])
l6 = np.array([0.00212, 0.00378, 0.00211, 0.00372])
m6 = np.array([1, 2, 1, 2])

#Computing theta for double slit 0.50, 0.008mm (diffraction)
theta6d = get_theta(l6, b)
print('Computed Theta (radians): ', theta6d)
print('Mean Theta (radians): ', np.mean(theta6d))

#Computing experimental slit width for double slit 0.50, 0.008mm
a_computed6 = get_slitwidth(m6, theta6d, wavelength)
print('Computed Slit Width (m): ', a_computed6)
print('Mean Slit Width (m): ', np.mean(a_computed6))

#Computing theta for double slit 0.50, 0.008mm (interference)
theta6i = get_theta(k6, b)
print('Computed Theta (radians): ', theta6i)
print('Mean Theta (radians): ', np.mean(theta6i))

#Computing experimental slit separation for double slit 0.50, 0.008mm
d_computed6 = get_slitseparation(m6, theta6i, wavelength)
print('Computed Slit Separation (m): ', d_computed6)
print('Mean Slit Separation (m): ', np.mean(d_computed6))

#Verifying Equation 9 within error margins
print((np.mean(a_computed)/wavelength)*(np.sin(np.arctan(np.mean(l1)/b))))
print((np.mean(a_computed2)/wavelength)*(np.sin(np.arctan(np.mean(l2)/b))))
print((np.mean(a_computed3)/wavelength)*(np.sin(np.arctan(np.mean(l3)/b))))

#Measured intensities from single slit 0.16mm
I_0 = 3.49964
I_i = np.array([0.23077, 0.07144, 0.03584, 0.21386, 0.09689, 0.02988])
I_ratio = get_intensityratio(I_0, I_i)
print('Intensity Ratios: ', I_ratio)
print('I_1/I_0: ', (I_ratio[0] + I_ratio[3])/2)
print('I_2/I_0: ', (I_ratio[1] + I_ratio[4])/2)
print('I_3/I_0: ', (I_ratio[2] + I_ratio[5])/2)

#Defining a model function for regression
def f(x, a):
    return a*(np.sinc(x)**2)

#Computing theta and phi for single slit 0.04mm
theta_plot1 = get_theta(position, b)
phi1 = (np.pi*(4.0*10**(-5))/wavelength)*(np.sin(theta_plot1))

#Non-linear regression for single slit 0.04mm
p_opt_1, p_cov_1 = curve_fit(f, phi1, intensity, p0=(0.14)) 
output1 = f(phi1, p_opt_1[0])

#Computing theta and phi for single slit 0.08mm
theta_plot2 = get_theta(position2, b)
phi2 = (np.pi*(4.0*10**(-5))/wavelength)*(np.sin(theta_plot2))

#Non-linear regression for single slit 0.08mm
p_opt_2, p_cov_2 = curve_fit(f, phi2, intensity2, p0=(0.8)) 
output2 = f(phi2, p_opt_2[0])

#Computing theta and phi for single slit 0.16mm
theta_plot3 = get_theta(position3, b)
phi3 = (np.pi*(4.0*10**(-5))/wavelength)*(np.sin(theta_plot3))

#Non-linear regression for single slit 0.16mm
p_opt_3, p_cov_3 = curve_fit(f, phi3, intensity3, p0=(3.5)) 
output3 = f(phi3, p_opt_3[0])

#Calculating chi squared
chi_sq_1 = (1/3956)*(np.sum(((intensity - output1) / I_error)**2))
print('Chi squared for plot 1: ', chi_sq_1)

chi_sq_2 = (1/3458)*(np.sum(((intensity2 - output2) / I_error2)**2))
print('Chi squared for plot 2: ', chi_sq_2)

chi_sq_3 = (1/3497)*(np.sum(((intensity3 - output3) / I_error3)**2))
print('Chi squared for plot 3: ', chi_sq_3)

#Over plot of non-linear regression and raw data
#Single Slit 0.04mm plot
plt.figure(figsize=(10,8))
plt.plot(position, intensity, '-', label='Raw Data')
plt.plot(position, output1, '-', label='Non-linear Regression')
plt.title('Single Slit 0.04mm Diffraction Pattern')
plt.xlabel('Position (m)')
plt.ylabel('Intensity (V)')
plt.errorbar(position, intensity, xerr=0, yerr=I_error, linestyle='none', ecolor='g', label='Error', capsize=2)
plt.legend()
plt.show()

#Single Slit 0.08mm plot
plt.figure(figsize=(10,8))
plt.plot(position2, intensity2, '-', label='Raw Data')
plt.plot(position2, output2, '-', label='Non-linear Regression')
plt.title('Single Slit 0.08mm Diffraction Pattern')
plt.xlabel('Position (m)')
plt.ylabel('Intensity (V)')
plt.errorbar(position2, intensity2, xerr=0, yerr=I_error2, linestyle='none', ecolor='g', label='Error', capsize=2)
plt.legend()
plt.show()

#Single Slit 0.16mm plot
plt.figure(figsize=(10,8))
plt.plot(position3, intensity3, '-', label='Raw Data')
plt.plot(position3, output3, '-', label='Non-linear Regression')
plt.title('Single Slit 0.16mm Diffraction Pattern')
plt.xlabel('Position (m)')
plt.ylabel('Intensity (V)')
plt.errorbar(position3, intensity3, xerr=0, yerr=I_error3, linestyle='none', ecolor='g', label='Error', capsize=2)
plt.legend()
plt.show()

#Calculating a/wavelength ratio for the single slit experiments
print('Experimental a/lambda: ', np.mean(a_computed)/wavelength, np.mean(a_computed2)/wavelength, np.mean(a_computed3)/wavelength)
print('Expected a/lambda: ', (4.0*(10**-5))/wavelength, (8.0*(10**-5))/wavelength, (16.0*(10**-5))/wavelength)
