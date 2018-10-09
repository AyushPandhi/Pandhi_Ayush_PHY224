#PyLab 2: Introduction to Fitting Methods
#Author: Ayush Pandhi (1003227457)
#Date: 10/09/2018

#Importing required modules
import numpy as np
import math as m
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#Defining the model function
def f(x, a, b):
    return a*x + b

#Loading the resistor data
voltage1 = np.loadtxt('resistor data.txt', skiprows=1, usecols=(0,))
current1 = (1/1000)*(np.loadtxt('resistor data.txt', skiprows=1, usecols=(1,)))

#Loading the potentiometer
voltage2 = np.loadtxt('potentiometer data.txt', skiprows=1, usecols=(0,))
current2 = (1/1000)*(np.loadtxt('potentiometer data.txt', skiprows=1, usecols=(1,)))

#Finding max of precision and accuracy error for resistor data
v_error1 = np.empty(len(voltage1))
for i in range(len(voltage1)):
    v_error1[i] = max(voltage1[i]*0.0025, 0.01)
    
i_error1 = np.empty(len(current1))
for i in range(len(current1)):
    i_error1[i] = max(current1[i]*0.0075, 0.1/1000)
    
#Finding max of precision and accuracy error for potentiometer data
v_error2 = np.empty(len(voltage2))
for i in range(len(voltage2)):
    v_error2[i] = max(voltage2[i]*0.0025, 0.01) 
    
i_error2 = np.empty(len(current2))
for i in range(len(current2)):
    i_error2[i] = max(current2[i]*0.0075, 0.1/1000)

#Calling curve_fit() for these data sets 
p_opt1 , p_cov1 = curve_fit(f, voltage1, current1, (1/100, 0), i_error1, True)
p_opt2 , p_cov2 = curve_fit(f, voltage2, current2, (1/57, 0), i_error2, True)

#Outputs based on the model function
output1 = f(voltage1, p_opt1[0], p_opt1[1])
output2 = f(voltage2, p_opt2[0], p_opt2[1])

#Plotting Voltage vs Current
plt.figure(figsize=(10,8))
plt.plot(voltage1, output1, 'r-', label='Modeled Resistor')
plt.plot(voltage2, output2, 'b-', label='Modeled Potentiometer')
plt.plot(voltage1, current1, 'k.', label='Experimental Resistor')
plt.plot(voltage2, current2, 'k+', label='Experimental Potentiometer')
plt.xlabel('voltage (V)')
plt.ylabel('Current (A)')
plt.title('Voltage vs Current')
plt.legend(loc='upper left')
plt.show()

#Defining chi squared data
chisqr1 = sum(((current1-output1)/i_error1)**2)
chisqr1 = chisqr1/8
chisqr2 = sum(((current2-output2)/i_error2)**2)
chisqr2 = chisqr2/8

#Printing resulting parameters
print('Experimental Resistor Resistance: ', 1/p_opt1[0], 'Ohms')
print('Experimental Potentiometer Resistance: ', 1/p_opt2[0], 'Ohms')
print('Experimental Resistor Chi Squared Value: ', chisqr1)
print('Experimental Potentiometer Chi Squared Value ', chisqr2)

