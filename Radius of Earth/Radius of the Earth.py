#Radius of the Earth
#Author: Ayush Pandhi (1003227457)
#Date: 11/12/2018

#Importing required modules
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#Defining the linear model function
def f(x, a, b):
    return a*x + b

#Loading the data for all 3 runs
floor = np.loadtxt('Radius of Earth Set 1.txt', skiprows=2, usecols=(0,))
mgals1 = (1000)*(0.10023)*(np.loadtxt('Radius of Earth Set 1.txt', skiprows=2, usecols=(1,)))
mgals2 = (1000)*(0.10023)*(np.loadtxt('Radius of Earth Set 2.txt', skiprows=2, usecols=(1,)))
mgals3 = (1000)*(0.10023)*(np.loadtxt('Radius of Earth Set 3.txt', skiprows=2, usecols=(1,)))
    
#Getting a new data set of the averages and setting error to standard deviation
mgals_mean = np.empty(11,)
mgals_std = np.empty(11,)
for i in range(len(mgals_mean)):
    a = np.array([mgals1[i], mgals2[i], mgals3[i]])
    mgals_mean[i] = np.mean(a)
    mgals_std[i] = np.std(a)
    
g = 9.81 - mgals_mean/100000
delg = np.empty(11,)
for i in range(len(delg)):
    delg[i] = g[i] - g[0]
g_error = 1000*mgals_std*0.10023/100000

#Linear regression
p_opt_1, p_cov_1 = curve_fit(f, floor, delg, (0, 0), g_error, True) 
lin_output = f(floor, p_opt_1[0], p_opt_1[1])
print('The two estimated parameters: ', p_opt_1[0],  p_opt_1[1])

#Plots of linear regression
plt.figure(figsize=(10,6))
plt.scatter(floor, delg, label = 'linear raw data', marker='.', color='k')
plt.plot(floor, lin_output, 'r-', label = 'linear fit')
plt.title('Change in g vs. Floor Number')
plt.xlabel('Floor Number')
plt.ylabel('Delta g (m/s^2)')
plt.errorbar(floor, delg, xerr=0, yerr=g_error, linestyle='none', ecolor='g', label='Error', capsize=2)
plt.legend()
plt.show()

#Calculating chi squared
chi_sq = (1/9)*(np.sum(((delg - lin_output) / g_error)**2))
print('Chi squared for linear regression: ', chi_sq)

#Estimating the radius of Earth using linear portion of data
r = 2*(np.mean(g))/p_opt_1[0]
r = 3.95*r/1000
print('Estimated Radius in km: ', r)
print('Error in Radius result: ', r*((p_cov_1[0,0]**2/p_opt_1[0])**2 + (np.max(g_error)/np.mean(g))**2)**0.5)

#Using measurements from the basement and 14th floor
mgals_anom = (1000)*(0.10023)*(np.loadtxt('Radius of Earth Anomalies Set.txt', skiprows=2, usecols=(1,)))
mgals_anom14, mgals_anomB = np.hsplit(mgals_anom, 2)
mgals_anom14_mean = np.mean(mgals_anom14)
mgals_anom14_std = np.std(mgals_anom14)
mgals_anomB_mean = np.mean(mgals_anomB)
mgals_anomB_std = np.std(mgals_anomB)

g_14 = 9.81 - mgals_anom14_mean/100000
delg_14 = g_14 - g[0]
g_14error = 1000*mgals_anom14_std*0.10023/100000

g_B = 9.81 - mgals_anomB_mean/100000
delg_B = g_B - g[0]
g_Berror = 1000*mgals_anomB_std*0.10023/100000

floor_anom = np.hstack((-1, floor, 14))
g_anom = np.hstack((g_B, g, g_14))
delg_anom = np.hstack((delg_B, delg, delg_14))
g_error_anom = np.hstack((g_Berror, g_error, g_14error))

#Linear regression
p_opt_1, p_cov_1 = curve_fit(f, floor_anom, delg_anom, (0, 0), g_error_anom, True) 
lin_output = f(floor_anom, p_opt_1[0], p_opt_1[1])
print('The two estimated parameters: ', p_opt_1[0],  p_opt_1[1])

#Plots of linear regression
plt.figure(figsize=(10,6))
plt.scatter(floor_anom, delg_anom, label = 'linear raw data', marker='.', color='k')
plt.plot(floor_anom, lin_output, 'r-', label = 'linear fit')
plt.title('Change in g vs. Floor Number')
plt.xlabel('Floor Number')
plt.ylabel('Delta g (m/s^2)')
plt.errorbar(floor_anom, delg_anom, xerr=0, yerr=g_error_anom, linestyle='none', ecolor='g', label='Error', capsize=2)
plt.legend()
plt.show()

#Calculating chi squared
chi_sq = (1/11)*(np.sum(((delg_anom - lin_output) / g_error_anom)**2))
print('Chi squared for linear regression: ', chi_sq)

#Estimating the radius of Earth using linear portion of data
r = 2*(np.mean(g_anom))/p_opt_1[0]
r = 3.95*r/1000
print('Estimated Radius in km: ', r)
print('Error in Radius result: ', r*((p_cov_1[0,0]**2/p_opt_1[0])**2 + (np.max(g_error)/np.mean(g))**2)**0.5)
