# PHY224 Computational Assignment 1
# Author: Ayush Pandhi (1003227457)
# Due Date: September 24, 2018

# Exercise 1

#Importing required modules
import numpy as np

#Defining variables
h = .1                              #Define h as the step size (x[i-1] - x[i])
x = np.arange(10)*h                 #A range of x values from 0 to 1
y = np.cos(x)                       #Cosine fuction applied on x

#Estimating the rate of change of y using a for loop
delta_y = []                        #Defining an empty list which we will append at each step
for i in range(len(x) - 1):
    delta_y.append(y[i+1] - y[i])   #Append the empty list with the rate of change at each step
    
#Printing the output of delta_y
print('Exercise 1:')
print('Rate of change of y at each step:')
print(delta_y)
print()

#Checking if delta_y has one less element than y
print('Number of elements in y array: ', len(y))
print('Number of elements in delta_y array: ', len(delta_y))
print()
print()

# Exercise 2

#Using array arithmetic to solve for delta_y instead
delta_y2 = y[1:] - y[:-1]         #Subtracting the first N-1 elements of y from the last N-1 elements (same as y[i+1] - y[i])

#Printing the output of delta_y2
print('Exercise 2:')
print('Rate of change of y at each step with array arithmetic:')
print(delta_y2)
print()

#Checking if delta_y2 has one less element than y
print('Number of elements in y array: ', len(y))
print('Number of elements in delta_y2 array: ', len(delta_y2))
print()
print()

# Exercise 3

#Defining given constants and measurements in SI units
g = 9.81                #Acceleration due to gravity in m/s^2
m = 1.42                #Mass in kg
m_error = 0.1           #Mass uncertainty in kg
v = 35.0                #Velocity in m/s
v_error = 0.9           #Velocity uncertainty in m/s
r = 120.0/100           #Radius in m
r_error = 2.0/100       #Radius uncertainity in m

#Computing tension in the rope
tension = m*(g**2 + (v**2/r)**2)**0.5

#Propogating error for tension
vsqr_error = 2*v*v_error
vsqr_div_r_error = (v**2/r)*((vsqr_error/v**2)**2 + (r_error/r)**2)**0.5
vsqr_div_r_sqr_error = 2*(v**2/r)*vsqr_div_r_error
vsqr_div_r_sqr_root_error = 0.5*((v**2/r) + g**2)*vsqr_div_r_sqr_error
tension_error = (m*(g**2 + (v**2/r)**2)**0.5)*((m_error/m)**2 + (((g**2 + (v**2/r)**2)**0.5)/vsqr_div_r_sqr_root_error)**2)**0.5

#Printing output of tension and its uncertainity
print('Exercise 3:')
print('The tension in the string in Newtons is: ', tension)
print('With uncertainity in Newtons: ', tension_error)
print()
print()

# Exercise 4

#Loading sample-correlation.txt data_correlation
data_correlation = np.loadtxt('sample-correlation.txt')

#Getting the covariance matrix of the data_correlation
corrcoef_array = np.corrcoef(np.transpose(data_correlation))

#Printing the covariance matrix
print('Exercise 4')
print('The Covariance Matrix:')
print(corrcoef_array)
print()

#The covariance array has elements: [C_ii, C_ij
#                                    C_ji, C_jj]
#Note that also this is a symmetric matrix, meaning that C_ij = C_ji
#We want to know the correlation between the two paramenter i and j, so we can print either C_ij or C_ji to get this coefficient

#Printing the correlation coefficient between the two parameters
print('The Correlation Coefficient is: ', corrcoef_array[0,1])
print()
print()

# Exercise 5

#Incorrect given for assignment:
#  import numpy as np
#  a, b = 0, np.pi/2
#  N = 50 # Number of intervals
#  dx = (a-b)/49
#  x = np.arange(a, b, N)
#  f = np.cos(x)
#  riemann_sum = np.sum(f * dx)
#  print(riemann_sum)

#Corrected script to get the reimann sum
import numpy as np
a, b = 0, np.pi/2
N = 50                       #Number of intervals
dx = (a-b)/50                #Changed denominator to 50 from 49
x = np.linspace(a, b, N)     #Changed arange to linspace
f = np.cos(x)
riemann_sum = np.sum(f * dx)
print('Exercise 5:')
print(riemann_sum)
print()
print()

# Exercise 6

#Defining the average function
def average(x):
    return sum(x)/len(x)

#Testing the function
print('Exercise 6:')
print('The expected value is: -0.5')
print('The function returns: ', average(np.arange(-5,5)))
print()
print()

# Exercise 7

#Defining the stdev function
def stdev(x):
    n = len(x)
    mu = (sum(x)/n)
    numerator = []
    for xi in x:
        numerator.append((xi - mu)**2)
    sigma = (sum(numerator)/n)**(1/2)
    return mu, sigma

#Testing the function
print('Exercise 7:')
print('The expected mean and standard deviation: ', np.mean(np.arange(10)), ' ', np.std(np.arange(10)))
print('The function returns: ', stdev(np.arange(10)))
print()
print()


# Exercise 8

#Had to remove the commas from the txt file and reorganize it into two columns to get it to work
data_resistor = np.loadtxt('sample-resistor.txt')
data_1, data_2 = np.hsplit(data_resistor, 2)

#Getting b_hat using average function defined in an earlier problem
x_mean = average(data_1)
y_mean = average(data_2)
num_x = [(x - x_mean) for x in data_1]
num_y = [(y - y_mean) for y in data_2]
num = sum(num_x)*sum(num_y)
denom = sum([num_xi**2 for num_xi in num_x])
b_hat = num/denom

#Getting a_hat
a_hat = y_mean - b_hat*x_mean

#Printing outputs of a_hat and b_hat
print('Exercise 8:')
print('The value for a_hat: ', a_hat)
print('The value for b_hat: ', b_hat)


# Outputs
# In case the code does not run correctly for the grader, the following is the output as observed by the author.

# Exercise 1:
 
# Rate of change of y at each step:
# [-0.0049958347219741794, -0.014937587436784194, -0.024730088715635645, -0.03427549512272088, -0.043478432112512344, -0.05224694698069454, -0.06049342762518983, -0.068135477937323, -0.075096741076501]
 
# Number of elements in y array:  10
# Number of elements in delta_y array:  9

 
# Exercise 2:

# Rate of change of y at each step with array arithmetic:
# [-0.00499583 -0.01493759 -0.02473009 -0.0342755  -0.04347843 -0.05224695
#  -0.06049343 -0.06813548 -0.07509674]
 
# Number of elements in y array:  10
# Number of elements in delta_y2 array:  9

 
# Exercise 3:
# 
# The tension in the string in Newtons is:  1449.6502649776662
# With uncertainity in Newtons:  102.08804953780205
 

# Exercise 4:
 
# The Covariance Matrix:
# [[1.         0.41298117]
#  [0.41298117 1.        ]]
 
# The Correlation Coefficient is:  0.412981167572698
 
 
# Exercise 5:
 
# -0.9956240366229848
 
 
# Exercise 6:
 
# The expected value is: -0.5
# The function returns:  -0.5
 
 
# Exercise 7:
 
# The expected mean and standard deviation:  4.5   2.8722813232690143
# The function returns:  (4.5, 2.8722813232690143)
 
 
# Exercise 8:
 
# The value for a_hat:  [10.2]
# The value for b_hat:  [6.11523444e-30]
