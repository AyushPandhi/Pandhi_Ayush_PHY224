#Author: Ayush Pandhi
#Course: PHY224
#Updated: 09/17/2018

#Importing required modules
import numpy as np

#Data for Loop 1
D1 = [5.80, 5.80, 5.85, 5.80, 5.80, 5.85]
D1_mean = np.mean(D1)
D1_std = np.std(D1)
Tsqr1 = [2.44/5, 2.44/5, 2.34/5, 2.43/5, 2.13/5, 2.50/5]
Tsqr1_mean = np.mean(Tsqr1)
Tsqr1_std = np.std(Tsqr1)
print('Loop 1:')
print('------------------------------------------------------')
print('Mean diameter: ' + str(D1_mean))
print('Standard deviation in diameter: ' + str(D1_std))
print('Mean period: ' + str(Tsqr1_mean))
print('Standard deviation in period: ' + str(Tsqr1_std))
print(' ')

#Data for Loop 2
D2 = [11.20, 11.20, 11.20, 11.25, 11.20, 11.30]
D2_mean = np.mean(D2)
D2_std = np.std(D2)
Tsqr2 = [3.34/5, 3.56/5, 3.31/5, 3.31/5, 3.41/5, 3.44/5]
Tsqr2_mean = np.mean(Tsqr2)
Tsqr2_std = np.std(Tsqr2)
print('Loop 2:')
print('------------------------------------------------------')
print('Mean diameter: ' + str(D2_mean))
print('Standard deviation in diameter: ' + str(D2_std))
print('Mean period: ' + str(Tsqr2_mean))
print('Standard deviation in period: ' + str(Tsqr2_std))
print(' ')

#Data for Loop 3
D3 = [21.45, 21.40, 21.50, 21.55, 21.50, 21.50]
D3_mean = np.mean(D3)
D3_std = np.std(D3)
Tsqr3 = [4.78/5, 4.75/5, 4.78/5, 4.81/5, 4.68/5, 4.66/5]
Tsqr3_mean = np.mean(Tsqr3)
Tsqr3_std = np.std(Tsqr3)
print('Loop 3:')
print('------------------------------------------------------')
print('Mean diameter: ' + str(D3_mean))
print('Standard deviation in diameter: ' + str(D3_std))
print('Mean period: ' + str(Tsqr3_mean))
print('Standard deviation in period: ' + str(Tsqr3_std))
print(' ')

#Data for Loop 4
D4 = [44.20, 44.50, 44.40, 44.90, 44.90, 44.90]
D4_mean = np.mean(D4)
D4_std = np.std(D4)
Tsqr4 = [6.75/5, 6.72/5, 7.01/5, 6.56/5, 6.84/5, 6.59/5]
Tsqr4_mean = np.mean(Tsqr4)
Tsqr4_std = np.std(Tsqr4)
print('Loop 4:')
print('------------------------------------------------------')
print('Mean diameter: ' + str(D4_mean))
print('Standard deviation in diameter: ' + str(D4_std))
print('Mean period: ' + str(Tsqr4_mean))
print('Standard deviation in period: ' + str(Tsqr4_std))