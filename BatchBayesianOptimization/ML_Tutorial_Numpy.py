# ML Python Numpy, Scipy, Sckikit-Learn Tutorial
import numpy as np

#Create an array from a list
#x = np.array([1., 2., 3., 4.])
x = np.array([1., 2., 3., 4.], ndmin=2) #2D array

#Get transpose of an array
x_transpose = x.T
x_transpose = np.transpose(x)

#by default, numpy has a 'one' dimensional array behaviour
print('array x: ', x)
print('shape x: ', x.shape)
print('x == x\': ', (x.shape == x_transpose.shape)) #comparing shapes, they are the same
# odd since transpose is opposite of original array
# Solution: make x 2D

