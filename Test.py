# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 12:33:02 2021

@author: shind
"""

import numpy as np
from scipy.stats import levy_stable
from scipy.optimize import minimize
import random
import matplotlib.pyplot as plt
from lmfit import Model, Parameter
from scipy.stats import beta as betaf

alpha,beta,gamma,delta = 1.7, 0.0, 1.0, 1.0 # defining parameters of stable destribution
number_of_points = 750 #number or points
number_quantile = int(float(number_of_points)/100) #number of points for quantile
range_of_x = 16 #total range of simulated range centered on the mean of distribution
fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True) #creation of the figure for bar charts

##### 1st part
numb = np.zeros(number_of_points) #x array

maxi = levy_stable.pdf(delta,alpha,beta,gamma,delta) #maximum value of pdf(probability distribution function) is located at the delta velue if beta == 0
while np.count_nonzero(numb)<number_of_points:
    x = random.random()*range_of_x-range_of_x/2+delta # next two rows is generation of random points in the rectangle with sides of x_range and maximum value of the function
    y = random.random()*maxi
    yreal = levy_stable.pdf(x,alpha,beta,gamma,delta)
    if (y<yreal):
        numb[np.count_nonzero(numb)]=x #if value is underneath the pdf, then the value is stored

##### 2nd part
def mult_d_r(num,n): #function using iterrative approach for calculation of an n-days return
    res = numb[num]
    for i in range(0,n-1):
        res = res*(numb[num+i]+1)+numb[num+i] #recursive formulae for defining (n+1)-day return
    return res

t_d_r = [] # ten days return
for i in range(0,740):
    t_d_r.append(mult_d_r(i,10)) #filling the array with 10-days return

t_d_r.sort() #sorting for defining quantile

##### 3rd part: result  presentation
bin_heights, bin_borders, _ = axs[0].hist(numb,bins = int(float(number_of_points)**(1./2)), label = "1-day return (Conventional units)")
bin_centers = bin_borders[:-1] + np.diff(bin_borders) / 2 #defining points for future fit
maxbin = max(bin_heights)
axs[1].hist(t_d_r,bins = int(float(number_of_points)**(1./2)*2), label = "10-day return (Conventional units)")
axs[1].axvline(x=t_d_r[number_quantile], color = "yellow") #quantile line
print(t_d_r[number_quantile])

##### 4th part: fitting 
def stab_dist(x,A): #definition of fitting function with fixed parameters and amplitude
    return A*levy_stable.pdf(x,alpha,beta,gamma,delta)
   
def fit_o_t(x,y): #fit function for original timeseries
    x_for_fit = np.linspace(x[0],x[len(x)-1],500)
    mod = Model(stab_dist, A=Parameter('A', min = 0))
    result = mod.fit(y, A=maxbin*maxi, x=x)
    y_for_fit = mod.eval(result.params,x=x_for_fit)
    axs[0].plot(x_for_fit,y_for_fit)
    print(result.fit_report())
    
fit_o_t(bin_centers,bin_heights) #fit of original timeseries

fig1, ax = plt.subplots(1, 1) #plotting of fit


##### 5th part: calculation of beta function for defining quantile uncertainity
a = 375+1
b = 750-350+1

x_b = np.linspace(betaf.ppf(0.01, a, b),
                betaf.ppf(0.99, a, b), 100)

ax.plot(x_b, betaf.pdf(x_b, a, b),
       'r-', lw=5, alpha=0.6, label='beta pdf')



""" spare part - for the generation using inverse function approach
def diff(x,a):
    yt = levy_stable.cdf(x,alpha,beta,gamma,delta)
    return (yt - a)**2

for idx,x_value in enumerate(numb[0]):
    res = minimize(diff, 0.0, args=(x_value), method='Nelder-Mead', tol=1e-4)
    #print(res.x,idx)
    numb[1][idx] = res.x[0]

f = open('function.txt', 'w')

for i in range(0,number_of_points):
    f.write(numb[0][i], numb[1][i])

f.close()
"""
