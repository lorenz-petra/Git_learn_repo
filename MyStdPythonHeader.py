# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 12:27:53 2020
Stadard Header for python coding - toolkit of packages

@author: LP
"""

####################################################################################
#                               IMPORTING LIBRARIES AND AUXILIARY PACKAGES
####################################################################################
import numpy as np
import scipy as sp
import pandas as pd
from pandas import ExcelWriter
from matplotlib import pyplot as plt
import matplotlib.colors as colors
from mpl_axes_aligner import align
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting 
######### Plotting features -  rcParams dict
#plt.rcParams["figure.figsize"] = (20,12)
from matplotlib import rcParams
rcParams['figure.dpi'] = 200    # good for publications
rcParams['axes.labelsize'] = 9
rcParams['xtick.labelsize'] = 9
rcParams['ytick.labelsize'] = 9
rcParams['legend.fontsize'] = 9
#rcParams['font.family'] = 'serif'
#plt.rc('font', family='serif')
#rcParams['font.serif'] = ['Computer Modern Roman']
#rcParams['text.usetex'] = True
#rcParams['figure.figsize'] =  (7.3, 4.2)     # if the figures should span one column in a paper
#rcParams['figure.figsize'] = (16, 10)     # if the figures should span two columns in a paper

#rc('text', usetex=True)
#from matplotlib import rcParams
#rcParams['figure.figsize'] = (10, 6)
#rcParams['legend.fontsize'] = 16
#rcParams['axes.labelsize'] = 16
# figure size parameters
Figwidth = 2*3.487
Figheight = Figwidth / 1.618
rcParams['figure.figsize'] = Figwidth,Figheight
#plt.rcParams["figure.figsize"] = (Figwidth,Figheight)   # useful when you plot inline (Jupyter) or without a plt.figure environment
colors_lista =  ['b', 'g', 'r', 'c', 'm', 'y', 'k']
lines_lista =   ['-', '--', '-.', ':', 'None', ' ', '']
markers_lista = [".", ".",  ".",  "4",  "d",   "s",  "x",    "o" ,  "p",      "h"]
legenda_lista = list(zip([".", ".",  ".",  "4",  "d",   "s",  "x"], ['b', 'g', 'r', 'c', 'm', 'y', 'k']))
legenda_lista = [ a + b for (a, b) in legenda_lista]

#colors_lista =  list(colors._colors_full_map.values()) 


import seaborn as sns
import math

import os
import sys
import platform
import argparse # to use command lines arguments

from functools import partial      # to allow building list of functions with only some arguments given (coefficients) and others left as variables

import nptdms    # library for LabView file reading/writing
from scipy import signal
from scipy import integrate
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
from scipy import fftpack
from scipy.optimize import least_squares
from scipy.optimize import fmin_slsqp

# For Symbolic plots
from sympy import symbols
from sympy.plotting import plot as symplot


#import pybobyqa

####################################################################################
#                               SWITCHERS
####################################################################################
Publication = "OFF"  # swithc to "ON" if you require publication quality figures

TwoyAxis = "OFF"

Trellis = "OFF"
####################################################################################
#                               Plot Example - object oriented approach
####################################################################################


print("================================== ======= ==================================")
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> SUMMARY <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
print("================================== ======= ==================================")


figCarboFeNO = plt.figure("Esempio plot OO")
axFeNO = figCarboFeNO.add_subplot(1,1,1)
axFeNO.set_xlabel("Exhaled Volume / L")
axFeNO.set_ylabel("FeNO /ppb")
axFeNO.set_title('FeNO flow-rate trend')

x = np.arange(0, 1, .1)
for i, j in enumerate(lines_lista):
    axFeNO.plot(x, x**i, color = colors_lista[i], marker = markers_lista[i], linestyle = lines_lista[i], label= legenda_lista[i])

axFeNO.legend(loc ="best")                

titolo = '_Figura{}'.format(i+1)
if Publication == "ON":
    figCarboFeNO.savefig(titolo + ".png", bbox_inches='tight', pad_inches=0.02, dpi=300, transparent = True)
    figCarboFeNO.savefig(titolo + ".pdf", bbox_inches='tight', pad_inches=0.02, dpi=300, transparent = True)
else:
    figCarboFeNO.savefig(titolo, bbox_inches='tight', pad_inches=0.02)



print("================================== ======= ==================================")
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>   END   <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
print("================================== ======= ==================================")




print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>   Two y axis scales plots   <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
if TwoyAxis == "ON":

    def fahrenheit2celsius(temp):
        """
        Returns temperature in Celsius.
        """
        return (5. / 9.) * (temp - 32)


    def convert_ax_c_to_celsius(ax_f):
        """
        Update second axis according with first axis.
        """
        y1, y2 = ax_f.get_ylim()
        ax_c.set_ylim(fahrenheit2celsius(y1), fahrenheit2celsius(y2))
        ax_c.figure.canvas.draw()
    
    fig, ax_f = plt.subplots()
    ax_c = ax_f.twinx()
    
    # automatically update ylim of ax2 when ylim of ax1 changes.
    ax_f.callbacks.connect("ylim_changed", convert_ax_c_to_celsius)
    ax_f.plot(np.linspace(-40, 120, 100))
    ax_f.set_xlim(0, 100)
    
    ax_f.set_title('Two scales: Fahrenheit and Celsius')
    ax_f.set_ylabel('Fahrenheit')
    ax_c.set_ylabel('Celsius')
    
    plt.show()






####################################################################################
#  >>>>>>>>>>>>>>>>>>>>>>>>>>>> 3D stacked  plots Example
####################################################################################

NANGLES = 200

title = "3S stacked spectra plot new"
fig = plt.figure(title)
ax = fig.add_subplot(111, projection='3d')
times = [0.5, 2.5, 5, 7.5, 10]    # secs o exhalation

for ti in range(len(times)):
    tn = times[ti]
    
    y = np.arange(NANGLES) / float(NANGLES)
    x = np.ones(NANGLES) *tn # time axis - descrete variable according to the list  times
    z = np.sin(tn*y*np.pi)  # intensity
    
    ax.plot(x, y, z)
    
ax.set_xlabel('Time / sec')
ax.set_xlim(0, 10)
ax.set_ylabel('Wavenumber / cm-1')
ax.set_zlabel('Intensity / arb')
#ax.set_yticklabels(times) 
ax.view_init(35, -70)   # set the point of view in terms of polar coordinate theta and azimuth angle phi
plt.savefig('3DstackedSpectra.png')
plt.show()











# print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>   Trellis plots   <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")

if Trellis == "ON":
    def facetgrid_two_axes(*args, **kwargs):
        data = kwargs.pop('data')
        dual_axis = kwargs.pop('dual_axis')
        alpha = kwargs.pop('alpha', 0.2)
        kwargs.pop('color')
        ax = plt.gca()
        if dual_axis:
            ax2 = ax.twinx()
            ax2.set_ylabel('Second Axis!')
    
        ax.plot(data['x'],data['y1'], **kwargs, color='red',alpha=alpha)
        if dual_axis:
            ax2.plot(df['x'],df['y2'], **kwargs, color='blue',alpha=alpha)
    
    
    df = pd.DataFrame()
    df['x'] = np.arange(1,5,1)
    df['y1'] = 1 / df['x']
    df['y2'] = df['x'] * 100
    df['facet'] = 'foo'
    df2 = df.copy()
    df2['facet'] = 'bar'
    
    df3 = pd.concat([df,df2])
    win_plot = sns.FacetGrid(df3, col='facet', size=6)
    (win_plot.map_dataframe(facetgrid_two_axes, dual_axis=True)
             .set_axis_labels("X", "First Y-axis"))
    plt.show()
    
    


#########################################
from scipy import stats
tips_all = sns.load_dataset("tips")
tips_grouped = tips_all.groupby(["smoker", "size"])
tips = tips_grouped.mean()
tips["error_min"] = tips_grouped.total_bill.apply(stats.sem) * 1.96
tips["error_max"] = tips_grouped.total_bill.apply(stats.sem) * 1.96
tips.reset_index(inplace=True)




def my_errorbar(*args, **kwargs):
    data = kwargs.pop('data')
    errors = np.vstack([data['error_min'], 
                        data['error_max']])

    #print(errors)
    ax = plt.gca()
    ax.errorbar(data[args[0]], 
                 data[args[1]], 
                 yerr=errors,
                 **kwargs);     
    ax.yaxis.set_label_coords(-0.025,1.05)    # with resepct to a 1x1 system, where to position the label of y-axis

g = sns.FacetGrid(tips, col="smoker", size=5)
g.map_dataframe(my_errorbar, "size", "total_bill", marker="o")


    
    
    
    
ax.yaxis.set_label_coords(-0.1,1.02)




######################################################################################

fig = plt.figure(figsize=(10, 4))
axFeNO = figCarboFeNO.add_subplot(1,1,1)

axFeNO.set_xlabel("Exhaled Volume / L")
axFeNO.set_ylabel("FeNO /ppb")
axFeNO.set_title('FeNO flow-rate trend')

ax2= fig.add_subplot(1, 2, 2)
# ..... same







####################################################################################
#           Symbolic Plot Example - useful to plot functions without data
####################################################################################
from sympy.plotting import plot


# The Burr XII, is useful for modeling lifetime data.
#This distribution is used to model a wide variety of phenomena including crop prices, household income, option market price distributions, risk (insurance) and travel time.
#It is particularly useful for modeling histograms.

x, c, k = symbols('x, c, k')
c = 3   # shape factor
k = 1     # shape factor
BurrXII = c * k * (x**(c-1)) / ((1 + x**c)**(k+1))

sympy_p1 = plot(BurrXII, (x, 0, 5), label = "Burr XII pdf c=3, k=1", show=False)




c = 2   # shape factor
k = 1     # shape factor
sympy_p2 = plot(BurrXII, (x, 0, 5), label = "Burr XII pdf c=2, k=1", show=False)

sympy_p1.append(sympy_p2[0])

sympy_p1.show()



pdf_BurrXII = c * k * (x**(c-1)) / ((1 + x**c)**(k+1))   # Burr probability distribution funciton
F_BurrXII =   1 - (1 + x**c)**(-k)   # Burr cumulative 



plt.legend(loc= "best")