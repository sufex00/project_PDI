# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 16:34:23 2018

@author: pedro_barros
"""

import matplotlib.pyplot as plt



Sens = [1/28., 10/42., 16/32., 23/34., 29/36., 34/39., 38/41., 33/34.]
Spec = [31/31., 17/18., 25/28., 21/26., 15/24., 9/27., 5/19., 0/26.]
auc = 0.
for i in range(1,len(Sens)):
    auc += (Spec[i]-Spec[i-1]) * (Sens[i] + Sens[i-1])/2

#plot the ROC curve
plt.clf()
plt.grid(True)
plt.plot([1-x for x in Spec], Sens, '-o', color='blue', lw=2, label = 'AUC = 0.7695')
plt.xlabel("1 - Specificity", size=12)
plt.ylabel("Sensitivity", size=12)
plt.title("ROC")
plt.xlim([0,1])
legend = plt.legend(loc='upper center', shadow=True)

# The frame is matplotlib.patches.Rectangle instance surrounding the legend.
frame = legend.get_frame()
frame.set_facecolor('0.90')

# Set the fontsize
for label in legend.get_texts():
    label.set_fontsize('large')

for label in legend.get_lines():
    label.set_linewidth(1.5)  # the legend line width