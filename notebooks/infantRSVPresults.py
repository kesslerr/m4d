
import numpy as np 
import pandas as pd
import pickle
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns

base_dir = os.path.join(os.path.dirname(os.getcwd()))
sys.path.append(base_dir)

subjects = ['sub-001', 'sub-002', 'sub-005', 'sub-007', 'sub-008', 'sub-009', 'sub-011', 'sub-013', 'sub-014', 'sub-015', 'sub-017', 'sub-018', 'sub-020', 'sub-021', 'sub-022', 'sub-023', 'sub-024', 'sub-025', 'sub-026', 'sub-027', 'sub-028', 'sub-030', 'sub-035', 'sub-036', 'sub-037', 'sub-039', 'sub-040', 'sub-042', 'sub-043', 'sub-044', 'sub-045', 'sub-046', 'sub-048', 'sub-049']
categories = ['aquatic', 'bird', 'clothing', 'fruits', 'furniture',  
              'human', 'insect', 'mammal', 'plants', 'tools']

# DEBUG
#subject=subjects[0]

# load matrices
results = np.empty((10,10,len(subjects)))
for s, subject in enumerate(subjects):
    # import numpy matrix
    mat2d = np.load(os.path.join(base_dir, 'models', 'eegnet', 'RSVP', subject, '10x10.csv.npy'))
    results[:,:,s] = mat2d
    


# heatmap  mean across participants
avgResults = np.mean(results, axis=2)
plt.figure(figsize=(10,10))
divcmap = sns.diverging_palette(300, 145, s=60, as_cmap=True, sep=100)
sns.heatmap(avgResults, 
            cmap=divcmap, #'PiYG', 
            center=0.5, annot=True, fmt=".3f",
            cbar_kws={'label': 'Accuracy'},
            xticklabels=categories, yticklabels=categories,
            )
plt.title("Average accuracy across all participants")
plt.show()


# for each participant
plt.figure(figsize=(5*5,5*7))
for s, subject in enumerate(subjects):
    plt.subplot(7,5,s+1)
    sns.heatmap(results[:,:,s], 
                cmap=divcmap, #'PiYG', 
                center=0.5, annot=True, fmt=".2f",
                cbar=False,
                xticklabels=categories, yticklabels=categories)
    plt.title(subject)
    plt.axis('off')
#plt.suptitle("Accuracy per participant")
plt.tight_layout()
plt.show()
