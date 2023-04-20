import glob
import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

methods = [
    'mean_elite',
    'rdm_elite',
    'naive'
]

method_labels = [
    'Mean Elite',
    'Random Elite',
    'Naive'
]

tasks = [
    #'StairClimber',
    'MazeSparse16',
    #'MazeSparse12',
    #'MazeSparse8'
]
data_directory = 'output/results'
output_folder = 'plots'


if __name__ == '__main__':
    os.makedirs(f'{output_folder}', exist_ok=True)
        
    for task in tasks:
        
        print(f'Making plot for task: {task}')

        list_median = []
        list_low = []
        list_high = []
    

        for method in methods:
            
            method_median = []
            method_low = []
            method_high = []

            for num_iterations in [10, 20, 30, 40, 50]:
            
                method_data = pd.read_csv(f'{data_directory}/{task}/{method}/{method}_{num_iterations}.csv')

                #print(list_model_data[0])
                
                method_median.append(method_data['Reward of Final Program:'].median())
                method_low.append(method_data['Reward of Final Program:'].quantile(0.2))
                method_high.append(method_data['Reward of Final Program:'].quantile(0.8))
            
            list_median.append(method_median)
            list_low.append(method_low)
            list_high.append(method_high)
        
        print(f"method: {method}; median:", method_low, method_high)

        plt.figure(figsize=(5,5))
        plt.suptitle(f'{task}')
        plt.xscale('linear')
        plt.xlabel('Number of Search Iterations')
        plt.ylabel('Best Reward') #averaged across seeds
        
        for median, low, high in zip(list_median, list_low, list_high): #median, low, high in zip(list_median, list_low, list_high):
            plt.fill_between([10,20,30,40,50], low, high, alpha=0.2, label='_nolegend_')
            plt.plot([10,20,30,40,50], median, alpha=0.8)
        
        plt.legend(method_labels)
        plt.tight_layout()
        plt.savefig(f'{output_folder}/{task}.png')
        plt.close()

