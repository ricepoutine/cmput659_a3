import glob
import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from search.latent_search import LatentSearch
from tasks import get_task_cls
from config import Config
from dsl import DSL
from logger.stdout_logger import StdoutLogger
from vae.models import load_model
import torch

if __name__ == '__main__':

    naive_data = pd.read_csv('output/results/StairClimber/naive/naive_50_resized.csv')
    
    naive_lengths = [len(program.split(" ")) for program in naive_data['Final program:']]

    print("naive 50 median, max, min on stairclimber:", np.mean(naive_lengths), np.max(naive_lengths), np.min(naive_lengths))
    
    elite_data = pd.read_csv('output/results/StairClimber/rdm_elite/rdm_elite_50_resized.csv')
    
    elite_lengths = [len(program.split(" ")) for program in elite_data['Final program:']]

    print("rdm_elite_50 median, max, min on stairclimber:", np.mean(elite_lengths), np.max(elite_lengths), np.min(elite_lengths))

    
    #generalization test: run on much larger stairclimber problem

    Config.env_height = 100
    Config.env_width = 100
    Config.env_seed = 100 # i dont think this changes much

    dsl = DSL.init_default_karel()

    if Config.disable_gpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = load_model(Config.model_name, dsl, device)
    
    task_cls = get_task_cls('StairClimber')
    
    params = torch.load(f'params/leaps_vae_256.ptp', map_location=device)
    model.load_state_dict(params, strict=False)
    
    executer = LatentSearch(model, task_cls, dsl)

    new_naive_rewards = pd.DataFrame(executer.execute_programs(naive_data['Final program:'].tolist()))
    new_elite_rewards = pd.DataFrame(executer.execute_programs(elite_data['Final program:'].tolist()))

    print(new_naive_rewards.median(), new_naive_rewards.quantile(0.2), new_naive_rewards.quantile(0.8))
    print(new_elite_rewards.median(), new_elite_rewards.quantile(0.2), new_elite_rewards.quantile(0.8))

