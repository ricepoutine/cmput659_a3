import torch
from config import Config
from dsl import DSL
from logger.stdout_logger import StdoutLogger
from vae.models import load_model
from search.latent_search import LatentSearch
from tasks import get_task_cls
import csv
import time

def trial():

    print(f"Beginning with trial with seed:", Config.env_seed)

    dsl = DSL.init_default_karel()

    if Config.disable_gpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = load_model(Config.model_name, dsl, device)
    
    task_cls = get_task_cls(Config.env_task)
    
    params = torch.load(f'params/leaps_vae_256.ptp', map_location=device)
    model.load_state_dict(params, strict=False)
    
    searcher = LatentSearch(model, task_cls, dsl)
    
    StdoutLogger.log('Main', f'Starting Latent Search with model {Config.model_name} for task {Config.env_task}')
    
    st = time.process_time()

    best_program, best_reward = searcher.search()
    #best_program, best_reward = searcher.search_better()
    et = time.process_time()

    #optional: look at gifs
    #searcher.save_gifs(best_program)
    
    StdoutLogger.log('Main', f'Final program: {best_program}') 
    StdoutLogger.log('Main', f'Reward of Final Program: {best_reward}') 

    with open(
        f"output/results/{Config.env_task}/{Config.search_type}/{Config.search_type}_{Config.search_number_iterations}" + ".csv", "a", newline=""
    ) as csvfile:
        logger = csv.writer(csvfile, delimiter=",", quoting=csv.QUOTE_MINIMAL)
        logger.writerow([Config.env_seed, str(et-st), best_reward.item(), best_program])



if __name__ == '__main__':
    #main experiment control

    num_trials = 20 #20 = number of seeds, used to report average best program and error
    search_iterations = [10, 20] #[10, 20, 30, 40, 50]

    for num_iterations in search_iterations:
        Config.search_number_iterations = num_iterations
        with open(
            f"output/results/{Config.env_task}/{Config.search_type}/{Config.search_type}_{Config.search_number_iterations}" + ".csv", "w+", newline=""
        ) as csvfile:
            logger = csv.writer(csvfile, delimiter=",", quoting=csv.QUOTE_MINIMAL)
            logger.writerow(["Seed:", "Runtime:", "Reward of Final Program:", "Final program:"])

        for seed in range(1,num_trials+1): # just to make counting seeds easier, in case more seeds are wanted
            Config.env_seed = seed
            trial()


# for ease of access: (maze sparse, no gpu)
# python main_latent_search.py --env_seed 1 --search_number_iterations 20 --search_population_size 1028 --env_is_crashable --search_type naive --env_task MazeSparse --disable_gpu