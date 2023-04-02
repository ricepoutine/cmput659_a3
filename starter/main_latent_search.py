import torch
from config import Config
from dsl import DSL
from logger.stdout_logger import StdoutLogger
from vae.models import load_model
from search.latent_search import LatentSearch
from tasks import get_task_cls


if __name__ == '__main__':
    
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
    
    best_program, best_reward = searcher.search()
    
    StdoutLogger.log('Main', f'Final program: {best_program}') 
    StdoutLogger.log('Main', f'Reward of Final Program: {best_reward}') 
