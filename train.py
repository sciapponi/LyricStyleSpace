import hydra
from sampler import *
from models import *
from data_processing import *
import random
import numpy as np 
import torch 
import os 
from tensorboardX import SummaryWriter





# Reproducibility settings
def set_reproducibility(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.use_deterministic_algorithms(True)
  os.environ['CUBLAS_WORKSPACE_CONFIG']=':16:8'

def seed_worker(worker_id):
  worker_seed = torch.initial_seed() % 2**32
  np.random.seed(worker_seed)
  random.seed(worker_seed)



# Training Loop
@hydra.main(version_base=None, config_path='config', config_name='train_predict')
def train(cfg)-> None:

	# Set reproducibility

	if cfg.deterministic:
		g_torch = torch.Generator()
		set_reproducibility(cfg.seed)
		g_np = np.random.default_rng (seed=cfg.seed)


	model_ckpt = cfg.model_ckpt
	summary_writer = SummaryWriter(log_dir=cfg.log_dir)
	dataset = load(cfg.data_path)


	# Tokenized Datasets

	train_features, val_features, test_features  = get_features(dataset, model_ckpt)
	
	print(train_features)
	# Samplers:

	# train_sampler = 
	print(dataset)





if __name__=="__main__":
	train()