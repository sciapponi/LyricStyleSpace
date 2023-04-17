import os
import random

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import umap
from cycler import cycler
from tensorboardX import SummaryWriter
from torch.nn.utils.clip_grad import clip_grad_norm_
from tqdm import tqdm

from data_processing import *
from models import *
from sampler import *


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

def get_all_embeddings(sampler, model,device):
  embeddings = []
  labels = []
  model.eval()
  with torch.no_grad():
    for _,((input_ids,attention_mask),label), in enumerate(tqdm(sampler)):
      labels.append(label)
      input_ids = input_ids.to(device)
      attention_mask = attention_mask.to(device)
      embeddings.append(model.compute_embeddings(query_input_ids=input_ids,query_attention_mask=attention_mask))
  return torch.vstack(embeddings), torch.cat(labels)

def visualize_embeddings(embeddings, labels,label_mappings):
  label_set = np.unique(labels)
  num_classes = len(label_set)
  plt.figure(figsize=(20,15))
  plt.gca().set_prop_cycle(
      cycler(
          "color", [plt.cm.nipy_spectral(i) for i in np.linspace(0,0.9, num_classes)]
      )
  )

  for i in range(num_classes):
    idx = (labels==label_set[i])
    plt.plot(embeddings[idx,0], embeddings[idx,1], ".", markersize=10, label=label_mappings[label_set[i]])
  plt.legend(loc="best", markerscale=1)
  plt.show()


def train_step(model,train_sampler,optimizer,loss_fn,writer,log_interval = 0,max_grad_norm=0):        	
	global global_step

	model.train()
	with torch.enable_grad():
		for idx,batch in enumerate(tqdm(train_sampler,position=1)):
			# Zero the gradients and clear the accumulated loss
			optimizer.zero_grad()

			# Move to device
			batch = tuple(t.to(device) for t in batch)
			query_input_ids,query_attention_mask,query_label,support_input_ids,support_attention_mask,support_label = batch

			# Compute loss
			pred = model(query_input_ids,query_attention_mask, support_input_ids,support_attention_mask,support_label)
			loss = loss_fn(pred, query_label)
			loss.backward()

			# Clip gradients if necessary
			if max_grad_norm > 0:
				clip_grad_norm_(model.parameters(), max_grad_norm)

			# Optimize
			optimizer.step()

			# Log training loss
			train_loss = loss.item()
			if log_interval > 0 and global_step % log_interval == 0:
				writer.add_scalar('Training/Loss_IT', train_loss, global_step)
			
			# Increment the global step
			global_step+=1

			# Zero the gradients when exiting a train step
			optimizer.zero_grad()

	return loss.item()


def test_step(model,train_eval_sampler,val_sampler,loss_fn): 
  model.eval()
  with torch.no_grad():

      # First compute prototypes over the training data
      embeddings, labels = [], []
      for batch in train_eval_sampler:
          source_input_ids,source_attention_mask, target = tuple(t.to(device) for t in batch)
          embedding = model.compute_embeddings(source_input_ids,source_attention_mask)
          labels.append(target.cpu())
          embeddings.append(embedding.cpu())
      # Compute prototypes
      embeddings = torch.cat(embeddings, dim=0)
      labels = torch.cat(labels, dim=0)
      prototypes = model.compute_prototypes(embeddings, labels).to(device)

      _preds, _targets = [], []
      for batch in val_sampler:
          # Move to device
          source_input_ids,source_attention_mask, target = tuple(t.to(device) for t in batch)

          pred = model(source_input_ids,source_attention_mask, prototypes=prototypes)
          _preds.append(pred.cpu())
          _targets.append(target.cpu())

      preds = torch.cat(_preds, dim=0)
      targets = torch.cat(_targets, dim=0)
      val_loss = loss_fn(preds, targets).item()
      val_metric = (pred.argmax(dim=1) == target).float().mean().item()
  return val_loss,val_metric

# Training Loop
@hydra.main(version_base=None, config_path='config', config_name='train_predict')
def train(cfg)-> None:
	
	global global_step
	global device 

	global_step = 0

	# Reading configs
	random_seed=cfg.seed
	data_path=cfg.data_path
	model_ckpt = cfg.model_ckpt
	device = cfg.device

	learning_rate = cfg.train.lr
	num_epochs = cfg.train.n_epochs
	n_support = cfg.train.n_support
	n_query = cfg.train.n_query
	n_episodes = cfg.train.n_episodes
	n_classes = cfg.train.n_classes
	eval_batch_size = cfg.train.eval_batch_size
	output_dir = cfg.train.output_dir
	max_grad_norm = cfg.train.max_grad_norm

	log_dir = cfg.logging.log_dir
	log_interval = cfg.logging.log_interval
	
	# Reproducibility settings
	if cfg.deterministic:
		set_reproducibility(random_seed)
		g_torch = torch.Generator()
		g_np = np.random.default_rng (seed=random_seed)

	# Datasets
	dataset = load(data_path)
	artists_mappings = dataset['train'].features['artist'].names
	train_features, val_features, test_features  = get_features(dataset, model_ckpt)	
	
	# Create samplers
	episodic_sampler = EpisodicSampler(MyDataset(train_features),
									n_support=n_support,
									n_query=n_query,
									n_episodes=n_episodes,
									n_classes=n_classes)

	# The train_eval_sampler is used to computer prototypes over the full dataset
	train_sampler = BaseSampler(MyDataset(train_features), batch_size=eval_batch_size)
	val_sampler = BaseSampler(MyDataset(val_features), batch_size=eval_batch_size)
	test_sampler = 	val_sampler = BaseSampler(MyDataset(test_features), batch_size=eval_batch_size)

	# create model	
	model = PrototypicalTransformerModel(model_ckpt,128).to(device)
	
	# Training procedure set up
	best_metric = None
	best_model: Dict[str, torch.Tensor] = dict()
	if not os.path.exists(log_dir):
		os.makedirs(log_dir)
	writer = SummaryWriter(log_dir=log_dir
			)
	loss_fn = torch.nn.CrossEntropyLoss()
	parameters = (p for p in model.parameters() if p.requires_grad)
	optimizer = torch.optim.Adam(parameters, lr=learning_rate)

	# Training start
	print('Beginning training')
	for epoch in tqdm(range(num_epochs),position=0):

		train_loss = train_step(model,episodic_sampler,optimizer,loss_fn,writer,log_interval,max_grad_norm)
		val_loss,val_metric = test_step(model,train_sampler,val_sampler)
	
		lr = optimizer.param_groups[0]['lr']
		# Update best model
		if best_metric is None or val_metric > best_metric:
			best_metric = val_metric
			best_model_state = model.state_dict()
			for k, t in best_model_state.items():
				best_model_state[k] = t.cpu().detach()
			best_model = best_model_state

		# Log metrics
		tqdm.write(f'Training Loss: {train_loss}')
		tqdm.write(f'Validation loss: {val_loss}')
		tqdm.write(f'Validation accuracy: {val_metric}')
		writer.add_scalar('Hyperparameters/Learning Rate', lr, epoch)
		writer.add_scalar('Training/Loss', train_loss, epoch)
		writer.add_scalar('Validation/Loss', val_loss, epoch)
		writer.add_scalar('Validation/Accuracy', val_metric, epoch)

	# Save the best model
	print("Finished training.")
	torch.save(best_model, os.path.join(output_dir, 'model.pt'))


	# Model evaluation 
	best_model = PrototypicalTransformerModel(model_ckpt,128).to(device)
	best_model.load_state_dict(torch.load('/content/out/prototype/model.pt'))
	best_model.eval()
	
	# Visualize the new embeddings 
	umap_visualizer = umap.UMAP()
	embeddings, labels = get_all_embeddings(MyDataset(train_features), best_model,device)
	embeddings_reduced = umap_visualizer.fit_transform(embeddings.cpu().numpy())
	visualize_embeddings(embeddings_reduced, labels.cpu().numpy(),artists_mappings)


if __name__=="__main__":
	train()
