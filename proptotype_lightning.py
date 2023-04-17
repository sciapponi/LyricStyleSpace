# %%
import numpy as np
import torch
import random
import os 
from datasets import load_dataset
import re
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm
from datasets import DatasetDict
import torch
from transformers import AutoModel, AutoTokenizer
from torch import Tensor
from torch.nn.utils.clip_grad import clip_grad_norm_
from typing import Optional,Union,Tuple, Dict, Any
from torch.utils.data import DataLoader
from cycler import cycler
import gc
from pytorch_metric_learning.samplers import MPerClassSampler
from torch.utils.data import Dataset,DataLoader
from typing import Tuple, Iterator, Dict, Any, Sequence
import umap
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score,adjusted_rand_score,accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import pytorch_lightning as pl
from torch.utils.tensorboard import SummaryWriter
# %%

def set_reproducibility(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.use_deterministic_algorithms(True)
  g_torch.manual_seed(random_seed)
  g_np = np.random.default_rng (seed=random_seed)
  os.environ['CUBLAS_WORKSPACE_CONFIG']=':16:8'

def seed_worker(worker_id):
  worker_seed = torch.initial_seed() % 2**32
  np.random.seed(worker_seed)
  random.seed(worker_seed)

random_seed = 42
g_torch = torch.Generator()
g_np = np.random.default_rng (seed=random_seed)
set_reproducibility(random_seed)



# dataset = load_dataset('json',data_files='data.json').class_encode_column('artist')
dataset = load_dataset('json',data_files='data_self_dump.json').class_encode_column('artist').filter(lambda x: len(x['lyrics'])>0)
artists_mappings = dataset['train'].features['artist'].names

# %%
def clean(example):
    # allowed_parts = ['verse','break','chorus','intro', 'interlude', 'bridge', 'outro']
    allowed_parts = None
    example['lyrics']=example['lyrics'][example['lyrics'].index('Lyrics')+6:] 
    if allowed_parts is not None: 
        for part in allowed_parts:
            example['lyrics']=re.sub("\[.*"+part+".*\]", f"[{part}]", example['lyrics'], flags=re.IGNORECASE)
        example['lyrics']=re.sub("\[(?!"+"|".join(allowed_parts)+").*?\]", "", example['lyrics'], flags=re.DOTALL)
    else: 
        example['lyrics']=re.sub("\[.*\]", "", example['lyrics'], flags=re.IGNORECASE)
    example['lyrics']=re.sub("[0-9]+embed", "", example['lyrics'], flags=re.IGNORECASE)
    return example

mapped_dataset = dataset.map(clean)


def list_song_parts(example):
    parts = re.findall(r'\[[^\[\]]+\]',example['lyrics']) # Capture everything enclosed in square brackets
    for i,part in enumerate(parts): 
        parts[i] = re.sub(r':.*(?=\])','',part) # Remove everything from : to the closing bracket ] (Most lyrics contain the name of the singer of these parts e.g. [Chorus: 2 Chainz])
    return {'parts': parts}
parts = mapped_dataset['train'].map(list_song_parts,remove_columns=dataset['train'].column_names)

parts:np.ndarray = np.unique([el for l in parts['parts'] for el in l ])


print(*parts)


tts_mapped_dataset = mapped_dataset['train'].train_test_split(train_size=0.7,stratify_by_column='artist')
tvs_mapped_dataset = tts_mapped_dataset['test'].train_test_split(train_size=0.5,stratify_by_column='artist')

train_test_val_dataset = DatasetDict({
    'train': tts_mapped_dataset['train'],
    'test':tvs_mapped_dataset['test'],
    'val': tvs_mapped_dataset['train']
    
})


def plot_dist(dataset,**kwargs):
    counts = {}
    for example in dataset:
        if example['artist'] not in counts.keys():
            counts[example['artist']] = 0
        else:
            counts[example['artist']] += 1
    plt.bar(counts.keys(), counts.values(),**kwargs)


plt.figure(figsize=(9,6))
plt.title("Train-Test-Validation barplot")
plt.ylabel("count")
plt.xlabel("artist")
plot_dist(train_test_val_dataset['train'],width=0.8)
plot_dist(train_test_val_dataset['test'],width=0.6)
plot_dist(train_test_val_dataset['val'],width=0.4)
plt.legend(['train', 'test', 'val'])
plt.show()



# LIGHTNING MODEL FOR TRAINING

class LyricStyleModel(pl.LightningModule):

    def __init__(self, config=None):

        super(LyricStyleModel, self).__init__()

        #training parameters
        self.learning_rate = config.learning_rate
        self.num_epochs = config.train.num_epochs
        self.batch_size = config.train.batch_size
        
        self.loss_fn = torch.nn.CrossEntropyLoss()

        #model config
        if config.model == "Prototype":
            self.net = PrototypicalTransformerModel(model_ckpt='bert-base-uncased',output_dim=config.num_classes)
        else:
            raise Exception(f" {config.model} model Not Yet implemented!")

    def forward(self, x):

        out = self.net(x)

        return out

    def configure_optimizers(self):
        opt = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
        )

        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    opt,
                    patience=20,
                    verbose=True,
                ),
                "monitor": "val_loss",
            },
        }

    def training_step(self, batch, batch_idx):
        query_input_ids,query_attention_mask,query_label,support_input_ids,support_attention_mask,support_label = batch

        # Compute loss
        pred = self.net(query_input_ids,query_attention_mask, support_input_ids,support_attention_mask,support_label)
        loss = self.loss_fn(pred, query_label)
        

        self.log(
            "train_loss", loss, on_epoch=True, prog_bar=True, logger=True, on_step=False
        )

        return loss

    # def test_step(self):
    #     pass




###############################################################################################################################################
device = 'cpu'
if torch.cuda.is_available():
  device='cuda'
  print('All good. A CPU is availbale')
else: 
  print('Restart the runtime with a compatible CUDA device.')

# Create model with warm-start weights from pretrained checkpoint
model_ckpt = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = PrototypicalTransformerModel(model_ckpt,128).to(device)


# %%
# Tokenize
def prepare_train_features(examples):
    tokenized_examples = tokenizer(
        examples['lyrics'],
        truncation=True,
        padding=True,
        max_length=512
        )
    tokenized_examples['labels'] = examples['artist']
    return tokenized_examples

train_features = train_test_val_dataset['train'].map(prepare_train_features, batched=True, remove_columns=train_test_val_dataset["train"].column_names).with_format('torch')
test_features = train_test_val_dataset['test'].map(prepare_train_features, batched=True, remove_columns=train_test_val_dataset["test"].column_names).with_format('torch')
val_features = train_test_val_dataset['val'].map(prepare_train_features, batched=True, remove_columns=train_test_val_dataset["val"].column_names).with_format('torch')

# %% [markdown]
# ## Utilities

# %%
def get_all_embeddings(dataset, model,device):
  dataloader = DataLoader(dataset,batch_size=16)
  embeddings = []
  labels = []
  model.eval()
  with torch.no_grad():
    for _,((input_ids,attention_mask),label), in enumerate(tqdm(dataloader)):
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

def clear_cache():
  gc.collect()
  torch.cuda.empty_cache()
  torch.cuda.synchronize()
   
clear_cache()
print(f'{torch.cuda.memory_allocated()/1024**2} MB')


class MyDataset(Dataset):
    def __init__(self,dataset) -> None:
        super().__init__()
        self.dataset = dataset
    def __getitem__(self, index):
        index = int(index)
        return (self.dataset[index]['input_ids'],self.dataset[index]['attention_mask']),self.dataset[index]['labels']
    def __len__(self):
        return len(self.dataset)


class EpisodicSampler(object):
    """Implement an Episodic sampler."""

    def __init__(self,
                 data: Sequence[Tuple[Tuple[Tensor,Tensor], Tensor]],
                 n_support: int,
                 n_query: int,
                 n_episodes: int,
                 n_classes: int = None,
                 balance_query: bool = False) -> None:
        """Initialize the EpisodicSampler.

        Parameters
        ----------
        data: Sequence[Tuple[torch.Tensor, torch.Tensor]]
            The input data as a list of (sequence, label) pairs
        n_support : int
            The number of support points per class
        n_query : int
            If balance_query is True, this should be the number
            of query points per class, otherwise, this is the total
            number of query points for the episode
        n_episodes : int
            Number of episodes to run in one "epoch"
        n_classes : int, optional
            The number of classes to sample per episode, defaults to all
        pad_index : int, optional
            The padding index used on sequences.
        balance_query : bool, optional
            If True, the same number of query points are sampled per
            class, otherwise query points are sampled uniformly
            from the input data.

        """

        self.n_support = n_support
        self.n_query = n_query
        self.n_classes = n_classes
        self.n_episodes = n_episodes

        self.balance_query = balance_query

        if len(data) == 0:
            raise ValueError("No examples provided")

        # Split dataset by target
        self.target_to_examples: Dict[int, Any] = dict()
        for source, target in data:
            self.target_to_examples.setdefault(int(target), []).append((source, target))

        self.all_classes = list(self.target_to_examples.keys())

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, ...]]:
        """Sample from the list of features and yields batches.

        Yields
        ------
        Iterator[Tuple[Tensor, Tensor, Tensor, Tensor]]
            In order: the query_source, the query_target
            the support_source, and the support_target tensors.
            For sequences, the batch is used as first dimension.

        """
        for _ in range(self.n_episodes):
            # Sample n_classes to run a training episode over
            classes = self.all_classes
            if self.n_classes is not None:
                classes = list(np.random.permutation(self.all_classes))[:self.n_classes]

            # Sample n_support and n_query points per class
            supports, queries = [], []
            for i, target_class in enumerate(classes):
                examples = self.target_to_examples[target_class]
                indices = np.random.permutation(len(examples))
                supports.extend([(examples[j][0], i) for j in indices[:self.n_support]])

                if self.balance_query:
                    query_indices = indices[self.n_support:self.n_support + self.n_query]
                    queries.extend([(examples[j][0], i) for j in query_indices])
                else:
                    queries.extend([(examples[j][0], i) for j in indices[self.n_support:]])

            if not self.balance_query:
                indices = np.random.permutation(len(queries))
                queries = [queries[i] for i in indices[:self.n_query]]

            query_source, query_target = list(zip(*queries))
            support_source, support_target = list(zip(*supports))

            query_input_ids,query_attention_mask = list(zip(*query_source))
            query_input_ids = torch.stack(query_input_ids)
            query_attention_mask = torch.stack(query_attention_mask)

            query_target = torch.tensor(query_target)

            support_input_ids,support_attention_mask = list(zip(*support_source))
            support_input_ids = torch.stack(support_input_ids)
            support_attention_mask = torch.stack(support_attention_mask)

            support_target = torch.tensor(support_target)

            if len(query_target.size()) == 2:
                query_target = query_target.squeeze()
            if len(support_target.size()) == 2:
                support_target = support_target.squeeze()

            yield (query_input_ids.long(),
                   query_attention_mask.long(),
                   query_target.long(),
                   support_input_ids.long(),
                   support_attention_mask.long(),
                   support_target.long())
    def __len__(self):
      return self.n_episodes

class BaseSampler(object):

    def __init__(self,
                 data: Sequence,
                 shuffle: bool = True,
                 batch_size: int = 16):
        """A basic sampler.

        Parameters
        ----------
        data : Sequence[Tuple[Tensor, Tensor]]
            The input data to sample from, as a list of
            (source, target) pairs
        shuffle : bool, optional
            Whether to shuffle the data, by default True
        batch_size : int, optional
            The batch size to use, by default 16

        """
        self.data = data
        self.shuffle = shuffle
        self.batch_size = batch_size

    def __iter__(self):
        """Sample from the list of features and yields batches.

        Yields
        ------
        Iterator[Tuple[Tensor, Tensor]]
            In order: source, target
            For sequences, the batch is used as first dimension.

        """
        if self.shuffle:
            indices = np.random.permutation(len(self.data))
        else:
            indices = list(range(len(self.data)))

        num_batches = len(indices) // self.batch_size
        indices_splits = np.array_split(indices, num_batches)
        for split in indices_splits:
            examples = [self.data[i] for i in split]
            source, target = list(zip(*examples))
            source_input_ids,source_attention_mask = list(zip(*source))
            source_input_ids = torch.stack(source_input_ids)
            source_attention_mask = torch.stack(source_attention_mask)
            target = torch.tensor(target)
            yield (source_input_ids.long(),source_attention_mask.long(), target.long())



# %% [markdown]
# 
# ## Training Loop

# %% [markdown]
# ### Definition

# %%
def train(model,train_sampler,optimizer,writer,epoch,max_grad_norm=None):
  loss_fn = torch.nn.CrossEntropyLoss()
  num_batches = len(train_sampler)
  log_interval = 1
  
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
      if max_grad_norm is not None:
          clip_grad_norm_(model.parameters(), max_grad_norm)

      # Optimize
      optimizer.step()

      
      # Log training loss
      train_loss = loss.item()
      if log_interval > 0:
        if idx % log_interval == 0:
            global_step = idx + (epoch * num_batches)
            writer.add_scalar('Training/Loss_IT', train_loss, global_step)

    # Zero the gradients when exiting a train step
    optimizer.zero_grad()
  return loss.item()


def test(model,train_eval_sampler,val_sampler,device): 
  loss_fn = torch.nn.CrossEntropyLoss()
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

def training_loop(model,train_sampler,train_eval_sampler,val_sampler,num_epochs=3,max_grad_norm=None,device='cpu'):
    """Run Training """
    learning_rate= 1e-5
    best_metric = None
    best_model: Dict[str, torch.Tensor] = dict()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    writer = SummaryWriter(log_dir=output_dir)

    loss_fn = torch.nn.CrossEntropyLoss()

    parameters = (p for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)

    # print("Beginning training.")
    for epoch in tqdm(range(num_epochs),position=0):

        #       TRAIN        
      
        train_loss = train(model,train_sampler,optimizer,writer,epoch,max_grad_norm)
        #       EVALUATE        #
        val_loss,val_metric = test(model,train_eval_sampler,val_sampler,device)


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

# %% [markdown]
# ### Hyperparameters

# %%


output_dir = 'out/prototype'

# Training hyperparameters
n_support = 4
n_query = 5
n_episodes = 20
n_classes = 3
eval_batch_size = 16

# Create samplers
train_sampler = EpisodicSampler(MyDataset(train_features),
                                n_support=n_support,
                                n_query=n_query,
                                n_episodes=n_episodes,
                                n_classes=n_classes)

# The train_eval_sampler is used to computer prototypes over the full dataset
train_eval_sampler = BaseSampler(MyDataset(train_features), batch_size=eval_batch_size)
val_sampler = BaseSampler(MyDataset(train_features), batch_size=eval_batch_size)

# %% [markdown]
# ### Train

# %%
clear_cache()
set_reproducibility(random_seed)
training_loop(model,train_sampler,train_eval_sampler,val_sampler,5,1,device)

# %% [markdown]
# ### Visualize embeddings

# Visualize the new embeddings 
best_model = PrototypicalTransformerModel(model_ckpt,128).to(device)
best_model.load_state_dict(torch.load('/content/out/prototype/model.pt'))
best_model.eval()
# Visualize the new embeddings 
umap_visualizer = umap.UMAP()
embeddings, labels = get_all_embeddings(MyDataset(train_features), best_model,device)
embeddings_reduced = umap_visualizer.fit_transform(embeddings.cpu().numpy())
visualize_embeddings(embeddings_reduced, labels.cpu().numpy(),artists_mappings)

# %%
def pca_lowrank(A,n=2):
  U,S,V = torch.pca_lowrank(A, q=n, center=True, niter=2)
  return torch.matmul(U[:,:n],torch.diag(S[:n]))


def visualize_embeddings3D(embeddings, labels,label_mappings):
  label_set = np.unique(labels)
  num_classes = len(label_set)
  fig = plt.figure(figsize=(20,15))
  ax = fig.add_subplot(projection='3d')

  ax.set_prop_cycle(
      cycler(
          "color", [plt.cm.nipy_spectral(i) for i in np.linspace(0,0.9, num_classes)]
      )
  )

  for i in range(num_classes):
    idx = (labels==label_set[i])
    ax.plot(embeddings[idx,0], embeddings[idx,1], embeddings[idx,2], ".", markersize=10, label=label_mappings[label_set[i]])
  plt.legend(loc="best", markerscale=1)
  plt.show()

embeddings_reduced = pca_lowrank(embeddings,3)
visualize_embeddings3D(embeddings_reduced.cpu().numpy(), labels.cpu().numpy(),artists_mappings)


# %% [markdown]
# ### Evaluation on the test set

# %%
test_sampler = BaseSampler(MyDataset(test_features),batch_size=eval_batch_size)
test(best_model,train_eval_sampler,test_sampler,device)


clustering = KMeans(len(artists_mappings))
clustering.fit(embeddings.cpu().numpy())


# %%
print('ADJ:',adjusted_rand_score(clustering.predict(embeddings.cpu().numpy()),labels.cpu().numpy()))
print('Silhouette:',silhouette_score(embeddings.cpu().numpy(),clustering.predict(embeddings.cpu().numpy())))

# %%
print('Silhouette:',silhouette_score(embeddings.cpu().numpy(),labels.cpu().numpy()))

# %%
accuracy_score(clustering.predict(embeddings.cpu().numpy()),labels.cpu().numpy())

# %%

X_train, Y_train = tuple(x.cpu().numpy() for x in get_all_embeddings(MyDataset(train_features), best_model,device))
X_test, Y_test = tuple(x.cpu().numpy() for x in get_all_embeddings(MyDataset(test_features), best_model,device))


# %%
metrics = []
ks = range(1,20)
for k in ks:
  classifier = KNeighborsClassifier(n_neighbors=k,metric='euclidean')
  classifier.fit(X_train,Y_train)
  Y_pred = classifier.predict(X_test)
  metrics.append(accuracy_score(Y_test,Y_pred))
plt.figure(figsize=(7,7))
plt.title('k - accuracy plot')
plt.plot(ks,metrics)
plt.show() 

best_knn = KNeighborsClassifier(n_neighbors=4)
best_knn.fit(X_train,Y_train)
Y_pred = best_knn.predict(X_test)
accuracy_score(Y_test,Y_pred)


