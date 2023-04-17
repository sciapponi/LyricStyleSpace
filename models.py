import torch

# NN Model
class PrototypicalTransformerModel(nn.Module):
    
    def __init__(self,
                 model_ckpt: str,
                 output_dim: int) -> None:
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_ckpt)
        self.dense_layer = torch.nn.Linear(self.transformer.config.hidden_size, output_dim)

    def compute_prototypes(self, support: Tensor, label: Tensor) -> Tensor:
        """Set the current prototypes used for classification.

        Parameters
        ----------
        data : torch.Tensor
            Input encodings
        label : torch.Tensor
            Corresponding labels

        """
        means_dict: Dict[int, Any] = {}
        for i in range(support.size(0)):
            means_dict.setdefault(int(label[i]), []).append(support[i])

        means = []
        n_means = len(means_dict)

        for i in range(n_means):
            # Ensure that all contiguous indices are in the means dict
            supports = torch.stack(means_dict[i], dim=0)
            if supports.size(0) > 1:
                mean = supports.mean(0).squeeze(0)
            else:
                mean = supports.squeeze(0)
            means.append(mean)

        prototypes = torch.stack(means, dim=0)
        return prototypes
    def compute_embeddings(self,
                           query_input_ids : Tensor,
                           query_attention_mask : Tensor):
      query_embedding = self.transformer(input_ids = query_input_ids, attention_mask = query_attention_mask)
      # return torch.nn.functional.relu(self.dense_layer(query_embedding["pooler_output"]))
      return self.dense_layer(query_embedding["pooler_output"])
      # return torch.nn.functional.softmax(self.dense_layer(query_embedding["pooler_output"]))
      
    def forward(self,  # type: ignore
                query_input_ids: Tensor,
                query_attention_mask: Tensor,
                support_input_ids: Optional[Tensor] = None,
                support_attention_mask: Optional[Tensor] = None,
                support_label: Optional[Tensor] = None,
                prototypes: Optional[Tensor] = None) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Run a forward pass through the network.
        
        Parameters
        ----------
        query: Tensor
            The query examples, as tensor of shape (seq_len x batch_size)
        support: Tensor
            The support examples, as tensor of shape (seq_len x batch_size)
        support_label: Tensor
            The support labels, as tensor of shape (batch_size)

        Returns
        -------
        Tensor
            If query labels are

        """

        query_encoding = self.compute_embeddings(query_input_ids=query_input_ids,query_attention_mask=query_attention_mask)
        
        if prototypes is not None:
            prototypes = prototypes
        elif support_input_ids is not None and support_attention_mask is not None and support_label is not None:
            support_encoding = self.compute_embeddings(query_input_ids=support_input_ids, query_attention_mask=support_attention_mask)
            prototypes = self.compute_prototypes(support_encoding, support_label)
        else:
          raise ValueError("No prototypes set or support vectors have been provided")

        dist = self.__euclidean_distance__(query_encoding, prototypes)
        clear_cache()

        return - dist
    
    def __euclidean_distance__(self, mat_1: Tensor, mat_2: Tensor):
        _dist = [torch.sum((mat_1 - mat_2[i])**2, dim=1) for i in range(mat_2.size(0))]
        dist = torch.stack(_dist, dim=1)
        return dist





#Metric Learning
class MetricTransformerModel(torch.nn.Module):
    def __init__(self,
                 model_name:str, 
                 hidden_dims:Tuple[int]=(128,)):
        super(TransformerModel, self).__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        self.dense_layers = torch.nn.ModuleList()
        last_hidden_size = self.transformer.config.hidden_size
        for dim in hidden_dims: 
          self.dense_layers.append(torch.nn.Linear(last_hidden_size, dim))
          last_hidden_size = dim

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids, attention_mask)
        output = outputs["pooler_output"]
        for layer in self.dense_layers: 
          output = torch.nn.functional.relu(layer(output))
        return output

def train(model, loss_func, mining_func, device, train_loader, optimizer, epoch):
    model.train()
    # Train your model
    for batch_idx,batch in enumerate(tqdm(train_loader)):
        # Extract the input ids and attention masks from the batch
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Encode the inputs using the pre-trained model
        embeddings = model(input_ids=input_ids, attention_mask=attention_mask)
        # print(embeddings)
        indices_tuple = mining_func(embeddings, labels)
        loss = loss_func(embeddings, labels, indices_tuple)
        loss.backward()
        optimizer.step()
        if batch_idx % 20 == 0:
            print(
                "\tEpoch {} Iteration {}:  Number of mined triplets = {}".format(
                    epoch, batch_idx, mining_func.num_triplets
                )
            )

    # Print the loss every epoch
    # print('\tEpoch [{}/{}], Loss: {}'.format(epoch, epochs, loss.item()))

def get_all_embeddings(dataloader, model):
  model.eval()
  embeddings, labels = [], []
  with torch.no_grad():
    for idx, batch in enumerate(tqdm(dataloader)):
      input_ids, attention_mask, label = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)
      embeddings.append(model(input_ids=input_ids, attention_mask=attention_mask))
      labels.append(label)

  return torch.vstack(embeddings), torch.cat(labels)

def test(train_loader, test_loader, model, accuracy_calculator):
  train_embeddings, train_labels = get_all_embeddings(train_loader, model)
  test_embeddings, test_labels = get_all_embeddings(test_loader, model)
  accuracies = accuracy_calculator.get_accuracy(test_embeddings, test_labels, train_embeddings, train_labels)

  print(f"Test set accuracy (Precision@1) = {accuracies['precision_at_1']}")


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