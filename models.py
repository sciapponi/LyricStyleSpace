import torch
from torch import nn
from transformers import AutoModel
from torch import Tensor
from typing import Tuple, Iterator, Dict, Any, Sequence, Optional, Union

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
