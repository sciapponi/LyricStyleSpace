import re
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
from torch.utils.data import Dataset


# Dataset Wrapper
class MyDataset(Dataset):
    def __init__(self,dataset) -> None:
        super().__init__()
        self.dataset = dataset
    def __getitem__(self, index):
        index = int(index)
        return (self.dataset[index]['input_ids'],self.dataset[index]['attention_mask']),self.dataset[index]['labels']
    def __len__(self):
        return len(self.dataset)

# Clearning Data With Regexps
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

# Lyrics Loading Function
def load(data_path):
	dataset = load_dataset('json',data_files=data_path).class_encode_column('artist').filter(lambda x: len(x['lyrics'])>0)
	artists_mappings = dataset['train'].features['artist'].names
	mapped_dataset = dataset.map(clean)

	tts_mapped_dataset = mapped_dataset['train'].train_test_split(train_size=0.7,stratify_by_column='artist')
	tvs_mapped_dataset = tts_mapped_dataset['test'].train_test_split(train_size=0.5,stratify_by_column='artist')

	return DatasetDict({'train': tts_mapped_dataset['train'],
    									'	test':tvs_mapped_dataset['test'],
    										'val': tvs_mapped_dataset['train']})

def prepare_train_features(tokenizer, examples):

    tokenized_examples = tokenizer(
        examples['lyrics'],
        truncation=True,
        padding=True,
        max_length=512
        )
    tokenized_examples['labels'] = examples['artist']
    return tokenized_examples



def get_features(dataset, model_ckpt):
	tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
	train_features = dataset['train'].map(lambda x: prepare_train_features(tokenizer, x), batched=True, remove_columns=dataset["train"].column_names).with_format('torch')
	test_features = dataset['test'].map(lambda x: prepare_train_features(tokenizer, x), batched=True, remove_columns=dataset["test"].column_names).with_format('torch')
	val_features = dataset['val'].map(lambda x: prepare_train_features(tokenizer, x), batched=True, remove_columns=dataset["val"].column_names).with_format('torch')

	return train_features, val_features, test_features