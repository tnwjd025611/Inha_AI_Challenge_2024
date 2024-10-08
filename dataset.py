import copy
import logging

from typing import Dict, Sequence
from dataclasses import dataclass
from torch.utils.data import Dataset
from tqdm import tqdm

import utils

import transformers
import torch

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = '[PAD]'
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
PROMPT = """ 
너는 금융 및 경제 AI야. 
아래의 배경지식을 참고해서 질문에 대한 답을 생성해줘.

배경 지식 : {context}

질문 : {question}

(####) 답변 :
"""

def smart_tokenizer_and_embedding_resize(
        special_tokens_dict: Dict,
        tokenizer: transformers.PreTrainedTokenizerFast,
        model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.
    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

def _tokenize_fn(
        strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizerFast
) -> Dict:
    """Tokenize a list of strings"""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors='pt',
            padding='longest',
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids = input_ids,
        labels = labels,
        input_ids_lens = input_ids_lens,
        labels_lens = labels_lens
    )

def preprocess(
        sources: Sequence[str],
        targets: Sequence[str],
        tokenizer: transformers.PreTrainedTokenizerFast,
) -> Dict:
    """Preprocess the data by tokenizing"""
    examples = [s+t for s,t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [
        _tokenize_fn(strings, tokenizer) for strings in (examples, sources)
    ]

    input_ids = examples_tokenized['input_ids']
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized['input_ids_lens']):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)

def get_prompt_format(example): #재작성 예장
    return PROMPT.format_map(example)

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning"""

    def __init__(self, data_path:str, tokenizer: transformers.PreTrainedTokenizerFast):
        super(SupervisedDataset, self).__init__()
        logging.warning('Loading data ...')
        sources = []
        targets = []
        
        list_data_dict = utils.load(data_path)

        for example in list_data_dict:
            p = get_prompt_format(example)
            sources.append(p)
        targets += [
            f'{example["answer"]}{tokenizer.eos_token}' for example in list_data_dict
        ]

        logging.warning('Tokenizing inputs ... This may take some time ...')
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict['input_ids']
        self.labels = data_dict['labels']
        
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids = self.input_ids[i], labels=self.labels[i])
        
@dataclass
class DataCollatorForSupervisedDataset(object):
    """collate exmaples for supervised fine-tuning"""
    tokenizer : transformers.PreTrainedTokenizerFast

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ['input_ids', 'labels']
        )

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first = True, padding_value = self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first = True, padding_value = IGNORE_INDEX
        )
        
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
    
def make_supervised_data_module(
        tokenizer : transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(
        tokenizer=tokenizer, data_path=data_args.data_path
    )

    # eval_dataset = SupervisedDataset(
    #     tokenizer=tokenizer, data_path=data_args.eval_path
    # )

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    return dict(
        train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
    )