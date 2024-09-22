# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"]= "0,1,2,3"  # Set the GPUs 

os.environ["WANDB_PROJECT"]="dacon-train"
os.environ["WANDB_LOG_MODEL"]="checkpoint"
os.environ["WANDB_WATCH"]="false"

import tempfile
from dataclasses import dataclass, field
from typing import Optional
import torch
import transformers
import logging
import wandb

import dataset

from tqdm import tqdm
from accelerate import Accelerator
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    BitsAndBytesConfig,
    PreTrainedTokenizerFast,
)
from peft import LoraConfig
from trl import SFTTrainer

@dataclass
class ScriptArguments:
    model_name : Optional[str] = field(default=None, metadata={'help': 'Please write model name'})
    data_path: Optional[str] = field(default=None, metadata={'help': 'train data path'})
    # eval_path: Optional[list[str]] = field(default=None, metadata={'help': 'Eval data path'})
    output_path : Optional[str] = field(default='./', metadata={'help': 'output data path'})
    use_8_bit : Optional[bool] = field(default=False, metadata={'help': 'use 8 bit precision'})
    use_4_bit : Optional[bool] = field(default=False, metadata={'help': 'use 4 bit precision'})
    bnb_4bit_quant_type: Optional[str] = field(default="nf4", metadata={"help": "precise the quantization type (fp4 or nf4)"})
    use_bnb_nested_quant: Optional[bool] = field(default=False, metadata={"help": "use nested quantization"})
    use_multi_gpu : Optional[bool] = field(default=False, metadata={'help': 'use multi GPU'})
    use_adapters : Optional[bool] = field(default=True, metadata={'help': 'use adapters'})
    batch_size : Optional[int] = field(default=1, metadata={'help': 'input batch size'})
    max_seq_length: Optional[int] = field(default= 2048, metadata={'help': 'max sequence length'})
    optimizer : Optional[str] = field(default='adamw_torch', metadata={'help': 'Optimizer name'})
    epochs : Optional[int] = field(default=3, metadata={'help': 'Epoch'})

def get_current_device():
    return Accelerator().process_index

def list_of_strings(arg):
    return arg.split(',')

parser = HfArgumentParser(ScriptArguments)
args = parser.parse_args_into_dataclasses()[0]

tqdm.pandas()

if args.use_multi_gpu:
    device_map='auto'
    print('Multi Gpu Auto Mode')
else:
    device_map: {'':get_current_device()}

if args.use_8_bit and args.use_4_bit:
    raise ValueError(
        "You can't use 8 bit and 4 bit precision at the same time"
    )

if args.use_4_bit:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=args.use_bnb_nested_quant,
    )
else:
    bnb_config = None

if args.use_adapters:
    peft_config = LoraConfig(
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type='CAUSAL_LM',
    )
else:
    peft_config = None
    if args.use_8_bit:
        raise ValueError(
            'You need to use adapters to use 8 bit precision'
        )

tokenizer = PreTrainedTokenizerFast.from_pretrained(
    args.model_name,
    # local_files_only=True,
    model_max_length = args.max_seq_length,
    padding_side = 'right',
    use_fast = False,
    token = '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',
)

tokenizer.add_special_tokens({'pad_token': '[PAD]'})

model = AutoModelForCausalLM.from_pretrained(
    args.model_name,
    load_in_8bit=args.use_8_bit,
    load_in_4bit=args.use_4_bit,
    device_map=device_map if (args.use_8_bit or args.use_4_bit) else 'auto',
    quantization_config=bnb_config,
    torch_dtype=torch.float32,
    # torch_dtype=torch.float16,
    # local_files_only=True,
    token = 'YOUR_TOKEN',
)

model.resize_token_embeddings(len(tokenizer))

data_module = dataset.make_supervised_data_module(tokenizer=tokenizer, data_args=args)

steps = int(args.max_seq_length / args.batch_size * args.epochs)

training_arguments = TrainingArguments(
    per_device_train_batch_size=args.batch_size,
    num_train_epochs=args.epochs,
    gradient_accumulation_steps=4,
    output_dir = args.output_path,
    report_to='wandb',
    optim=args.optimizer,
    fp16=True,
    logging_strategy='epoch'
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    **data_module,
    peft_config=peft_config,
    max_seq_length=args.max_seq_length,
    args=training_arguments
)

trainer.train(resume_from_checkpoint=True) #resume_from_checkpoint=True
trainer.save_model(args.output_path)

output_files = os.listdir(args.output_path)
logging.info(f"Output files: {output_files}")

# assert "adapter_model.bin" in os.listdir(args.output_path)
