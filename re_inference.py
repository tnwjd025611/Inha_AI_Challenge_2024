import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"]= "0"  # Set the GPUs 

from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    PreTrainedTokenizerFast,
    HfArgumentParser,
    AutoTokenizer,
)
from peft import PeftModel
import torch
import utils
import pandas as pd
import numpy as np
import random

from dataclasses import dataclass, field
from typing import Optional

random_seed = 5959
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

@dataclass
class ScriptArguments:
    base_model_path : Optional[str] = field(default=None, metadata={'help': 'Please write model name'})
    model_path : Optional[str] = field(default=None, metadata={'help': 'Please write model name'})
    test_data_path : Optional[str] = field(default=None, metadata={'help': 'Test data path'})
    output_path : Optional[str] = field(default=None, metadata={'help': 'Output data path'})
    file_name : Optional[str] = field(default=None, metadata={'help': 'Output file name'})

TEMPLATE = [
    """당신은 세계적인 금융 기관에서 20년 이상의 경험을 가진 글로벌 금융 전문가입니다. 당신은 금융 시장의 최신 동향과 국제 경제 이슈를 깊이 이해하고 있으며, 복잡한 경제 데이터를 명확하게 해석할 수 있습니다.
주어진 신문 기사를 읽고, 아래 질문에 대해 상세하고 깊이 있는 답변을 작성하세요. 답변을 작성할 때는 다음 단계를 따르세요:

1.기사의 전반적인 내용과 주요 경제적 맥락을 이해합니다.
2.글로벌 금융 시장과 관련된 핵심 정보와 데이터를 식별합니다.
3.질문의 구체적인 요구 사항과 의도를 분석합니다.
4.기사의 정보와 질문 사이의 연관성을 분석하여 종합합니다.
5.글로벌 경제 동향과 맥락을 반영하여 명확하고 정확한 답변을 작성합니다.

배경 정보: {context}
질문: {question}

(####) 답변:
    """,
    """너는 경제경영학과 대학원생이야. 너는 수업의 과제로 금융 및 경제 관련 기사를 읽고 질문에 대한 답변을 해야해.
아래의 배경지식을 참고해서 질문에 대한 답을 생성해줘.

이때, 해당 단계를 거쳐서 답변을 생성해봐.
1. 내용을 전체적으로 파악: 기사의 전체적인 맥락과 주제를 이해해.
2. 주요 정보 식별: 기사에서 중요한 통계, 주장, 전문가 의견 등을 식별해.
3. 질문 분석: 질문을 주의 깊게 읽고, 핵심 요구사항을 파악해.
4. 질문과 정보의 연결고리 찾기: 질문에 답하기 위해 기사에서 관련된 정보를 찾아 연결해.
5. 정답 도출: 위 단계를 통해 얻은 정보를 바탕으로 명확하고 간결한 답변을 작성해.
해당 단계를 따라서 문제에 대한 답을 생각해줘.

배경 지식 : {context}
질문 : {question}

(####) 답변 :
""",
    """너는 경제경영학과 대학원생이야. 너는 수업의 과제로 금융 및 경제 관련 기사를 읽고 질문에 대한 답변을 해야해.
아래의 배경지식을 참고해서 질문에 대한 답을 생성해줘.

이때, 해당 단계를 거쳐서 답변을 생성해봐.
1. 내용을 전체적으로 파악
2. 주요 정보 식별
3. 질문 분석
4. 질문과 정보의 연결고리 찾기
5. 정답 도출
해당 단계를 따라서 문제에 대한 답을 생각해줘.

배경 지식 : {context}
질문 : {question}

(####) 답변 : 
""",
    ]
RE_TEMPLATE = """
당신은 금융 및 경제 분석 전문가입니다. 아래 신문 기사를 읽고, 제공된 배경 정보를 바탕으로 질문에 대한 적절한 답변을 선택하세요. 답변은 제공된 선택지 중에서만 골라야 합니다. 답변을 결정할 때는 다음 단계를 따르세요:

기사의 전반적인 내용과 핵심 정보를 이해합니다.
질문의 정확한 요구 사항과 의미를 파악합니다.
질문과 선택지 간의 관련성을 분석합니다.
선택지 중에서 가장 적절한 답변을 찾습니다.
선택한 답변이 질문에 가장 잘 부합하는 이유를 확인합니다.
이유가 적절하다면 해당 답변을 최종 답변으로 선택합니다.

배경 정보: {context}
질문: {question}

답변 선택지: {answer_list}

(####) 답변:
"""

def most_frequent(data):
    return max(data, key=data.count)

def do_generate(model, tokenizer, data_list):
    generated = []

    for data in tqdm(data_list): #수정해야함
        answer = []
        
        for template in TEMPLATE:
            formatted_question_texts = template.format(context=data['context'], question=data['question'])
    
            tokens = tokenizer(formatted_question_texts, return_tensors = 'pt')
            tokens = {key: tensor.to(model.device) for key, tensor in tokens.items()}
    
            outputs = model.generate(tokens['input_ids'], max_new_tokens=200)
            decoded_output = tokenizer.decode(outputs[0][tokens['input_ids'].size(1):], skip_special_tokens=True)
            answer.append(decoded_output)
            
        re_formatted_question_texts = RE_TEMPLATE.format(context=data['context'], question=data['question'], answer_list=answer)
        tokens = tokenizer(re_formatted_question_texts, return_tensors = 'pt')
        tokens = {key: tensor.to(model.device) for key, tensor in tokens.items()}

        outputs = model.generate(tokens['input_ids'], max_new_tokens=200)
        decoded_output = tokenizer.decode(outputs[0][tokens['input_ids'].size(1):], skip_special_tokens=True)
                
        if type(decoded_output) == float:
            print("got Nan")
            decoded_output = '.'
        generated.append(decoded_output)
            
    return generated

parser = HfArgumentParser(ScriptArguments)
args = parser.parse_args_into_dataclasses()[0]

try:
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    else:
        print("Output Path Exist!")
except OSError:
    print("Error: Failed to create the directory.")

tokenizer = PreTrainedTokenizerFast.from_pretrained(
    args.model_path,
    local_files_only=True,
    model_max_length = 2048,
    padding_side = 'right',
    use_fast= False,
    device_map='auto',
    return_token_type_ids=False,
)
tokenizer.add_special_tokens({'pad_token' : '[PAD]'})

model = AutoModelForCausalLM.from_pretrained(
    args.base_model_path,
    device_map='auto',
    torch_dtype=torch.float32,
)
model.resize_token_embeddings(len(tokenizer))
model = PeftModel.from_pretrained(model=model, model_id = args.model_path)

test_data_list = utils.test_load(args.test_data_path)
results = do_generate(model, tokenizer, test_data_list)

results_transformed = utils.to_pandas(args.test_data_path, results)
results_transformed.to_csv(f'{args.output_path}/{args.file_name}.csv', index=False)