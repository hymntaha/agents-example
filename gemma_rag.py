import torch 
import transformers
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from peft import prepare_model_for_kbit_training, LoraConfig, PeftModel, get_peft_model

model_id = "mistralai/Mistral-7B-Instruct-v0.2"
device = "cuda"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Loading the model and tokenizer

model = AutoModelForCausalLM.from_pretrained(model_id,quantization_config=bnb_config)
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    model_max_length=512,
    padding_side="left",
    add_eos_token=True)
tokenizer.pad_token = tokenizer.eos_token

train_dataset = load_dataset('gem/viggo', split='train', trust_remote_code=True)
eval_dataset = load_dataset('gem/viggo', split='validation', trust_remote_code=True)
test_dataset = load_dataset('gem/viggo', split='test', trust_remote_code=True)

print(test_dataset)

def tokenize(prompt):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=512,
        padding="max_length",
    )
    result["labels"] = result["input_ids"].copy()
    return result

def generate_and_tokenize_prompt(data_point):
    full_prompt =f"""Given a target sentence construct the underlying meaning representation of the input sentence as a single function with attributes and attribute values.
This function should describe the target string accurately and the function must be one of the following ['inform', 'request', 'give_opinion', 'confirm', 'verify_attribute', 'suggest', 'request_explanation', 'recommend', 'request_attribute'].
The attributes must be one of the following: ['name', 'exp_release_date', 'release_year', 'developer', 'esrb', 'rating', 'genres', 'player_perspective', 'has_multiplayer', 'platforms', 'available_on_steam', 'has_linux_release', 'has_mac_release', 'specifier']


### Target sentence:
{data_point["target"]}


### Meaning representation:
{data_point["meaning_representation"]}
"""
    return tokenize(full_prompt)