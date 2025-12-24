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

train_dataset = load_dataset('GEM/viggo', split='train')
eval_dataset = load_dataset('GEM/viggo', split='validation')
test_dataset = load_dataset('GEM/viggo', split='test')

print(test_dataset[0])