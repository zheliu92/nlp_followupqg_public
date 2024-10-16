import json
from transformers import BartForConditionalGeneration, BartTokenizer, set_seed
import torch
import numpy as np

def generate_follow_up(question, answer, mode):
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        input_text = question + "<SEP>" + answer + "<QUS>"
        inputs = tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True)
        inputs = {key: val.to(device) for key, val in inputs.items()}
        
        if mode == 'topkp':
        # top-k & top-p sampling
          outputs = model.generate(**inputs, max_length=1024, num_return_sequences=10, num_beams=10, early_stopping=True,  do_sample=True, temperature=1.0, top_k=50, top_p=0.95)
        elif mode == 'beam':
        # beamsearch sampling
          outputs = model.generate(**inputs, max_length=1024, num_return_sequences=10, num_beams=10, early_stopping=True, do_sample=True, temperature=1.0)

        return tokenizer.batch_decode(outputs, skip_special_tokens=True)
    except Exception as e:
        print(f"Error: {e}")
        return "------------ Error in generating follow-up question ------------"


with open("test.json", "r") as f:
    data = json.load(f)

for model_version in ["org", "small", "full", 'gpt']: # "org", "small", "full", 'gpt'
  for mode in ['beam', 'topkp']:

    result_filename = model_version + f"_result_repeat_{mode}.json"
    model_name = f"/home/zheliu92/scratch/trained_{model_version}_model"

    seed = 42
    set_seed(seed)

    print("Loading Models")
    model = BartForConditionalGeneration.from_pretrained(model_name)
    tokenizer = BartTokenizer.from_pretrained(model_name)
    print("Models Loaded")

    res = []

    for i in range(len(data)):
    # for i in range(5):
      # Load task samples
      dict_with_follow_up = data[i]
      q = data[i]["question"]
      a = data[i]["answer"]
      id = data[i]["id"]

      generated_follow_up = generate_follow_up(q, a, mode)

      dict_with_follow_up["generated_follow_up"] = generated_follow_up
      res.append(dict_with_follow_up)

      print(f"Task {id} Completes")

      json_data = json.dumps(res, indent=2)
      with open(result_filename, "w") as f:
        f.write(json_data)