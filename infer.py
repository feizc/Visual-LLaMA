import torch 
from llama import LlamaTokenizer, LlamaForCausalLM
from model import MultimodalLlamaLLM

ckpt_path = './ckpt'
special_tokens_dict = {'additional_special_tokens': ['[boi]','[eoi]']}
tokenizer = LlamaTokenizer.from_pretrained(ckpt_path)
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict) 

llama_model = LlamaForCausalLM.from_pretrained(ckpt_path) 
llama_model.resize_token_embeddings(len(tokenizer)) 
    
model = MultimodalLlamaLLM(image_length=10, llama=llama_model,)
model.load_state_dict(torch.load('out2.pt'))
model.eval()


input = '[boi] hellow [eoi]'
id = tokenizer.encode(input) 
print(id)
print(tokenizer.decode(id))
print(tokenizer.pad_token)

"""
prompt = "Hey, are you consciours? Can you talk to me?"
inputs = tokenizer(prompt, return_tensors="pt")
generate_ids = model.generate(inputs.input_ids, max_length=30)
print(tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])
"""

