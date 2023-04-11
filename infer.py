import torch 
from llama import LlamaTokenizer, LlamaForCausalLM
from model import MultimodalLlamaLLM
from transformers import CLIPProcessor, CLIPModel 
from PIL import Image 
import torch.nn.functional as nnf
from tqdm import trange 

device = 'cuda'


def sampling_generate(model, tokenizer, embed=None, tokens=None, prompt=None): 
    # decode hyperparameters
    entry_count = 1
    entry_length = 50
    top_p = 0.8
    temperature = 1.0
    
    generated_num = 0
    generated_list = [] 
    stop_token_index = tokenizer.pad_token 
    filter_value = -float("Inf") 

    with torch.no_grad(): 
        for entry_idx in trange(entry_count): 
            if embed is not None:
                generated = embed 
            else:
                if tokens is None: 
                    tokens = torch.tensor(tokenizer.encode(prompt))
                    tokens = tokens.unsqueeze(0).to(device)
                generated = model.llm.model.embed_tokens(tokens) 

        for i in range(entry_length): 
            outputs = model.llm(inputs_embeds=generated)
            logits = outputs.logits
            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(nnf.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                                                ..., :-1
                                                ].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[:, indices_to_remove] = filter_value
            next_token = torch.argmax(logits, -1).unsqueeze(0)
            next_token_embed = model.llm.model.embed_tokens(next_token)
            if tokens is None:
                tokens = next_token
            else:
                tokens = torch.cat((tokens, next_token), dim=1)
            generated = torch.cat((generated, next_token_embed), dim=1)
            if stop_token_index == next_token.item():
                break

        output_list = list(tokens.squeeze().cpu().numpy())
        output_text = tokenizer.decode(output_list)
        generated_list.append(output_text)

    return generated_list[0]





def inference(model, tokenizer, clip_model, clip_processor, image=None): 

    if image is not None:
        clip_inputs = clip_processor(images=image, return_tensors='pt') 

        with torch.no_grad():
            clip_inputs['pixel_values'] = clip_inputs['pixel_values'].to(device)
            image_features = clip_model.get_image_features(**clip_inputs) 
            image_embed = model.image_project(image_features).reshape(1, model.image_length, -1) 
    else:
        image_embed = None 
    generated_text = sampling_generate(model, tokenizer, embed=image_embed, tokens=None, prompt='[eoi]') 
    return generated_text 



ckpt_path = './ckpt' 
clip_path = './clip' 
image_path = 'case.png'

special_tokens_dict = {'additional_special_tokens': ['[boi]','[eoi]']}
tokenizer = LlamaTokenizer.from_pretrained(ckpt_path)
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict) 
tokenizer.pad_token = 0

llama_model = LlamaForCausalLM.from_pretrained(ckpt_path) 
llama_model.resize_token_embeddings(len(tokenizer)) 
    
model = MultimodalLlamaLLM(image_length=10, llama=llama_model,)
model.load_state_dict(torch.load('out.pt'))
model = model.to(device)
model.eval()

clip_model = CLIPModel.from_pretrained(clip_path)
clip_processor = CLIPProcessor.from_pretrained(clip_path) 
clip_model.to(device)

image = None 
if image_path is not None: 
    image = Image.open(image_path) 

generated_text = inference(model, tokenizer, clip_model, clip_processor, image)
print(generated_text)
