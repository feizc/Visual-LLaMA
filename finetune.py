import os 
import torch.nn as nn 
import argparse 
import torch 
from torch.utils.data import DataLoader
from torch import optim 
import tqdm 
import numpy as np 

from torch.cuda.amp import autocast
import peft 
import loralib as lora 
from llama import LlamaTokenizer, LlamaForCausalLM
from peft import (
    prepare_model_for_int8_training,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
)
from peft.tuners import lora
from utils import ImageTextDataSet
from model import MultimodalLlamaLLM 


special_tokens_dict = {'additional_special_tokens': ['[boi]','[eoi]', '[quest]', '[ans]']}


def parse_args():
    parser = argparse.ArgumentParser(description="Instructing tuning a multimodal llama model")
    parser.add_argument(
        "--train_file", type=str, default='train.pkl', help="A pkl file containing the training data."
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default='./ckpt',
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--resume_path",
        type=str,
        default='out.pt',
        help="Path to ckpt.",
    )
    parser.add_argument("--output_dir", type=str, default='out', help="Where to store the final model.")
    parser.add_argument(
        "--image_length",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=10, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--lr",
        type=float,
        default=4e-3,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    args = parser.parse_args()
    return args 


def save_tunable_parameters(model, path):
    saved_params = {
        k: v.to("cpu")
        for k, v in model.named_parameters()
        if v.requires_grad
    }
    torch.save(saved_params, path)


def main(): 
    args = parse_args()
    print(args) 
    device = 'cuda'

    tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path)
    num_added_tokens = tokenizer.add_special_tokens(special_tokens_dict) 
    llama_model = LlamaForCausalLM.from_pretrained(args.model_name_or_path) 
    model = MultimodalLlamaLLM(image_length=args.image_length, llama=llama_model,) 

    if args.resume_path is not None:
        model.llm.resize_token_embeddings(len(tokenizer) - 2) 
        model.load_state_dict(torch.load(args.resume_path)) 
    
    model.llm.resize_token_embeddings(len(tokenizer)) 

    config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=['q_proj', 'v_proj'],
        bias='none',
        task_type='CAUSAL_LM' 
    )
    
    model.llm = get_peft_model(model.llm, config) 

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    trainable_params = sum([np.prod(p.size()) for p in model_parameters])

    model_parameters = filter(lambda p: not p.requires_grad, model.parameters())
    non_trainable_params = sum([np.prod(p.size()) for p in model_parameters]) 

    print('trainable_params:{} ({:.2f}%)'.format(trainable_params, trainable_params/non_trainable_params*100,))

    train_dataset = ImageTextDataSet(args.train_file, tokenizer=tokenizer, image_length=args.image_length)
    train_loader = DataLoader(train_dataset, batch_size=args.per_device_train_batch_size) 

    optimizer = torch.optim.AdamW(model.llm.parameters(), lr=args.lr)

    model.to(device)
    model.llm.train() 
    print(model.llm.model.model)
    for param in model.llm.model.model.named_parameters():
        print(param[0])
    
    for epoch in range(args.num_train_epochs):
        total_loss = 0
        for step, batch in enumerate(t:=tqdm.tqdm(train_loader)):
            image_embedding, tokens, mask = batch 
            image_embedding, tokens, mask = image_embedding.to(device), tokens.to(device), mask.to(device) 
            outputs = model(tokens=tokens, labels=tokens, image_embedding=image_embedding, mask=mask) 
            loss_d = outputs.loss.detach().float()
            t.set_description(f"loss: {loss_d}")
            total_loss += loss_d
            loss = outputs.loss / args.gradient_accumulation_steps
            loss.backward()
            if (step+1) % args.gradient_accumulation_steps == 0:
                optimizer.step() 
                optimizer.zero_grad()
        print('save modeling')
        save_tunable_parameters(model, "adapter_model.bin")

if __name__ == "__main__":
    main()

