<p align="center">
     <img src="figures/logo.png" alt="logo" width = "600">
     <br/>
</p>



## Open LLaMA Eyes to See the World

This project aims to optimize LLaMA model for visual information understanding like GPT-4 and further explore the potentional of large language model. 

Generally, we use CLIP vision encoder to extract image features, then image features are projected with MLP-based or Transformer-based connection network into text embedding dimensionality. Then, visual representation (including additional special tokens [boi] and [eoi]) is concatenated with text representation to learn in a autoregressive manner. The framework is similar to [kosmos-1](https://arxiv.org/pdf/2302.14045.pdf) and [PaLM-E](https://palm-e.github.io/).


- [X] Code adjustation to support for multi-modal generation. Download [clip](https://huggingface.co/openai/clip-vit-large-patch14) and [LLaMA](https://huggingface.co/decapoda-research/llama-7b-hf) models from huggingface. Meantime, we test the scripts are also compatible with other LLaMA model size. Please use script ```preprocess.py``` to deal with the data.  

- [X] Supervised training stage: freeze llama and clip-encoder models and only optimize the connection network. In this stage, we use COCO, CC-3M and COYO-700M datasets with training scripts ```train.py```. 
     We provide the training hyper-parameter used in our experiemnts on A100 GPU(80G).  We also evaluate the image captioning performance in COCO testing set. 
       
     | Argument | Values |
     |------|------|
     | `batch size` | 1 * 8 * 8 |
     | `epochs` | 3 |
     | `cut length` | 256 |
     | `learning rate` | 4e-3 |
     | `image sequence length` | 10 |



- [X] Instructing tuning stage: fine-tuning full model with mixed VQA and language-only instructing dataset. We use lora strategy to optimize the entire model with fine-tuning scripts ```finetune.py```. 

     | Argument | Values |
     |------|------|
     | `batch size` | 1024 |
     | `epochs` | 3 |
     | `cut length` | 256 |
     | `learning rate` | 2e-5 |
     | `image sequence length` | 10 |


- [ ] Open source trained ckpt on huggingface and gradio interface for multi-model generation. 


## Reference 

[1] https://github.com/facebookresearch/llama 

[2] https://github.com/tloen/alpaca-lora




