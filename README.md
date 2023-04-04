<p align="center">
     <img src="figures/logo.png" alt="logo" width = "600">
     <br/>
</p>



## Open LLaMA Eyes to See the World

This project aims to optimize LLaMA model for visual information understanding and further explore the potentional of large language model. 

Generally, we use CLIP vision encoder to extract image features, then image features are projected with MLP or Transformer into text embedding dimensionality. Then, visual representation (including special tokens [boi] and [eoi]) is concatenated with text representation to learn in a autoregressive manner. The framework is similar to [kosmos-1](https://arxiv.org/pdf/2302.14045.pdf) and [PaLM-E](https://palm-e.github.io/).


- [X] Code adjustation support for multi-modal generation. Download [clip](https://huggingface.co/openai/clip-vit-large-patch14) and [LLaMA](https://huggingface.co/decapoda-research/llama-7b-hf) models from huggingface. 

- [X] Supervised training stage: freeze llama and clip-encoder models and only optimize the connection network. In this stage, we use COCO, CC-3M, and YOLO-700M datasets. 


- [ ] Instructing tuning stage: fine-tuning full model with VQA and language-only instructing dataset. 





