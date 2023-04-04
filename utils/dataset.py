import pickle 
import torch 
from torch.utils.data import Dataset


CUTOFF_LEN = 256


def tokenize(obj,tokenizer):
    if isinstance(obj, str):
        return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
    if isinstance(obj, dict):
        return dict((n, tokenize(o)) for n, o in obj.items())
    return list(tokenize(o) for o in obj) 


class ImageTextDataSet(Dataset):
    def __init__(self, file_path, tokenizer, image_length, image_normalize=False):
        super().__init__()
        self.image_length = image_length 
        self.image_normalize = image_normalize 
        self.tokenizer = tokenizer 

        with open(file_path, 'rb') as f: 
            all_data = pickle.load(f) 
        
        self.image_embeddings = all_data['image_embedding']
        self.texts = all_data['text'] 

    
    def __len__(self): 
        return len(self.texts) 


    def pad_tokens(self, tokens): 
        padding = CUTOFF_LEN - tokens.shape[0] 
        if padding > 0:
            tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            tokens = tokens[:CUTOFF_LEN] 
        mask = tokens.ge(0)  # mask is zero where we out of sequence
        tokens[~mask] = 0
        mask = mask.float()
        mask = torch.cat((torch.ones(self.image_length), mask), dim=0)  # adding image mask and [boi], [eoi]
        return tokens, mask


    def __getitem__(self, index):
        image_embedding = self.image_embeddings[index] 
        tokens = torch.tensor(self.tokenizer.encode('[eoi]' + self.texts[index]), dtype=torch.int64) 
        tokens, mask = self.pad_tokens(tokens=tokens) 

        if self.image_normalize == True: 
            image_embedding = image_embedding.float() 
            image_embedding = image_embedding / image_embedding.norm(2, -1)

        return image_embedding, tokens, mask 

        
