"""
    This file provides a case for multi-modal dataset construction. 
"""
import os 
import torch 
from transformers import CLIPProcessor, CLIPModel 
from PIL import Image 
import pickle
import argparse  
import json 
from tqdm import tqdm 



def main(): 
    parser = argparse.ArgumentParser() 
    parser.add_argument('--clip_path', default='./clip')
    parser.add_argument('--dataset_path', default='../COCO')
    parser.add_argument('--debug', default=True)
    args = parser.parse_args()

    clip_model = CLIPModel.from_pretrained(args.clip_path)
    processor = CLIPProcessor.from_pretrained(args.clip_path) 

    with open(os.path.join(args.dataset_path, 'annotations/captions_train2014.json'), 'r') as f: 
        data = json.load(f)['annotations']
    
    all_embeddings = [] 
    all_texts = []
    for i in tqdm(range(len(data))): 
        d = data[i] 
        img_id = d['image_id']  
        filename = os.path.join(args.dataset_path, f"train2014/COCO_train2014_{int(img_id):012d}.jpg")

        image = Image.open(filename)
        inputs = processor(images=image, return_tensors='pt') 

        with torch.no_grad():
            image_features = clip_model.get_image_features(**inputs) 
        all_embeddings.append(image_features) 
        all_texts.append(d['caption']) 
        if args.debug == True and i > 10:
            break 
    
    with open('train.pkl', 'wb') as f: 
        pickle.dump({'image_embedding': torch.cat(all_embeddings, dim=0), 
                        'text': all_texts}, f)


if __name__ == '__main__': 
    main()

