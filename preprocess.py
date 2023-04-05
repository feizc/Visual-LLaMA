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


def coco_process(args, clip_model, processor): 

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



def vqav2_process(args, clip_model, processor): 
    with open(os.path.join(args.dataset_path, 'questions.json'), 'r') as f: 
        question_data = json.load(f)['questions'] 
    
    with open(os.path.join(args.dataset_path, 'annotations.json'), 'r') as f: 
        annotation_data = json.load(f)['annotations']
    
    all_embeddings = [] 
    all_texts = []
    for i in tqdm(range(len(question_data))): 
        img_id = question_data[i]['image_id']
        
        filename = os.path.join(args.dataset_path, f"scene_img_abstract_v002_train2017/abstract_v002_train2015_{int(img_id):012d}.png") 
        image = Image.open(filename)
        inputs = processor(images=image, return_tensors='pt') 

        with torch.no_grad():
            image_features = clip_model.get_image_features(**inputs) 

        text = '[quest] ' + question_data[i]['question'] + ' [ans] ' + annotation_data[i]['answers'][0]['answer']
        all_embeddings.append(image_features) 
        all_texts.append(text)

        if args.debug == True and i > 100: 
            break 
    
    with open('train.pkl', 'wb') as f: 
        pickle.dump({'image_embedding': torch.cat(all_embeddings, dim=0), 
                        'text': all_texts}, f) 



def main(): 
    parser = argparse.ArgumentParser() 
    parser.add_argument('--clip_path', default='./clip')
    parser.add_argument('--dataset_path', default='../COCO')
    parser.add_argument('--dataset_type', default='VQAV2') 
    parser.add_argument('--debug', default=True)
    args = parser.parse_args()

    clip_model = CLIPModel.from_pretrained(args.clip_path)
    processor = CLIPProcessor.from_pretrained(args.clip_path) 

    if args.dataset_type == 'COCO': 
        coco_process(args, clip_model, processor) 
    elif args.dataset_type == 'VQAV2':
        vqav2_process(args, clip_model, processor) 
    else:
        pass 


if __name__ == '__main__': 
    main()

