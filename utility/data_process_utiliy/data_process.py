
from .img_change import img_transform
from transformers import AutoTokenizer  #还有其他与模型相关的tokenizer，如BertTokenizer
import os
from PIL import Image
from tqdm import tqdm
from .dataset import CustomerDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

def map_label_to_index(label):
    label_mapping = {'positive': 0, 'neutral': 1, 'negative': 2, 'null': 3}
    
    # 使用get方法，如果label不存在于映射中，返回默认值3
    index = label_mapping.get(label, 4)
    
    return index

def read_txt_file(file_path):
    try:
        with open(file_path, 'r',encoding='utf-8') as file:
            lines = file.read()
    except:
        try:
            with open(file_path,'r',encoding = 'gbk') as file:
                lines = file.read()
        except:
            with open(file_path,'r',encoding = 'iso-8859-1') as file:
                lines = file.read()
    return lines



def read_process(config,file_path,data_path,single_type=None):
    guids,texts,images,labels=[],[],[],[]
    tokenizer=AutoTokenizer.from_pretrained(config.bert_name) 
    print("")
    with open(file_path) as f:
        for line in tqdm(f.readlines(), desc='----- [reading texts and image]'):
            
            guid, label = line.replace('\n', '').split(',')
            
            text_path = os.path.join(data_path, f'{guid}.txt')
            image_path = os.path.join(data_path, f'{guid}.jpg')
            
            if guid == 'guid': continue
                
            if single_type == 'img':
                text=''
            else:
                raw_text = read_txt_file(text_path)
                text = raw_text.strip('\n').strip('\r').strip(' ').strip()
            text = tokenizer.tokenize('[CLS]' + text + 'SEP')
            text2integer = tokenizer.convert_tokens_to_ids(text)
            
            if single_type == 'text':
                img = Image.new(mode='RGB', size=(224, 224), color=(0, 0, 0))
                image_tra=img_transform(img,config.image_size)
                
            else:
                with Image.open(image_path) as img:
                    # 应用图像转换
                    image_tra=img_transform(img,config.image_size)

            guids.append(guid)
            labels.append(map_label_to_index(label))
            texts.append(text2integer)
            images.append(image_tra)
            
    return guids,texts,images,labels


def get_loader(guids, texts, images, labels,split=True):
    
    if split == False:
        test_dataset = CustomerDataset(guids,texts,images,labels)
        return DataLoader(dataset=test_dataset, batch_size=32,shuffle=False, collate_fn=test_dataset.collate_fn)
    
    # 假设 data 是一个包含四个部分的元组 (guids, texts, images, labels)
    data = (guids, texts, images, labels)
    # 使用 zip 合并四个部分成一个可迭代对象
    combined_data = list(zip(*data))
    # 使用 train_test_split 对 combined_data 进行切分
    train_combined, val_combined = train_test_split(combined_data, train_size=0.8, test_size=0.2, random_state=42)
    

    # 使用 zip(*result) 恢复切分后的结果为四个独立的元组
    train_data = tuple(zip(*train_combined))
    val_data = tuple(zip(*val_combined))

    train_dataset=CustomerDataset(*train_data)
    val_dataset=CustomerDataset(*val_data)

    #loa=DataLoader(dataset=train_data, batch_size=16,shuffle=True)
    train_loader=DataLoader(dataset=train_dataset, batch_size=16,shuffle=True, collate_fn=train_dataset.collate_fn)
    val_loader=DataLoader(dataset=val_dataset, batch_size=16,shuffle=False, collate_fn=val_dataset.collate_fn)

    return train_loader,val_loader

