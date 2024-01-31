from tqdm import tqdm
import torch
from sklearn.metrics import accuracy_score
import os

def train(model,train_loader,val_loader,optimizer,loss,epoches,device):
    print(device)
    #model.train()

    model = model.to(device)

    true_labels,pred_labels=[],[]
    train_acc,val_acc=[],[]
    train_loss,val_loss=[],[]
    best_acc = 0
    for epoch in range(epoches):
        model.train()
        for i, batch in tqdm(enumerate(train_loader), total=len(train_loader), desc="Training"):
            optimizer.zero_grad()
            guids,texts,texts_mask,images,labels=batch
            texts,texts_mask,images,labels = texts.to(device),texts_mask.to(device),images.to(device),labels.to(device)
            pred=model(texts,texts_mask,images)

            losses=loss(pred,labels)
            #print(losses)
            pred = torch.argmax(pred, dim=1)
            pred_labels.extend(pred.tolist())
            losses.backward()
            optimizer.step()
            with torch.no_grad():
                train_loss.append(losses.item())
                true_labels.extend(labels.tolist())
        val_l,val_accuracy=val(model,val_loader,loss,device)
        train_accuracy = accuracy_score(true_labels, pred_labels)
        print("train_acc",train_accuracy)
        val_loss.extend(val_l)
        val_acc.append(val_accuracy)
        train_acc.append(train_accuracy)

        if(val_accuracy>best_acc):
            best_acc = val_accuracy
            #保存当前在验证集上表现最好的模型
            torch.save(model, './trained_model/best_model.pt')

    return train_loss,train_acc,val_loss,val_acc


def val(model,val_loader,loss,device):
    print(device)
    model.eval()
    val_loss=[]
    model = model.to(device)
    true_labels,pred_labels=[],[]
    for batch in tqdm(val_loader,desc='----[validating]'):
        guids,texts,texts_mask,images,labels=batch
        texts,texts_mask,images,labels = texts.to(device),texts_mask.to(device),images.to(device),labels.to(device)
        pred=model(texts,texts_mask,images)
        losses=loss(pred,labels)
        val_loss.append(losses.item())
        true_labels.extend(labels.tolist())
        pred_label = torch.argmax(pred, dim=1)
        pred_labels.extend(pred_label.tolist())

    val_accuracy = accuracy_score(true_labels, pred_labels)
    print("val_acc",val_accuracy)

    return val_loss,val_accuracy


def predict(model,test_loader,device):
    model.eval()
    pred_guids, pred_labels = [], []

    for batch in tqdm(test_loader, desc='----- [Predicting] '):
        guids, texts, texts_mask, imgs, labels = batch
        texts, texts_mask, imgs = texts.to(device), texts_mask.to(device), imgs.to(device)
        pred_voc = model(texts, texts_mask, imgs)
        pred = torch.argmax(pred_voc, dim=1)

        pred_guids.extend(guids)
        pred_labels.extend(pred.tolist())

    return [(guid, label) for guid, label in zip(pred_guids, pred_labels)]


def test(model, test_loader, output_path, device, model_path='./trained_model/best_model.pt'):
    # 读取保存的在验证集上表现最好的模型进行预测
    model = torch.load(model_path).to(device) 
    outputs = predict(model, test_loader, device=device)

    formated_outputs = ['guid,tag']
    for guid, label in tqdm(outputs, desc='----- [Decoding]'):
        formated_outputs.append((str(guid) + ',' + map_index_to_label(label)))


    if not os.path.exists(output_path): os.makedirs(output_path)

    with open(output_path, 'w') as f:
        for line in tqdm(formated_outputs, desc='----- [Writing]'):
            f.write(line)
            f.write('\n')

def map_index_to_label(label):
    label_mapping = {0:'positive', 1:'neutral', 2:'negative',3:'null'}
    # 使用get方法，如果label不存在于映射中，返回默认值3
    sentiment = label_mapping.get(label, 'null')
    return sentiment

