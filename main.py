from utility import *
from utility.data_process_utiliy.data_process import read_process,get_loader
from utility.train_test_predict import train,test
from utility.draw.draw import plot_loss_accuracy
from model.crossmodel import Model
import argparse
import torch
from config import config
import torch.nn as nn
import torch.optim as optim

def dlt(args):
    lr = args.lr
    dropout = args.dropout
    epoch = args.epoch
    use_trained = bool(args.use_trained)
    weight_decay = args.weight_decay
    model_path = args.model_path
    device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # file path
    root_path = 'data/data'  # Change this to your actual data folder
    train_file_path = 'data/train.txt'
    test_file_path = 'data/test_without_label.txt'

    test_loader = get_loader(*read_process(config,test_file_path,root_path),split=False)

    if use_trained:
        """
        model path:存放model 参数的路径文件
        output_path:存放最终的预测文件
        """
        test(None,test_loader=test_loader,output_path="./predict/result.txt",model_path=model_path,device=device)
        return

    train_loader,val_loader= get_loader(*read_process(config,train_file_path,root_path))

    #get model
    crossattention_model = Model(config)

    #train model
    # 定义损失函数
    fun_loss = nn.CrossEntropyLoss()
    # 定义优化器
    optimizer = optim.AdamW(crossattention_model.parameters(), lr=lr,weight_decay=weight_decay)
    

    train_loss,train_acc,val_loss,val_acc=train(crossattention_model,train_loader,val_loader,optimizer,fun_loss,epoch,device)
    plot_loss_accuracy(train_loss, train_acc,[], val_acc)

    #predicet
    test(crossattention_model,test_loader=test_loader,output_path="./predict",device=device)

    # 打印选择的模型和超参数
    print(f"Learning Rate: {lr}")
    print(f"Dropout: {dropout}")
    print(f"epoch:{epoch}")





if __name__ == "__main__":
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description='Model Training with Hyperparameters')

    # 添加命令行参数
    parser = argparse.ArgumentParser()
    #模型option 选0仅用图片训练，1仅用文本训练，2为图片和文本的双模态融合模型
    parser.add_argument('--option', type=int, default=2, help='0-only image 1-only text 2-fusion') 
    parser.add_argument('--lr', type=float, default=5e-6, help='Learning rate')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--weight_decay',type=float, default=1e-4, help='weigth decay')
    parser.add_argument('--epoch', type=int, default=10, help='epoch times')
    parser.add_argument('--use_trained',type=bool,default=False,help = 'use already trained model ')
    parser.add_argument('--model_path',type=str,default='./multi_model.pt',help = 'use already trained model ')

    # 解析命令行参数
    args = parser.parse_args()

    # 调用主函数
    dlt(args)