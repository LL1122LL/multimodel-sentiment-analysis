# Multimodal-Sentiment-Analysis

基于BERT+ResNet34的融合方法，实现了针对于静态图像和文本的多模态情感分析任务。

## Project Structure

```
├─data
│  ├─data#存放图片和文本数据
│  ├─train.txt#用于训练的图片和数据索引
│  ├─test_without_label.txt#用于预测的文件的索引
├─model#使用的多模态模型
├─trained_model#存放训练好的模型的参数
├─utility
│  ├─data_process_utiliy#对data内的数据进行预处理，返回train_loader,val_loader,test_loader
│  ├─draw#用于绘画loss和accuracy曲线
│  ├─train_test_predict#用于对模型的训练、验证，输入数据的情感预测以及保存最优的模型参数
├─config.py#设置模型相关网络的参数
├─main.py#入口，运行整个模型
```



## Setup

依赖环境：

- matplotlib==3.7.1
- numpy==1.24.2
- Pillow==9.5.0
- Pillow==10.2.0
- scikit_learn==1.3.1
- torch==2.1.1
- torchvision==0.16.1
- tqdm==4.66.1
- transformers==4.34.0

```
pip install -r requirements.txt
```



## Run

进入main.py所在的根目录

```
python main.py 
```

这个命令会默认执行lr=5e-6,drop_out=0.2,epoch=10,的模型训练，验证，预测过程，



```
python main.py --use_trained True
```

这个命令会将执行先前训练好保存下来的模型，输出test_without_label.txt预测结果,若这里还没有模型，可能需要你先训练一遍，并且修改main.py中output_path的默认值，使它指向你训练好的模型的地址，



```
python main.py --option 2
```

选择是否进行消融实验，当值为0时，仅有图片数据;值为1时，仅有文本数据，值为2时，两者都有。



## Attribution

参考的仓库
[NHFNET]:(https://github.com/ECNU-Cross-Innovation-Lab/NHFNet)

