import os
class config:
    #类别数量
    num_labels = 3

    # Fuse相关
    middle_hidden_size = 32
    attention_nhead = 8
    attention_dropout = 0.4
    fuse_dropout = 0.5
    out_hidden_size = 128
    
    # BERT相关
    bert_name = 'roberta-base'
    bert_dropout = 0.2

    # ResNet相关
    image_size = 224
    resnet_dropout = 0.2
    img_hidden_seq = 64
