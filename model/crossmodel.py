import torch.nn as nn
import torch
from transformers import RobertaModel
import torchvision.models as models
class TextModel(nn.Module):
    def __init__(self,config):
        super(TextModel,self).__init__()
        self.bert = RobertaModel.from_pretrained('roberta-base')
        self.trans = nn.Sequential(
            nn.Dropout(config.bert_dropout),
            nn.Linear(self.bert.config.hidden_size, config.middle_hidden_size),
            nn.ReLU(inplace=True)
        )

        for param in self.bert.parameters():
            param.requires_grad = True


    def forward(self, bert_inputs, masks, token_type_ids=None):
        bert_out = self.bert(input_ids=bert_inputs, token_type_ids=token_type_ids, attention_mask=masks)
        hidden_state = bert_out['last_hidden_state']
        pooler_out = bert_out['pooler_output']

        return self.trans(hidden_state), self.trans(pooler_out)

class ImageModel(nn.Module):
    def __init__(self, config):
        super(ImageModel, self).__init__()
        self.full_resnet = models.resnet34(pretrained=True)
        self.resnet_h = nn.Sequential(
            *(list(self.full_resnet.children())[:-2]),
        )

        # (batch, 512, 7, 7) -> (batch, img_hidden_seq, middle_hidden_size)
        self.hidden_trans = nn.Sequential(
            nn.Conv2d(self.full_resnet.fc.in_features, config.img_hidden_seq, 1),
            nn.Flatten(start_dim=2),
            nn.Dropout(config.resnet_dropout),
            nn.Linear(7*7, config.middle_hidden_size),    # 这里的7*7是根据resnet34，原img大小为224*224的情况来的
            nn.ReLU(inplace=True)
        )
        for param in self.full_resnet.parameters():
            param.requires_grad = True

    def forward(self, imgs):
        hidden_state = self.resnet_h(imgs)

        return self.hidden_trans(hidden_state)



class Model(nn.Module):
    def __init__(self, config, text_ablation=False, image_ablation=False, attn_ablation=False):
        super(Model,self).__init__()
        self.text_model = TextModel(config)
        self.image_model = ImageModel(config)

        self.attention = nn.MultiheadAttention(
            embed_dim=config.middle_hidden_size,
            num_heads=config.attention_nhead,
            dropout=config.attention_dropout,
        )

        self.attention_layer_norm = nn.LayerNorm(config.middle_hidden_size)  # 添加层标准化

        # 全连接分类器
        self.text_img_classifier = nn.Sequential(
            nn.Dropout(config.fuse_dropout),
            nn.Linear(config.middle_hidden_size * 2, config.out_hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(config.fuse_dropout),
            nn.Linear(config.out_hidden_size, config.num_labels),
            nn.Softmax(dim=1)
        )


        # 全连接分类器
        encoder_layer = nn.TransformerEncoderLayer(d_model=config.middle_hidden_size, nhead=config.attention_nhead,batch_first=True)
        transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.classifier = transformer_encoder

        self.fc = nn.Sequential(
            nn.Linear(config.middle_hidden_size,config.num_labels),
            nn.Softmax(dim=1)
        )


    def forward(self, texts, texts_mask, imgs, labels=None):
        text_hidden_state, text_feature = self.text_model(texts, texts_mask)

        img_hidden_state= self.image_model(imgs)


        text_hidden_state = text_hidden_state.permute(1, 0, 2)
        img_hidden_state = img_hidden_state.permute(1, 0, 2)


        #self-attetion

        img_result = self.classifier(img_hidden_state)

        #cross attention
        text_img_attention_out, _ = self.attention(img_hidden_state,text_hidden_state, text_hidden_state)
        img_text_attention_out, _ = self.attention(text_hidden_state,img_hidden_state, img_hidden_state)
        text_img_attention_out = self.attention_layer_norm(text_img_attention_out)#新增部分
        img_text_attention_out = self.attention_layer_norm(img_text_attention_out)


        #self attention
        text_img_attention_out,_ = self.attention(text_img_attention_out,text_img_attention_out,text_img_attention_out)
        img_text_attention_out,_ = self.attention(img_text_attention_out,img_text_attention_out,img_text_attention_out)
        text_img_attention_out = self.attention_layer_norm(text_img_attention_out)#新增部分
        img_text_attention_out = self.attention_layer_norm(img_text_attention_out)


        #shape:(src_len,batch_size,num_hidden)
        #img_result = img_result[-1,:,:]
        img_result = torch.mean(img_result, dim=0).squeeze(0)
        #text_result=text_result[-1,:,:]
        text_img_attention_out = text_img_attention_out[-1,:,:]
        img_text_attention_out = torch.mean(img_text_attention_out, dim=0).squeeze(0)
        result = img_result + img_text_attention_out + text_feature + text_img_attention_out
        prob_vec = self.fc(result)

        return prob_vec