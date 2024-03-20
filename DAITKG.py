# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 05:18:14 2023

@author: 28257
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer,AutoModel
from models.PERTA import PERTA
from models.RelaGraph import KGEModel
import argparse

class DAITKG(nn.Module):
    def __init__(self, 
                 pretrained_bert,pretrained_kge,
                 prompt_len,
                 bottleneck_dim,
                 class_num):
        super(DAITKG, self).__init__()
        for param in pretrained_bert.parameters():
            param.requires_grad=False

        for param in pretrained_kge.parameters():
            param.requires_grad=False
        for param in pretrained_kge.rela_before_attn_in.parameters():
            param.requires_grad=True
        for param in pretrained_kge.rela_before_attn_out.parameters():
            param.requires_grad=True
        for param in pretrained_kge.attn_layers.parameters():
            param.requires_grad=True

        self.bottleneck_dim=bottleneck_dim
        self.feature_dim=pretrained_kge.entity_dim
        self.prompt_len=prompt_len
        
        self.perta=PERTA(pretrained_bert,self.prompt_len,self.bottleneck_dim,self.feature_dim)
        self.kge=pretrained_kge
        
        self.integrate=nn.Linear(self.perta.hidden_size, self.feature_dim)
        self.dense = nn.Linear(self.feature_dim, class_num)
        self.sigmoid=nn.Sigmoid()
        self.softmax=nn.Softmax(dim=1)

    def forward(self, paper,text,mask,labels=None):
        embed=self.embedding(paper,text,mask)
        logits=self.dense(embed)
        prob=self.sigmoid(logits)
        if labels is not None:
            
            MultiLabelLoss=-(labels*torch.log(prob+1e-15)+(1-labels)*torch.log(1-prob+1e-15))
            PositiveWeight=self.softmax(-logits*2-(1-labels)*1e15)
            NegativeWeight=self.softmax(logits*2-labels*1e15)
            PositiveLoss=torch.sum(PositiveWeight*MultiLabelLoss,dim=1)
            NegativeLoss=torch.sum(NegativeWeight*MultiLabelLoss,dim=1)
            loss=torch.mean(PositiveLoss+NegativeLoss)
            '''
            MultiLabelLoss=-(labels*torch.log(prob+1e-15)+(1-labels)*torch.log(1-prob+1e-15))
            loss=torch.mean(torch.sum(MultiLabelLoss,dim=1))
            '''
            return (prob,loss)
        return prob
    
    def embedding(self, paper,text,mask):
        kg_emb=self.kge._encode_func(paper)
        text_emb=self.perta(input_ids=text,attention_mask=mask)
        
        embed=self.integrate(text_emb)+kg_emb
        return embed

if __name__=='__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--cuda', action='store_true', help='use GPU', default=True)

    parser.add_argument('--do_train', action='store_true', default=True)
    parser.add_argument('--do_valid', action='store_true', default=False)
    parser.add_argument('--do_test', action='store_true', default=False)
    parser.add_argument('--evaluate_train', action='store_true', help='Evaluate on training data')

    parser.add_argument('--data_name', type=str, default='triples', help='dataset name, default to KMHEO')
    parser.add_argument('--use_RelaGraph', default=True, type=bool)
    
    parser.add_argument('-n', '--negative_sample_size', default=64, type=int)
    parser.add_argument('-d', '--hidden_dim', default=64, type=int)
    parser.add_argument('-g', '--gamma', default=6.0, type=float)
    parser.add_argument('-adv', '--negative_adversarial_sampling', action='store_true', default=True)
    parser.add_argument('-a', '--adversarial_temperature', default=2.0, type=float)
    parser.add_argument('-b', '--batch_size', default=256, type=int)
    parser.add_argument('--test_batch_size', default=16, type=int, help='valid/test batch size')
    parser.add_argument('--uni_weight', action='store_true',
                        help='Otherwise use subsampling weighting like in word2vec')

    parser.add_argument('-lr', '--learning_rate', default=0.001, type=float)
    parser.add_argument('-cpu', '--cpu_num', default=2, type=int)
    parser.add_argument('-randomSeed', default=0, type=int)
    parser.add_argument('-init', '--init_checkpoint', default=None, type=str)
    parser.add_argument('-save', '--save_path', default=None, type=str)
    parser.add_argument('--max_steps', default=50000, type=int)
    parser.add_argument('--warm_up_steps', default=None, type=int)

    parser.add_argument('--save_checkpoint_steps', default=100000, type=int)
    parser.add_argument('--valid_steps', default=1000, type=int)
    parser.add_argument('--log_steps', default=1000, type=int, help='train log every xx steps')
    parser.add_argument('--test_log_steps', default=2000, type=int, help='valid/test log every xx steps')

    parser.add_argument('--print_on_screen', action='store_true', default=True, help='log on screen or not')
    parser.add_argument('--ntriples_eval_train', type=int, default=200000,
                        help='number of training triples to evaluate eventually')
    parser.add_argument('--neg_size_eval_train', type=int, default=500,
                        help='number of negative samples when evaluating training triples')

    parser.add_argument('--true_negative', action='store_true', default=True, help='whether to remove existing triples from negative sampling')
    parser.add_argument('--inverse', action='store_true', help='whether to add inverse edges')
    parser.add_argument('--val_inverse', action='store_true', help='whether to add inverse edges to the validation set')
    parser.add_argument('--drop', type=float, default=0.3, help='Dropout in layers')
    
    parser.add_argument('-u', '--triplere_u', default=1.0, type=float)
    parser.add_argument('--anchor_size', default=0.2, type=float, help='size of the anchor set, i.e. |A|')
    parser.add_argument('-ancs', '--sample_anchors', default=5, type=int)
    parser.add_argument('-path', '--use_anchor_path', default=True)
    parser.add_argument('--sample_neighbors', default=5, type=int)
    parser.add_argument('--max_relation', default=3, type=int)
    parser.add_argument('-center', '--sample_center', default=True)
    parser.add_argument('--node_dim', default=0, type=int)
    parser.add_argument('-merge', '--merge_strategy', default='mean_pooling', type=str,
                        help='how to merge information from anchors, chosen between [ mean_pooling, linear_proj ]')
    parser.add_argument('-layers', '--attn_layers_num', default=1, type=int)
    parser.add_argument('--mlp_ratio', default=2, type=int)
    parser.add_argument('--head_dim', default=8, type=int)
    parser.add_argument('-type', '--add_type_embedding', default=True)
    parser.add_argument('-share', '--anchor_share_embedding', default=True)
    parser.add_argument('-skip', '--anchor_skip_ratio', default=0.2, type=float)
    args = parser.parse_args([])
    processor = DataProcessor(data_name=args.data_name,
                              inverse=args.inverse, val_inverse=args.val_inverse)
    pretrained_kge = KGEModel(
        processor=processor,
        args=args,
    )
    if args.cuda:
        pretrained_kge = pretrained_kge.cuda()
    if args.init_checkpoint:
        # Restore model from checkpoint directory
        checkpoint = torch.load('outputs/kge/checkpoint')
        pretrained_kge.load_state_dict(checkpoint['model_state_dict'])
    pretrained_bert=AutoModel.from_pretrained("bert-base-uncased")
    
    daitkg=DAITKG(pretrained_bert,pretrained_kge,
                  2,
                  8,
                  235).cuda()
    
    paper=torch.tensor([123,231]).cuda()
    len1=128
    len2=255
    text1=[514]*len1+[103]*(512-len1)
    text2=[114]*len2+[103]*(512-len2)
    mask1=[1]*(len1+2)+[0]*(512-len1)
    mask2=[1]*(len2+2)+[0]*(512-len2)
    text=torch.tensor([text1,text2]).cuda()
    mask=torch.tensor([mask1,mask2]).cuda()
    labels=torch.tensor([[1]+[0]*230+[1]+[0]*3,[0]*120+[1]+[0]*100+[1]+[0]*13]).cuda()
    daitkg(paper,text,mask,labels)
    
    