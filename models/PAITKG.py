# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 05:18:14 2023

@author: 28257
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer,AutoModel
from models.AdaptedBERT import AdaptedBERT
from models.RelaGraph import KGEModel
import argparse

class PAITKG(nn.Module):
    def __init__(self, 
                 pretrained_bert,pretrained_kge,
                 bottleneck_dim,
                 class_num):
        super(PAITKG, self).__init__()
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
        
        self.perta=AdaptedBERT(pretrained_bert,self.bottleneck_dim,self.feature_dim)
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