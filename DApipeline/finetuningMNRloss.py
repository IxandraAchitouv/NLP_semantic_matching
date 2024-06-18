# I. Achitouv 2023 This code fine tune on labelled data algo on pre-trained model
# Here I used a trick on using labelled data (supervise learning)
# from unsupervised results of many models or better from humain eyes pairing. 
#https://www.sbert.net/docs/training/overview.html
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from spacy.lang.en import English
import re
from transformers_domain_adaptation import DataSelector
from numpy.linalg import norm 

import nltk
#nltk.download('punkt')
from nltk.tokenize import sent_tokenize,word_tokenize
import torch
from transformers import AutoTokenizer, AutoModel,AutoModelForMaskedLM
from sentence_transformers import SentenceTransformer, InputExample,util
from torch.utils.data import DataLoader

#######
#outputfile='./sbert_train+FT_mnr/domain_pre_training_bertfrompretrain'
#outputfile='./sbert_train+FT_mnr/domain_pre_training_bert'
inputmodel="all-MiniLM-L6-v2"

# this are score confidence cut from training set
#to select good but not perfect match
csmall=70
chigh=95

outputfile="/rds/general/user/iachitou/home/DomainA_all-MiniLM-L6-v2/resultsFT/"+inputmodel+"_FT"+str(csmall)+"-"+str(chigh)



####### this is the training input file here the one you previously sent me
file="/rds/general/user/iachitou/home/DomainA/mappingData_AM.csv"
df=pd.read_csv(file,sep=";", error_bad_lines=False)
df.head(20)
df2bis = df[df["Confidence"] > csmall ]
df3=df2bis[df2bis["Confidence"] < chigh ]
df3=df3.dropna()
print(len(df),len(df3))
#df3.tail(15)
df3 = df3.sample(frac=1,random_state=1)
Ncut=int(0.8*len(df3))

df2=df3[:Ncut]
dfval=df3[Ncut:]
print(Ncut,len(df2),len(dfval))
dfval.to_csv("/rds/general/user/iachitou/home/DomainA_all-MiniLM-L6-v2/resultsFT/dfvalidation_"+str(csmall)+"-"+str(chigh)+".csv")
dfval.head()
########


from sentence_transformers import SentenceTransformer, InputExample, losses,SentencesDataset
from torch.utils.data import DataLoader
from sentence_transformers import models, SentenceTransformer
from sentence_transformers import evaluation


# with pretrain
model=SentenceTransformer(inputmodel)
#bert = models.Transformer(inputmodel)
#pooler = models.Pooling(
#    bert.get_word_embedding_dimension(),
#    pooling_mode_mean_tokens=True)
#model = SentenceTransformer(modules=[bert, pooler])


train_examples=[InputExample(texts=[df2["RegSent"].iloc[i], df2["PolicySent"].iloc[i]])for i in range(len(df2))]
len(train_examples)

             
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=32)    

train_loss = losses.MultipleNegativesRankingLoss(model)

#Fine Tune the model
epochs = 3
warmup_steps = int(len(train_dataloader) * epochs * 0.1)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=epochs,
    steps_per_epoch=10,
    #evaluator=evaluator,
    warmup_steps=warmup_steps,
    output_path=outputfile
)  


