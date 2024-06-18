# Further train a pre trained model with MLM  I. Achitouv 2023
# qsub AdaptativePre-Training.py
# qstat
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from spacy.lang.en import English
import re
import random
from numpy.linalg import norm 

import nltk
#nltk.download('punkt')
from nltk.tokenize import sent_tokenize,word_tokenize
import torch
from transformers import AutoTokenizer, AutoModel,AutoModelForMaskedLM

from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers import DataCollatorForLanguageModeling, DataCollatorForWholeWordMask
from transformers import Trainer, TrainingArguments

########################
# load model to further train
# this one is nice because it is fast and has good score for semantic search
model_card = "all-MiniLM-L6-v2" 
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

#'bert-base-uncased'
#model = AutoModelForMaskedLM.from_pretrained(model_card)
#tokenizer = AutoTokenizer.from_pretrained(model_card)

#model_card ="sentence-transformers/gpt2"
#tokenizer = AutoTokenizer.from_pretrained("gpt2")
#model=SentenceTransformer(model_card)



# model train
per_device_train_batch_size = 64
per_device_eval_batch_size=32
save_steps = 1000               #Save model every 1k steps
num_train_epochs =3             #Number of epochs 3 by default
use_fp16 = False                #Set to True, if your GPU supports FP16 operations
max_length = 200                #Max length for a text input
do_whole_word_mask = True       #If set to true, whole words are masked
mlm_prob = 0.15 #0.15                 #Probability that a word is replaced by a [MASK] token


#### CHANGE THE PATH HERE 
output_dir="/rds/general/user/iachitou/home/DomainA_all-MiniLM-L6-v2/results/domain_pre_trainingv2_mlm0.15"



#Cleaning functions
def sentence_clean(text):
    
     # removing mentions
    text = re.sub("@\S+", "", text)
     # remove urls
    text = re.sub("https?:\/\/.*[\r\n]*", "", text)
     # removing hashtags
    text = re.sub("#", "", text)
    # remove special characters and digits
    text=re.sub("(\\d|\\W)+"," ",text)
    
    # remove digits
    text = re.sub("\d+", "", text)
    # remove bracket
    text=re.sub(r"[\([{})\]]", "", text)
    text=re.sub(r"\([^()]*\)", "", text)
    
    return text


alphabets= "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov)"
digits = "([0-9])"


def split_into_sentences(text):
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    text = re.sub(digits + "[.]" + digits,"\\1<prd>\\2",text)
    if "..." in text: text = text.replace("...","<prd><prd><prd>")
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    
    text = text.replace(":"," ")
    text = text.replace(","," ")
    text = text.replace('.'," ")
    text = text.replace(']'," ")
    text = text.replace('['," ")
    text = text.replace(','," ")
    text = text.replace("'"," ")
    text = text.replace('('," ")
    text = text.replace('’'," ")
    text = text.replace(')'," ")
    text = text.replace('“'," ")
    text = text.replace('”'," ")
    text = text.replace(';'," ")
    text = text.replace(','," ")
    text = text.replace('//'," ")
    text = text.replace('``'," ")
    text = text.replace("''"," ")
    text = text.replace("'-"," ")
    text = text.replace('\\\\'," ")
    # remove digits
    text = re.sub("\d+", "", text)
    # remove bracket
    text=re.sub(r"[\([{})\]]", "", text)
    
    text=re.sub(r"\([^()]*\)", "", text)
   # text = text.replace(," ")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return sentences


def cleancorpus(Corpus):
    Corpus_s=[]
    for i in range(len(Corpus)):
        t=split_into_sentences(Corpus[i])
        for j in range(len(t)):
            t2=t[j]
            if len(t2)>2:
                Corpus_s.append(t2)
    
    
    cleanCorpus=[]
    for i in range(len(Corpus_s)):
        cleanCorpus.append(sentence_clean(Corpus_s[i]))
        
    #datas=random.shuffle(cleanCorpus)
    return cleanCorpus


def splitchunk(data,lchunck):
    data2=[]
    for sent in data:
        if len(sent)<lchunck:
            data2.append(sent)
        else:
            for i in range(0,len(sent),lchunck):
                data2.append(sent[i:i+lchunck])
    return data2



#A dataset wrapper, that tokenizes our data on-the-fly
class TokenizedSentencesDataset:
    def __init__(self, sentences, tokenizer, max_length, cache_tokenization=False):
        self.tokenizer = tokenizer
        self.sentences = sentences
        self.max_length = max_length
        self.cache_tokenization = cache_tokenization

    def __getitem__(self, item):
        if not self.cache_tokenization:
            return self.tokenizer(self.sentences[item], add_special_tokens=True, truncation=True, max_length=self.max_length, return_special_tokens_mask=True)

        if isinstance(self.sentences[item], str):
            self.sentences[item] = self.tokenizer(self.sentences[item], add_special_tokens=True, truncation=True, max_length=self.max_length, return_special_tokens_mask=True)
        return self.sentences[item]

    def __len__(self):
        return len(self.sentences)



###################### READ CORPUS FROM WHICH WE FURTHER TRAIN
db_file ="/rds/general/user/iachitou/home/Finreg-E/Rulebooks.db"
con = sqlite3.connect(db_file)
cur = con.cursor()


sqlQry = "select * from FCARulebook_FEB22"
dataList = pd.read_sql_query(sqlQry, con)
    # Be sure to close the connection
con.close()


print(len(dataList))
TextRul=[]
ic=0
for i in range(0, len(dataList)):
    dataToPrint = dataList['RegulatorRuleFullRequirement'][i]
    if dataList['RegulatorRuleFullRequirement'][i]!="[deleted]":
        TextRul.append(dataToPrint)
        ic+=1
print("there was",ic,"rules",len(TextRul))


######################## prepare and train data
data=cleancorpus(TextRul)
random.shuffle(data)
data=splitchunk(data,200)
print("we now have",len(data),"rules")

train_sentences=data[:int(0.8*len(data))]
dev_sentences=data[int(0.8*len(data)):]
#train_sentences=data[:200]
#dev_sentences=data[201:230]
print("Train sentences:", len(train_sentences))
print("Val sentences:", len(dev_sentences))

train_dataset = TokenizedSentencesDataset(train_sentences, tokenizer, max_length)
dev_dataset = TokenizedSentencesDataset(dev_sentences, tokenizer, max_length, cache_tokenization=True)

if do_whole_word_mask:
    data_collator = DataCollatorForWholeWordMask(tokenizer=tokenizer, mlm=True, mlm_probability=mlm_prob)
else:
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=mlm_prob)

training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    num_train_epochs=num_train_epochs,
    evaluation_strategy="steps",
    per_device_train_batch_size=per_device_train_batch_size,
    eval_steps=save_steps,
    save_steps=save_steps,
    logging_steps=save_steps,
    save_total_limit=2,
    prediction_loss_only=True,
    fp16=use_fp16
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset
)

print("Save tokenizer to:", output_dir)
tokenizer.save_pretrained(output_dir)

trainer.train()

print("Save model to:", output_dir)
model.save_pretrained(output_dir)

print("Training done")

