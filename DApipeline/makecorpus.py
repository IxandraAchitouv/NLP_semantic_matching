#### PREPARE THE LEARNING DATASET from rulebook for FINREG-E
#### by I. ACHITOUV 2023



import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from spacy.lang.en import English
import re

from numpy.linalg import norm 

import nltk
#nltk.download('punkt')
from nltk.tokenize import sent_tokenize,word_tokenize
import torch
import random

########################Cleaning functions
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


def main():
    print('start reading data...')


########## update the paths with yours
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

    data=cleancorpus(TextRul)
    random.shuffle(data)
    data=splitchunk(data,100)

    train_sentences=data[:int(0.8*len(data))]
    dev_sentences=data[int(0.8*len(data)):]
    print("Train sentences:", len(train_sentences))
    print("Val sentences:", len(dev_sentences))

    # open file in write mode
    with open(r'/rds/general/user/iachitou/home/Finreg-E/RuleDAtrain.txt', 'w') as fp:
    for item in train_sentences:
        # write each item on a new line
        fp.write("%s\n" % item)

    # open file in write mode
    with open(r'/rds/general/user/iachitou/home/Finreg-E/RuleDAtest.txt', 'w') as fp2:
    for item in dev_sentences:
        # write each item on a new line
        fp2.write("%s\n" % item)
    
        
    print('Done')
    
######################
if __name__ == '__main__':
    # freeze_support() here if program needs to be frozen
    main()  # execute this only when run directly, not when imported!
###############################
