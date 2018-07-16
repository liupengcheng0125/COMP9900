#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  9 22:30:59 2018

@author: liupengcheng
"""
import nltk
import os
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from gensim.models.doc2vec import Doc2Vec
##nltk.download("punkt")
path = "/Users/liupengcheng/Downloads/result_stem"
files = os.listdir(path)
#print(files)
data = []
i = 0
for file in files: 
    if i < 29906:
        i=i+1
        temp_file = open(path + "/" +file).read()
        data.append(temp_file)
    else:
        break
#print(all_file)
'''data = ["I love machine learning. Its awesome.",
        "I love coding in python",
        "I love building chatbots",
        "they chat amagingly well"]'''
print(len(data))
tagged_data = [TaggedDocument(gensim.utils.simple_preprocess(_d), tags=[str(i)]) for i, _d in enumerate(data)]
print(list(tagged_data)[0])

max_epochs = 10
vec_size = 300
alpha = 0.025

model = Doc2Vec(size=vec_size,
                alpha=alpha, 
                min_alpha=0.025,
                min_count=1,
                dm =1)
  
model.build_vocab(tagged_data)

for epoch in range(max_epochs):
    print('iteration {0}'.format(epoch))
    model.train(tagged_data,
                total_examples=model.corpus_count,
                epochs=model.iter)
    # decrease the learning rate
    model.alpha -= 0.0002
    # fix the learning rate, no decay
    model.min_alpha = model.alpha

model.save("d2v.model")
print("Model Saved")

model= Doc2Vec.load("d2v.model")
#to find the vector of a document which is not in training data
test_data = word_tokenize("I love chatbots".lower())
v1 = model.infer_vector(test_data)
print("V1_infer", v1)

# to find most similar doc using tags
similar_doc = model.docvecs.most_similar('1')
print(similar_doc)


# to find vector of doc in training data using tags or in other words, printing the vector of document at index 1 in training data
print(model.docvecs['3'])