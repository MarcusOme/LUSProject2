#USAGE
#run with python run.py -options
from itertools import izip
import string
import os
import os.path
import numpy as np
import math
from collections import Counter
from collections import defaultdict
import time
import sys
import random

def check_labels(labels_list, sentence):
	result = False
	for label in labels_list:
		if label in sentence:
			result = True
		else:
			result = False
	return result

def find_batch(number):
	divisors = []
	batch_size = 0
	print("Calculating divisors...")
	for i in range(10, number):
		if number%i==0:
			divisors.append(i)
	if len(divisors)>0:
		min_value = min(divisors)
		batch_size = min_value
	else:
		batch_size = 100
	print(batch_size)
	return batch_size

			

#get arguments
type_option = '-n'
if len(sys.argv)>1:
	type_option = sys.argv[1]

#variables
train_IOB = "dataset/data/NLSPARQL.train.data"
test_IOB = "dataset/data/NLSPARQL.test.data"
train_feats = "dataset/data/NLSPARQL.train.feats.txt"
test_feats = "dataset/data/NLSPARQL.test.feats.txt"
train_complete = "train_complete.txt"
test_complete = "test_complete.txt"
dictionary = "word_dict.txt"
labels = "labels_dict.txt"
dev_set = "dev_set.txt"
sentences = list()
iob_tags = []

#create dictionary
dic_out = open(dictionary, 'w')
lab_out = open(labels, 'w')
labels_list = []
word_list = []
tmp_str = ""
tmp_labels = ""
with open(train_IOB) as file:
	for line in file:
		w = line.split()
		if len(w)>0:
			labels_list.append(w[1])
			word_list.append(w[0])
			tmp_str += w[0]+" "+w[1]+"\n"
		else:
			sentences.append(tmp_str+"\n")
			tmp_str = ""


label_counter = dict(Counter(labels_list))	#count the label occurence, to avoid the labels 1 freq cutoff
#extract only the labels that appears one time
sensitive_labels = []
for k, v in label_counter.items():
	if v==1:
		sensitive_labels.append(k)
#delete duplicates
labels_list = list(set(labels_list))


#divide test in dev and test, use the desired percentage
dev_out = open(dev_set, "w")
output = open(train_complete, "w")
total_number = 3338
percentage = int(total_number*0.1) 	#get the 10% for validation
random_index = np.random.randint(0, total_number, percentage)	#get the index of n random sentences

for i in range(0,total_number):
	if i in random_index:
		#check for unique labels
		'''
		if check_labels(sensitive_labels, sentences[i]) == True:
			#not write to dev, but to train
			output.write(sentences[i])
		else:
			dev_out.write(sentences[i])
			'''
		dev_out.write(sentences[i])
	else:
		output.write(sentences[i])

output.close()
dev_out.close()

#now get the word list present inside the train only
counter = 0

#creating dev and test files
output2 = open(test_complete, "w")
with open(test_IOB) as file1, open(test_feats) as file2:
	for x, y in izip(file1, file2):
		w1 = x.split()
		w2 = y.split()
		#get the value
		print_str = ""
		if len(w1)>0:
			print_str = w1[0]+" "+w1[1]+"\n"
		else:
			print_str = "\n"

		output2.write(print_str)

output2.close()

#also include test files
with open(test_complete) as file:
	for line in file:
		w = line.split()
		if len(w)>0:
			word_list.append(w[0])
word_list = list(set(word_list))
#write to file
for w in word_list:
	dic_out.write(w+" "+str(counter)+"\n")
	counter += 1
dic_out.write("<UNK> "+str(counter)+"\n")
counter = 0
for elem in labels_list:
	lab_out.write(elem+" "+str(counter)+"\n")
	counter += 1	
lab_out.write("<UNK> "+str(counter)+"\n")
dic_out.close()
lab_out.close()

counter = 0



print_str = ""
with open(dev_set) as file:
	for line in file:
		w = line.split()
		if len(w)>0:
			print_str += w[0]+" "+w[1]+"\n"
		else:
			print_str += "\n"
#now update the set
output3 = open(dev_set, 'w')
output3.write(print_str)
output3.close()
print("-- FILE PREPARATION DONE --")
print("-- TRAIN RNN --")
#get the number of sentences in train set
number_sentences = 0
with open(train_complete) as file:
	for line in file:
		if line=='\n':
			number_sentences += 1
#see what are the possible divisors of that number

lr = [0.1]	#low learning rate requires more epochs to train, so associate learning rate to epochs
bs = [7]
nh = [100]
ep = [25]
#train the RNN
if type_option=='-j':
	#jordan type
	counter = 0
	while counter < len(lr):
		for batch_size in bs:
			for hidden_layers in nh:
				config_file = open("rnn_slu/config.cfg", "w")
				config_string = "lr: "+str(lr[counter])+"\nwin: 9\nbs: "+str(batch_size)+"\nnhidden: "+str(hidden_layers)+"\nseed: 3842845\nemb_dimension: 100\nnepochs: "+str(ep[counter])+"\n"
				config_file.write(config_string)
				config_file.close()
				out_name = "results/"+"jordan_"+str(lr[counter])+"_"+str(batch_size)+"_"+str(hidden_layers)+"_"+str(ep[counter])+"_out.txt"
				os.system("python rnn_slu/lus/rnn_jordan_train.py train_complete.txt dev_set.txt "+dictionary+" "+labels+" rnn_slu/config.cfg model_elman")
				out_file = open(out_name, "w")
				out_file.close()
				os.system("python rnn_slu/lus/rnn_jordan_test.py model_elman test_complete.txt "+dictionary+" "+labels+" rnn_slu/config.cfg test_out.txt")
				os.system("./conlleval.pl < test_out.txt > "+out_name)
		counter += 1
elif type_option=='-e':
	#ellman type
	counter = 0
	while counter < len(lr):
		for batch_size in bs:
			for hidden_layers in nh:
				config_file = open("rnn_slu/config.cfg", "w")
				config_string = "lr: "+str(lr[counter])+"\nwin: 9\nbs: "+str(batch_size)+"\nnhidden: "+str(hidden_layers)+"\nseed: 3842845\nemb_dimension: 100\nnepochs: "+str(ep[counter])+"\n"
				config_file.write(config_string)
				config_file.close()
				out_name = "results/"+"elman_"+str(lr[counter])+"_"+str(batch_size)+"_"+str(hidden_layers)+"_"+str(ep[counter])+"_out.txt"
				os.system("python rnn_slu/lus/rnn_elman_train.py train_complete.txt dev_set.txt "+dictionary+" "+labels+" rnn_slu/config.cfg model_elman")
				out_file = open(out_name, "w")
				out_file.close()
				os.system("python rnn_slu/lus/rnn_elman_test.py model_elman test_complete.txt "+dictionary+" "+labels+" rnn_slu/config.cfg test_out.txt")
				os.system("./conlleval.pl < test_out.txt > "+out_name)
		counter += 1
elif type_option=="-gru":
	counter = 0
	while counter < len(lr):
		for batch_size in bs:
			for hidden_layers in nh:
				config_file = open("rnn_slu/config.cfg", "w")
				config_string = "lr: "+str(lr[counter])+"\nwin: 11\nbs: "+str(batch_size)+"\nnhidden: "+str(hidden_layers)+"\nseed: 3842845\nemb_dimension: 100\nnepochs: "+str(ep[counter])+"\n"
				config_file.write(config_string)
				config_file.close()
				out_name = "results/"+"gru_"+str(lr[counter])+"_"+str(batch_size)+"_"+str(hidden_layers)+"_"+str(ep[counter])+"_out.txt"
				os.system("python rnn_slu/lus/rnn_gru_train.py train_complete.txt dev_set.txt "+dictionary+" "+labels+" rnn_slu/config.cfg model_elman")
				out_file = open(out_name, "w")
				out_file.close()
				os.system("python rnn_slu/lus/rnn_gru_test.py model_elman test_complete.txt "+dictionary+" "+labels+" rnn_slu/config.cfg test_out.txt")
				os.system("./conlleval.pl < test_out.txt > "+out_name)
		counter += 1
elif type_option=="-lstm":
	counter = 0
	while counter < len(lr):
		for batch_size in bs:
			for hidden_layers in nh:
				config_file = open("rnn_slu/config.cfg", "w")
				config_string = "lr: "+str(lr[counter])+"\nwin: 11\nbs: "+str(batch_size)+"\nnhidden: "+str(hidden_layers)+"\nseed: 3842845\nemb_dimension: 100\nnepochs: "+str(ep[counter])+"\n"
				config_file.write(config_string)
				config_file.close()
				out_name = "results/"+"lstm_"+str(lr[counter])+"_"+str(batch_size)+"_"+str(hidden_layers)+"_"+str(ep[counter])+"_out.txt"
				os.system("python rnn_slu/lus/rnn_lstm_train.py train_complete.txt dev_set.txt "+dictionary+" "+labels+" rnn_slu/config.cfg model_elman")
				out_file = open(out_name, "w")
				out_file.close()
				os.system("python rnn_slu/lus/rnn_lstm_test.py model_elman test_complete.txt "+dictionary+" "+labels+" rnn_slu/config.cfg test_out.txt")
				os.system("./conlleval.pl < test_out.txt > "+out_name)
		counter += 1
elif type_option=="-lstm_base":
	counter = 0
	while counter < len(lr):
		for batch_size in bs:
			for hidden_layers in nh:
				config_file = open("rnn_slu/config.cfg", "w")
				config_string = "lr: "+str(lr[counter])+"\nwin: 11\nbs: "+str(batch_size)+"\nnhidden: "+str(hidden_layers)+"\nseed: 3842845\nemb_dimension: 100\nnepochs: "+str(ep[counter])+"\n"
				config_file.write(config_string)
				config_file.close()
				out_name = "results/"+"lstm_base_"+str(lr[counter])+"_"+str(batch_size)+"_"+str(hidden_layers)+"_"+str(ep[counter])+"_out.txt"
				os.system("python rnn_slu/lus/rnn_lstm_base_train.py train_complete.txt dev_set.txt "+dictionary+" "+labels+" rnn_slu/config.cfg model_elman")
				out_file = open(out_name, "w")
				out_file.close()
				os.system("python rnn_slu/lus/rnn_lstm_base_test.py model_elman test_complete.txt "+dictionary+" "+labels+" rnn_slu/config.cfg test_out.txt")
				os.system("./conlleval.pl < test_out.txt > "+out_name)
		counter += 1





			


