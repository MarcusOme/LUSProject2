#USAGE
#compile it with python code.py
#OPTIONS
#insert -r to recompile CRF
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

#get arguments
recompile_option = '-n'
if len(sys.argv)>1:
	recompile_option = sys.argv[1]
#variables
train_IOB = "dataset/data/NLSPARQL.train.data"
test_IOB = "dataset/data/NLSPARQL.test.data"
train_feats = "dataset/data/NLSPARQL.train.feats.txt"
test_feats = "dataset/data/NLSPARQL.test.feats.txt"
train_complete = "train_complete.txt"
test_complete = "test_complete.txt"
test_test = "dataset2/test.txt"
test_train = "dataset2/train.txt"

lexer = []
converter = []
labels = []
unigrams = defaultdict(int)
bigrams = defaultdict(int)
tags_list = []

#create the new file that contain IOB tags and grammar tags for training
output = open(train_complete, "w")
with open(train_IOB) as file1, open(train_feats) as file2:
	for x, y in izip(file1, file2):
		w1 = x.split()
		w2 = y.split()
		#get the value
		print_str = ""
		if len(w1)>0:
			if w1[1]=='O':
				print_str = w1[0]+" "+w1[1]+" "+w1[1]+"\n"
			else:	
				print_str = w1[0]+" "+w1[1]+" "+w1[1]+"\n"	#structure word prefix prefix&grammar final_tag
		else:
			print_str = "\n"

		output.write(print_str)

output.close()

#creating test files for CRF
output2 = open(test_complete, "w")
with open(test_IOB) as file1, open(test_feats) as file2:
	for x, y in izip(file1, file2):
		w1 = x.split()
		w2 = y.split()
		print(w1)
		print(w2)
		#get the value
		print_str = ""
		if len(w1)>0:
			if w1[1]=='O':
				print_str = w1[0]+" "+w1[1]+"\n"
			else:	
				print_str = w1[0]+" "+w1[1]+"\n"
		else: 
			print_str = "\n"

		output2.write(print_str)

output2.close()

if not os.path.exists('model.txt') or recompile_option=='-r':
	print("Recompile the model")
	os.system("crf_learn template.txt "+train_complete+" model.txt")

os.system("echo '-- TRAIN DONE --'")
os.system("echo '-- START TESTING --'")
os.system("crf_test -m model.txt "+test_complete+" -o results.txt")
os.system("echo '-- EVAL MODEL --'")
os.system("./conlleval.pl -d '\t' < results.txt > eval.txt")
os.system("echo '-- EVAL DONE --'")
print('--Done--')
