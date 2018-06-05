# LUSProject2
University of Trento - Final project

## Required tools
In order to run the project is necessary to install several tools:
* [Python 2.7](https://www.python.org/downloads/): installed and configured properly
* [CRF++](https://taku910.github.io/crfpp/): follow installation instructions and if needed add the PYTHONPATH
* [Theano](http://www.deeplearning.net/software/theano/): follow installation procedure

## Download and execute the code
Clone and enter in the Github repository using the following command:

```
git clone https:https://github.com/MarcusOme/LUSProject2.git
cd LUSProject1
```

After downloading you will see two folders inside the directory: CRF and RNN.

### CRF
Enter the folder and run one of the two python code present:

```
cd CRF
python base_test.py -r
```

or

```
cd CRF
python tag_test.py -r
```

or

```
cd CRF
python best.py -r
```

Where the option -r indicates the necessity to retrain the model. Excluding it mean testing with the current saved model (the one present in "model.txt").

### RNN
Enter the folder and run one of the two python code present:

```
cd RNN
python run.py -option
```
For executing RNN code several options are present:
* -j: to train and test a jordan type RNN
* -e: to train and test a elman type RNN
* -lstm_base: to train and test with basic version of long short term memory RNN
* -lstm: to train and test a CNNLSTM
* -gru: to train and test a gated recurrent unit RNN

### Paramenters update in CRF
To use personalized paramenters change the "template.txt" or "template_base.txt" file.

### Parameters update in RNN
To use personalized paramenters change the code inside "run.py".

## Folder structure
The project is organized in several folders. First of all the division of CRF and RNN.

### \CRF
In this folder is present the implementation for the Conditional Random Fields over the movie dataset. In the main folders are present the script to execute the CRF analysis and the input and output files of the analysis. In particular "template.txt" and "template_base.txt" contains the template for the CRF and "train_complete.txt" and "test_complete.txt" contains the train and test data. The python scripts instead are two:
* base_test.py: contains the baseline analysis for the dataset and uses "template_base.text"
* tag_test.py: contains the code to perform the analysis that merges IOB-tags and lemmas, use the "template.txt" file
* best_test.py: contains the code for the best analysis registered on the dataset, use "template.txt" file

In the dataset folder are present the dataset that includes IOB-tags document and lemmas. In results folder the result for the best.py file is saved to have a comparison metric for other analysis.

### \RNN
In this folder are present the files to train and test different models of Recurrent Neural Network. In dataset folder are present the current train and test files. In the folder model_elman are saved the parameters of the trained network. In results there are several results obtained with different methods and paramenters. In rrn_slu instead are present the specification for each RNN model and the script to create, train and test each model.

In order to read the results in the \results folder is important undertand the name notation: each file is in the format method_lr_bs_nh_ep.txt, where:
* method: can be elman, jordan, lstm, lstm_base and gru
* lr: the starting learning rate
* bs: batch size
* nh: number of hidden layer
* ep: number of epochs for training

Several files are already present for the analysis already done. Remember that, in case of parameters change a new file will be created in case not exists yet.

## Author

Marco Omezzolli - marco.omezzolli@studenti.unitn.it
