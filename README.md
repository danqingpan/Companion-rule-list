# Companion Rule List
Python code for Companion rule list. 

## Requirements
To run this program, we need common libs such as numpy ro pandas.
Especially, bitarray and fim are necessary to run the program.

lib list:
* python 3.6
* bitarray
* fim
* numpy
* pandas
* itertools
* operator
* scipy
* maptlotlib
* math
* argparse

## Usage:
### Prepare data:
Data should be prepared in binary form. e.g.
age   gender  y
20    0       0
30    1       1

### Set Parameters:
* --file: binarized data, last column should be the label, default as adult.csv
* --blx_file: black box classification label, default as rf_adult.csv
* --alpha: hyper parameter to control rule length, default as 0.001
* --step: training steps, default as 20000
* --card: rule cardinality, default as 2
* --supp: rule mining min support, keep rules cover > support, default as 0.05
* --n: number of positive and negtive rules, defaule as 200

## Rusult form
The output form of the program is shown as follows:
[if condition_1 is 1/0 and condition_2 is 1/0, then y= 1/0. transparency = 20%, accuracy = 90%]
[if condition_3 is 1/0 and condition_4 is 1/0, then y= 1/0. transparency = 40%, accuracy = 88%]
...

* condition_1 and condition_2 represents the condition for making decision. 
* Transparency represents the proportion of data classified by the rule list.
* Accuracy represents the prediction performance of hybrid models with corresponding number of rules.
