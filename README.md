# Companion Rule List
Python code for Companion rule list. 

## Requirements

## Usage:
Prepare data:

Install associated libs.

## Parameters:
--file: binarized data, last column should be the label, default as adult.csv
--blx_file: black box classification label, default as rf_adult.csv
--alpha: hyper parameter to control rule length, default as 0.001
--step: training steps, default as 20000
--card: rule cardinality, default as 2
--supp: rule mining min support, keep rules cover > support, default as 0.05
--n: number of positive and negtive rules, defaule as 200

## Rusult form
[if condition_1 is 1/0 and condition_2 is 1/0, then y= 1/0. transparency = ?, accuracy = ?]
...
Here, condition_1 and condition_2 are data features and y is the prediction. 
Accuracy represents the prediction performance of hybrid models with different number of rules.
