# Description:
This is the code for paper [Companion rule list]. In this paper, we designed a model(CRL) with flexible interpretability.
This model provides a seris of hybrid submodels with different transparency. The input data should be in a binarized form.
In the example, we use 80% data(adult) to train the model and the result was reflexed on the left 20% data. 
This value is adjustable by setting parameter 'ratio' in the program.
A file containing black box prediction for the data should be prepared before training.

## Usage:
1. Install associated libs.
2. Download the code.
3. Run.

## Parameters:
--file: binarized data, last column should be the label, default adult.csv
--blx_file: black box classification label, default rf_adult.csv
--alpha: hyper parameter to control rule length, default 0.001
--step: training steps, default 20000
--card: rule cardinality, default 2
--supp: rule mining min support, keep rules cover > support, default 0.05
--n: number of positive and negtive rules, defaule 200

## Rusult form
[if condition_1 is 1/0 and condition_2 is 1/0, then y= 1/0. transparency = ?, accuracy = ?]
...
Here, condition_1 and condition_2 are data features and y is the prediction. 
Accuracy represents the prediction performance of hybrid models with different number of rules.
