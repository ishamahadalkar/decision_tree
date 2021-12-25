# decision_tree
Implemented the Decision tree algorithm from scratch and analyzing a 650+ line dataset of passengers onboard the Titanic to predict which passengers would survive the disaster.

Performed extensive analysis by calculating the entropy and information gain and using cross validation to create decision tress with an overall accuracy of 87% on the validation data.

I implemeted different versions of the binary decision tree

1. Implementing a binary decision tree with no pruning using the ID3 (Iterative Dichotomiser) algorithm. 

Format of calling the function and accuracy you will get after training:
$ python3 ID3.py --trainFolder ./path/to/train-folder \
--testFolder ./path/to/test-folder \
--model vanilla \
--crossValidK 5

2. Implementing a binary decision tree with a given maximum depth. 

The format of calling the function and accuracy you will get after training:
$ python3 ID3.py --trainFolder ./path/to/train-folder \
--testFolder ./path/to/test-folder \
--model depth \
--crossValidK 5 \
--depth 10

3. Implementing a binary decision tree with a given minimum sample split size. 
Format of calling the function and accuracy you will get after training:
$ python3 ID3.py --trainFolder ./path/to/train-folder \
--testFolder ./path/to/test-folder \
--model minSplit \
--crossValidK 5 \
--minSplit 2

4. Implementing a binary decision tree with post-pruning using Reduced Error Pruning.

Format of calling the function and accuracy you will get after training:
$ python3 ID3.py --trainFolder ./path/to/train-folder \
--testFolder ./path/to/test-folder \
-- model postPrune \
--crossValidK 5

Sample Output:
fold=1, train set accuracy= ____, validation set accuracy= _______
fold=2, train set accuracy= ____, validation set accuracy= _______
fold=3, train set accuracy= ____, validation set accuracy= _______
fold=4, train set accuracy= ____, validation set accuracy= _______
fold=5, train set accuracy= ____, validation set accuracy= _______
Test set accuracy= ____