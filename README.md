# ResNet Generator Script

## Summary

A Python script to generate train and solver prototxt files for Deep Residual Networks.

There are 4 stages in a typical ResNet, each stage can have multiple blocks, each block have three or four units. If it’s the first block of a stage then it will have four units; and three for all subsequent blocks. See this awesome visualization for reference (http://ethereon.github.io/netscope/#/gist/db945b393d40bfa26006).

stages = [3, 2, 2, 2]	# Number of layers at each stage

Total number of layers can be computed as follows
num_layers = sum(stages)*3 + len(stages) + 1

NOTE: Convolution layer parameters can be modified from make_train_test_net() function.

Apologies for not writing an argument parser, I feel comfortable making modifications here when the argument count is greater than 3-4.

NOTE: Current settings will generate a 32 layered network for CIFAR-10