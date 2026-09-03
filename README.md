# GaussianFuzzy

## Files

### demo.py

A simple demo generating rules with the Wang-Mendel algorithm and doing some comparisons

### src/comparison.py

Functions to compare rules

The aggregation of the comparison can be modified

antecedent_comparison: compares all the elements of the provided list

comparison: for classification if the consequents are different returns 0, and if they are equal it calls antecedent_comparison for the antecedents

compare_rule_base: returns a matrix comparing all the rules with each other

### src/mamdani.py

Implementation of the Mamdani inference, Wang-Mendel method and some performance measures (MSE)

### src/fuzzy_partitions.py

Some functions to create and manage fuzzy partitions from data

### datasets/load_dataset.py

Functions to load .dat files

