import numpy as np
import sys, os
sys.path.append(os.path.abspath("."))
import argparse
from pathlib import Path

from src.comparisons import compare_rule_base, antecedent_comparison
from src.fuzzy_partitions import generate_fuzzy_partitions_lims, get_y_sets
from datasets.load_dataset import Data
from src.mamdani import compute_MSE, wang_mendel


def demo(datasetName='abalone', n_labels=3):
    print(f'{datasetName} dataset with {n_labels} labels')

    train_path = Path( "datasets" ) / datasetName / f"{datasetName}.dat"  # The data is not partitioned train/test
    test_path = Path( "datasets" ) / datasetName / f"{datasetName}.dat" 

    # Get the data from the .dat file
    train_DataInfos = Data(train_path)
    train_data = train_DataInfos.get_data() # FIXME: May fail if there is any non-numeric attribute
    x = train_data[1]
    y = train_data[2]
    xy = np.append(x, np.expand_dims(y, 1), axis=1)

    # optionally split here the x, y data to create the test suite
    test_DataInfos = Data(test_path)
    test_data = test_DataInfos.get_data() # FIXME: May fail if there is any non-numeric attribute
    test_x = test_data[1]
    test_y = test_data[2]
    test_xy = np.append(test_x, np.expand_dims(test_y, 1), axis=1)

    attributes = train_data[0]['attributes']
    attribs_lims = np.array([v for v in attributes.values()])
    fp = generate_fuzzy_partitions_lims(attribs_lims, n_labels=n_labels)
    '''
    fp represents the fuzzy partitions.
    is a list containting the partitions for each atribute/consequent
    For each attribute there is a list of terms,
    and each list has the 3 key points of the triangular fuzzy set

    [
        [  # First fuzzy variable
            [ 0.00000e+00  1.00000e+00  2.00000e+00]  # "LOW
            [ 1.00000e+00  2.00000e+00  3.00000e+00]  # "MEDIUM"
            [ 2.00000e+00  3.00000e+00  4.00000e+00]  # "HIGH"
        ]

        [  # Second ...
            [-2.95000e-01  7.50000e-02  4.45000e-01]
            [ 7.50000e-02  4.45000e-01  8.15000e-01]
            [ 4.45000e-01  8.15000e-01  1.18500e+00]
        ]
        ...
    ]

    '''
    xy_range, y_sets = get_y_sets(fp)




    # # train
    print("Wang - Mendel Method")
    rules_wm, _ = wang_mendel(xy, fp)
    '''
    rules_wm is a np.array of list.
    Each sublist is a rule, representing the index of the set of the antecedent
    The last value is the consequent.
    [
      [1 2 1 0 1 ] # the 2nd term of the 1st varialbe, the 3rd of the second, ...
      ...

    ]
    '''
    print('Fuzzy Partitions:')
    print(fp)
    print('Rules:')
    print(rules_wm)

    # Compare two rules:
    rule1 = rules_wm[0]
    rule2 = rules_wm[1]
    rule_comp = antecedent_comparison(rule1, rule2, fp)
    print(f'Rule comparison for 1st and 2nd rule: {rule_comp}')


    print(f'Rule comparison for 1st and 2nd rule (min): {antecedent_comparison(rule1, rule2, fp, np.min)}')
    print(f'Rule comparison for 1st and 2nd rule (mean): {antecedent_comparison(rule1, rule2, fp, np.mean)}')
    print(f'Rule comparison for 1st and 2nd rule (max): {antecedent_comparison(rule1, rule2, fp, np.max)}')
    # Compare the whole rule base, each rule against each other
    # Returns a symmetric matrix for the values of compare rule i and j
    comp_mat = compare_rule_base(rules_wm, fp)
    print('Comparison matrix:')
    print(comp_mat)

    # If you need to delete or add rules, change the rules_wm np.array
    # add rule
    rules_wm = np.append(rules_wm, [2, 2, 2, 0, 1, 1, 1, 1, 2])
    rules_wm = np.delete(rules_wm, 1, axis=0)

    # Evaluation MSE
    MSE_wm = compute_MSE(rules_wm, test_xy, fp)
    print(f'Test MSE: {MSE_wm}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parameter parse of this project")
    
    parser.add_argument('--dataset', type=str, default='abalone',
                        help='Dataset: abalone, ailerons, airfoil, ANACALT, autoMPG6, autoMPG8, \
                            baseball, california, compactiv, concrete, dee, delta_ail, delta_elv, \
                            diabetes, ele-1, ele-2, elevators, forestFires, friedman, house, laser, \
                            machineCPU, mortgage, mv, plastic, pole, puma32h, quake, stock, tic, \
                            treasury, wankara, wizmir]')

    parser.add_argument('--nLabels', type=int, default=3,
                        help='nSamples (d = 3)') 


    args = parser.parse_args()

    nLabels = args.nLabels
    dataset = args.dataset
    demo(dataset, nLabels)
