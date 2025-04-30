"""
Gets the best performing model for each h-param optimization and combines this into a common csv.
"""

# native modules
import os

# 3rd party modules
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# own modules


def main():

    root = os.path.join('models')

    results_files = [f for f in os.listdir(root) if os.path.isfile(os.path.join(root, f)) and 'metrics.csv' in f]

    df_aggregate = None

    for f in results_files:
        df = pd.read_csv(os.path.join(root,f),index_col=False)

        idx_max = df[['valid_acc']].idxmax() # get the best model index
        best_metrics = df.iloc[idx_max]

        # add meta data
        best_metrics['data'] = 'Compressed' if 'Compressed' in f else 'Separate'
        best_metrics['trial'] = f.split('_')[0]

        if df_aggregate is None:
            df_aggregate = best_metrics
        else:

            df_aggregate = pd.concat([df_aggregate,best_metrics], ignore_index=True)

    # save the combined metrics
    df_aggregate.to_csv('Aggregate_best_model.csv',index=False)

    # create graphs of the results
    sns.set_theme()
    dist = 0.2

    fig = plt.figure(figsize=(14,8))
    X_axis = np.arange(len(df_aggregate))
    plt.bar(X_axis + (-1) * dist, df_aggregate["train_acc"], dist, label='train')
    plt.bar(X_axis + 0 * dist, df_aggregate["valid_acc"], dist, label='valid')
    plt.bar(X_axis + 1 * dist, df_aggregate["test_acc"], dist, label='test')
    plt.xticks(X_axis, df_aggregate['trial'])
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Aggregate_best_model.png')

if __name__ == '__main__':

    main()