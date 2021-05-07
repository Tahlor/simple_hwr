""" Plots seaborn graph over time

"""

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def test():
    d = {}
    d['means'] = np.array([1,2,3,4])
    d['stds'] = np.array([1,1.2,1.3,.8])
    d['label'] = "label1"
    d['xaxis'] = list(range(len(d['means'])))
    plot_data([d])

def read_data():
    N  = 5
    df = pd.read_csv("data.csv", delimiter='\t')
    print(df.columns)
    df.x = df.x.astype(float)
    d1 = {}
    d1['means'] = np.convolve(df.y1, np.ones((N,))/N, mode='valid')
    d1['stds1'] = np.random.randn(*d1['means'].shape) / 100
    d1['stds2'] = np.random.randn(*d1['means'].shape) / 100

    d1['label'] = "No GT adaptation"
    d1['xaxis'] = df.x[:len(d1['means'])]

    d2 = {}
    d2['means'] = np.convolve(df.y3, np.ones((N,))/N, mode='valid')
    d2['stds'] = np.random.randn(*d2['means'].shape)/100
    d2['label'] = "With  GT adaptation"
    d2['xaxis'] = df.x[:len(d2['means'])]
    plot_data([d1,d2])


def plot_data(list_of_data_dict):
    """

    Args:
        dict with:
            means:
            stds:
            xaxis:
            label:

    Returns:

    """
    fig, ax = plt.subplots()
    clrs = sns.color_palette("husl", 5)

    with sns.axes_style("darkgrid"):
        for i,d in enumerate(list_of_data_dict):
            meanst = np.array(d['means'], dtype=np.float64)
            std = np.array(d['stds'], dtype=np.float64)
            ax.plot(d['xaxis'], meanst, label=d['label'], c=clrs[i])
            ax.fill_between(d['xaxis'], meanst-std, meanst+std ,alpha=0.3, facecolor=clrs[i])
        ax.legend()
        #ax.set_yscale('log')
    fig.show()

if __name__=='__main__':
    #test()
    read_data()