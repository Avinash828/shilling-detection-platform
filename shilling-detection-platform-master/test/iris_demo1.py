# # coding=utf-8
# from __future__ import division
# from sklearn import datasets
# import pandas as np
# import pandas as pd
#
#
# iris = datasets.load_iris()
#
# # We'll also import seaborn, a Python graphing library
# import warnings # current version of seaborn generates a bunch of warnings that we'll ignore
# warnings.filterwarnings("ignore")
# import seaborn as sns
# import matplotlib.pyplot as plt
# sns.set(style="white", color_codes=True)
#
# # Next, we'll load the Iris flower dataset, which is in the "../input/" directory
# # iris = pd.read_csv("/Users/jay/Desktop/datasets/iris/Iris.csv")  # the iris dataset is now a Pandas DataFrame
#
# # Let's see what's in the iris data - Jupyter notebooks print the result of the last thing you do
# iris.head()
#
# from bubbly.bubbly import bubbleplot
# from plotly.offline import plot
#
# figure = bubbleplot(dataset=iris, x_column='SepalLengthCm', y_column='PetalLengthCm', z_column='SepalWidthCm',
#     bubble_column='Id', size_column='PetalWidthCm', color_column='Species',
#     x_title="SepalLength(Cm)", y_title="PetalLength(Cm)", z_title='SepalWidth(Cm)',
#                     title='IRIS Visualization',
#     x_logscale=False,scale_bubble=0.1,height=600)
#
# plot(figure, config={'scrollzoom': True})