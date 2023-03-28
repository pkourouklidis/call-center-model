from pickle import load
import os
import pandas as pd
from sklearn import tree

os.chdir(os.path.dirname(os.path.abspath(__file__)))
with open("model.joblib", "rb") as file:
    clf = load(file)

tree.plot_tree(clf)
dot_data = tree.export_graphviz(clf, out_file="test.dot", filled=True) 