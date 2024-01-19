import numpy as np
import pandas as pd
import argparse

def decision_tree(data, target_attribute_name, parent_node_class=None, depth=0, pre_att=None, pre_val=None) :

    if depth == 0:
        entropy_val = entropy(data, target_attribute_name)
        print(f"0,root,{entropy_val},no_leaf")

    if len(np.unique(data[target_attribute_name])) == 1:
        entropy_val = entropy(data, target_attribute_name)
        print(f"{depth},{pre_att}={pre_val},{abs(entropy_val)},{np.unique(data[target_attribute_name])[0]}")
        return np.unique(data[target_attribute_name])[0]

    if entropy(data, target_attribute_name) == 1:
        majority_class = data[target_attribute_name].mode().iloc[0]
        print(f"{depth},{pre_att}={pre_val},1,{majority_class}")
        return majority_class

    if len(data) == 0 or len(data.columns) == 1:
        entropy_val = entropy(data, target_attribute_name)
        print(f"{depth},{pre_att}={pre_val},{entropy_val},{parent_node_class}")
        return parent_node_class
    best_att = selection(data, target_attribute_name)

    if depth > 0:
        print(f"{depth},{pre_att}={pre_val},{entropy(data, target_attribute_name)},no_leaf")
    tree = {best_att: {}}
    for value in np.unique(data[best_att]):
        sub_data = data.where(data[best_att] == value).dropna()
        subtree = decision_tree(sub_data, target_attribute_name, np.unique(sub_data[target_attribute_name])[0], depth + 1, best_att, value)
        tree[best_att][value] = subtree
    return tree

def entropy(data, target_attribute_name) :
    values, counts = np.unique(data[target_attribute_name], return_counts=True)
    total_instances = len(data)
    entropy_val = -np.sum([(counts[i] / total_instances) * (1/np.log2(c))*np.log2(counts[i] / total_instances) for i in range(len(values))])
    return entropy_val

def info_gain(data, attribute_name, target_attribute_name) :
    total_entropy = entropy(data, target_attribute_name)
    values, counts = np.unique(data[attribute_name], return_counts=True)
    weighted_entropy = 0
    total_instances = len(data)
    for i in range(len(values)) :
        value_data = data[data[attribute_name] == values[i]]
        value_proportion = len(value_data) / total_instances
        value_entropy = entropy(value_data, target_attribute_name)
        weighted_entropy += value_proportion * value_entropy
    info_gain = total_entropy - weighted_entropy
    return info_gain

def selection(data, target_attribute_name):
    attributes = [col for col in data.columns if col != target_attribute_name]
    information_gains = []
    for att in attributes :
        information_gains.append(info_gain(data, att, target_attribute_name))
    best_att = attributes[np.argmax(information_gains)]
    return best_att


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Linear Regression with Gradient Descent")
    parser.add_argument("--data", required=True, help="CSV file containing the data for linear regression")
    path = parser.parse_args()
    df = pd.read_csv(path.data, header=None)
    column_names = [f'att{i}' for i in range(len(df.columns))]
    df.columns = column_names
    c = df[column_names[-1]].nunique()
    decision_tree(df, column_names[-1])
