#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 19:20:00 2020

@author: francesco
"""
import numpy as np
from sklearn.tree._tree import TREE_LEAF
from sklearn.tree import DecisionTreeClassifier
import copy

from lasts.explanations.rule import Inequality, Rule


class Node:
    def __init__(self, idx, idxleft, idxright, idxancestor, feature, threshold, label):
        """
        Parameters
        ----------
        idx : int
            node index in the _tree scikit structure
        idxleft : int
            index of the left node
        idxright : int
            index of the right node
        idxancestor : int
            index of the ancestor
        feature : int
            idx of the feature in the dataset
        threshold : float
            threshold value for the tree node
        label : int
            majority class for instances passing through that node

        """
        self.idx = idx
        self.idxleft = idxleft
        self.idxright = idxright
        self.idxancestor = idxancestor
        self.feature = feature
        self.threshold = threshold
        self.label = label
        self.left = None
        self.right = None
        self.ancestor = None


class SklearnDecisionTreeConverter(object):
    def __init__(self, decision_tree: DecisionTreeClassifier):
        """

        Parameters
        ----------
        decision_tree : DecisionTreeClassifier object
            scikit decision tree classifier
        """
        self.n_nodes = np.array(decision_tree.tree_.node_count)
        self.children_left = np.array(decision_tree.tree_.children_left)
        self.children_right = np.array(decision_tree.tree_.children_right)
        self.features = np.array(decision_tree.tree_.feature)
        self.thresholds = np.array(decision_tree.tree_.threshold)
        # self.labels = np.array(estimator.tree_.value.argmax(axis=2).ravel())
        labels_idxs = np.array(decision_tree.tree_.value.argmax(axis=2).ravel())
        self.labels = []
        for idx in labels_idxs:
            self.labels.append(decision_tree.classes_[idx])
        self._build()

    def _build(self):
        nodes = []
        for node_idx in range(self.n_nodes):
            if (len(np.argwhere(self.children_right == node_idx)) == 0) and (
                len(np.argwhere(self.children_left == node_idx)) == 0
            ):  # if the node isn't ever a child (ancestor)
                idxancestor = None
            else:
                if len(np.argwhere(self.children_right == node_idx)) != 0:
                    idxancestor = np.argwhere(self.children_right == node_idx).ravel()[
                        0
                    ]
                else:
                    idxancestor = np.argwhere(self.children_left == node_idx).ravel()[0]
            new_node = Node(
                idx=node_idx,
                idxleft=self.children_left[node_idx],
                idxright=self.children_right[node_idx],
                idxancestor=idxancestor,
                feature=self.features[node_idx],
                threshold=self.thresholds[node_idx],
                label=self.labels[node_idx],
            )
            nodes.append(new_node)
        for node in nodes:
            node.left = nodes[node.idxleft] if node.idxleft != -1 else None
            node.right = nodes[node.idxright] if node.idxright != -1 else None
            node.ancestor = (
                nodes[node.idxancestor] if node.idxancestor is not None else None
            )
        self.nodes = nodes
        return self

    def _get_rule(self, root_leaf_path, as_contained=False, labels=None):
        thresholds_signs = []
        for i, node_idx in enumerate(root_leaf_path["path"][:-1]):
            node = self.nodes[node_idx]
            if node.left.idx == root_leaf_path["path"][i + 1]:
                thresholds_signs.append("<=")
            else:
                thresholds_signs.append(">")
        root_leaf_path["thresholds_signs"] = thresholds_signs
        conditions = list()
        for i, node_idx in enumerate(root_leaf_path["path"][:-1]):
            condition = Inequality(
                root_leaf_path["features"][i],
                root_leaf_path["thresholds_signs"][i],
                root_leaf_path["thresholds"][i],
                as_contained=as_contained,
            )
            conditions.append(condition)
        rule = Rule(conditions, root_leaf_path["labels"][-1], labels=labels)
        return rule

    def get_factual_rule_by_idx(self, idx, as_contained=False, labels=None):
        """
        Parameters
        ----------
        labels
        idx : int
            leaf index obtained via scikit .apply method
        as_contained

        Returns
        -------

        """
        return self.get_factual_rule(
            self._get_node_by_idx(idx), as_contained=as_contained, labels=labels
        )

    def get_factual_rule(self, node: Node, as_contained=False, labels=None):
        path = []
        features = []
        majority_labels = []
        thresholds = []
        while node is not None:
            path.append(node.idx)
            features.append(node.feature)
            majority_labels.append(node.label)
            thresholds.append(node.threshold)
            node = node.ancestor

        rule = self._get_rule(
            {
                "path": path[::-1],
                "features": features[::-1],
                "labels": majority_labels[::-1],
                "thresholds": thresholds[::-1],
                "thresholds_signs": None,
            },
            as_contained=as_contained,
            labels=labels,
        )
        return rule

    def _print_subtree(self, node: Node, level=0):
        if node != None:
            self._print_subtree(node.left, level + 1)
            print(
                "%s -> %s %.2f %s"
                % (" " * 12 * level, node.feature, node.threshold, node.label)
            )
            self._print_subtree(node.right, level + 1)

    def print_tree(self):
        self._print_subtree(self.nodes[0])

    def _minimum_distance(self, x: Node):
        return minimum_distance(self.nodes[0], x)

    def get_counterfactual_rule(
        self, factual_node: Node, as_contained=False, labels=None
    ):
        _, nearest_leaf = self._minimum_distance(factual_node)
        counterfactual = self.get_factual_rule(
            self.nodes[nearest_leaf], as_contained=as_contained, labels=labels
        )
        return counterfactual

    def _get_node_by_idx(self, idx):
        return self.nodes[idx]

    def get_counterfactual_rule_by_idx(
        self, factual_idx, as_contained=False, labels=None
    ):
        return self.get_counterfactual_rule(
            self._get_node_by_idx(factual_idx), as_contained=as_contained, labels=labels
        )


def find_leaf_down(root: Node, lev, min_dist, min_idx, x):
    # base case
    if root is None:
        return
    # If this is a leaf node, then check if it is closer than the closest so far
    if (root.left is None and root.right is None) and root.label != x.label:
        if (lev < (min_dist[0])) and lev > 0:
            min_dist[0] = lev
            min_idx[0] = root.idx
        return
    # Recur for left and right subtrees
    find_leaf_down(root.left, lev + 1, min_dist, min_idx, x)
    find_leaf_down(root.right, lev + 1, min_dist, min_idx, x)


def find_through_parent(root: Node, x: Node, min_dist, min_idx):
    """
    Find if there is closer a leaf to x through a parent node.
    Parameters
    ----------
    root
    x
    min_dist
    min_idx

    Returns
    -------

    """
    # Base cases
    if root is None:
        return -1
    if root == x:
        return 0

    # Search x in left subtree of root
    l = find_through_parent(root.left, x, min_dist, min_idx)

    # If left subtree has x
    if l != -1:
        # Find closest leaf in right subtree
        find_leaf_down(root.right, l + 2, min_dist, min_idx, x)
        return l + 1

    # Search x in right subtree of root
    r = find_through_parent(root.right, x, min_dist, min_idx)

    # If right subtree has x
    if r != -1:
        # Find closest leaf in left subtree
        find_leaf_down(root.left, r + 2, min_dist, min_idx, x)
        return r + 1

    return -1


def minimum_distance(root: Node, x: Node):
    """
    Find the minimum distance of a counterfactual leaf from a given node x

    Parameters
    ----------
    root
    x

    Returns
    -------

    """
    # Initialize result (minimum distance from a leaf)
    min_dist = [np.inf]

    min_idx = [None]

    # Find closest leaf down to x
    find_leaf_down(x, 0, min_dist, min_idx, x)

    # See if there is a closer leaf
    # through parent
    find_through_parent(root, x, min_dist, min_idx)

    return min_dist[0], min_idx[0]


def get_branch_length(node: Node):
    count = -1
    while node is not None:
        count += 1
        node = node.ancestor
    return count


def prune_index(inner_tree, decisions, index=0):
    # Start pruning from the bottom - if we start from the top, we might miss
    # nodes that become leaves during pruning.
    # Do not use this directly - use prune_duplicate_leaves instead.
    if not is_leaf(inner_tree, inner_tree.children_left[index]):
        prune_index(inner_tree, decisions, inner_tree.children_left[index])
    if not is_leaf(inner_tree, inner_tree.children_right[index]):
        prune_index(inner_tree, decisions, inner_tree.children_right[index])

    # Prune children if both children are leaves now and make the same decision:
    if (
        is_leaf(inner_tree, inner_tree.children_left[index])
        and is_leaf(inner_tree, inner_tree.children_right[index])
        and (decisions[index] == decisions[inner_tree.children_left[index]])
        and (decisions[index] == decisions[inner_tree.children_right[index]])
    ):
        # turn node into a leaf by "unlinking" its children
        inner_tree.children_left[index] = TREE_LEAF
        inner_tree.children_right[index] = TREE_LEAF
        # print("Pruned {}".format(index))


def prune_duplicate_leaves(dt: DecisionTreeClassifier):
    # Remove leaves if both
    decisions = (
        dt.tree_.value.argmax(axis=2).flatten().tolist()
    )  # Decision for each node
    prune_index(dt.tree_, decisions)


def prune(tree: DecisionTreeClassifier):
    tree = copy.deepcopy(tree)
    dat = tree.tree_
    nodes = range(0, dat.node_count)
    ls = dat.children_left
    rs = dat.children_right
    classes = [[list(e).index(max(e)) for e in v] for v in dat.value]

    leaves = [(ls[i] == rs[i]) for i in nodes]

    LEAF = -1
    for i in reversed(nodes):
        if leaves[i]:
            continue
        if leaves[ls[i]] and leaves[rs[i]] and classes[ls[i]] == classes[rs[i]]:
            ls[i] = rs[i] = LEAF
            leaves[i] = True
    return tree


def is_leaf(inner_tree, index):
    # Check whether node is leaf node
    return (
        inner_tree.children_left[index] == TREE_LEAF
        and inner_tree.children_right[index] == TREE_LEAF
    )


if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn import tree

    X, y = load_iris(return_X_y=True)
    clf = tree.DecisionTreeClassifier(random_state=0)
    clf = clf.fit(X, y)

    newtree = SklearnDecisionTreeConverter(clf)

    rule = newtree.get_factual_rule(newtree.nodes[1])
    print(rule)
