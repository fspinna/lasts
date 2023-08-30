#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 15:16:29 2020

@author: francesco
"""
import pathlib

from pyts.datasets import make_cylinder_bell_funnel, load_gunpoint, load_coffee
from tslearn.generators import random_walk_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler  # doctest: +NORMALIZE_WHITESPACE
import warnings
from tslearn.datasets import UCR_UEA_datasets
from lasts.utils import get_project_root, convert_sktime_to_numpy, minmax_scale
from pyts.datasets import fetch_uea_dataset
from sktime.utils.data_io import load_from_tsfile_to_dataframe


def load_ucr_dataset(
    name,
    verbose=True,
    exp_dataset_threshold=10000,
    exp_dataset_ratio=0.3,
    random_state=0,
):
    data_loader = UCR_UEA_datasets()
    X_train, y_train, X_test, y_test = data_loader.load_dataset(name)

    assert len(X_train.shape) == 3
    assert X_train.shape[1] != 1

    label_encoder = False
    if y_train.min() != 0:
        label_encoder = True
        le = LabelEncoder()
        le.fit(y_train)
        y_train = le.transform(y_train)
        y_test = le.transform(y_test)

    if verbose:
        print("DATASET INFO:")
        if label_encoder:
            print("Label Encoding:", le.classes_)
        print("X_train: ", X_train.shape)
        print("y_train: ", y_train.shape)
        unique, counts = np.unique(y_train, return_counts=True)
        print("\nCLASSES BALANCE")
        for i, label in enumerate(unique):
            print(label, ": ", round(counts[i] / sum(counts), 2))
        print()
        print("X_test: ", X_test.shape)
        print("y_test: ", y_test.shape)
        unique, counts = np.unique(y_test, return_counts=True)
        print("\nCLASSES BALANCE")
        for i, label in enumerate(unique):
            print(label, ": ", round(counts[i] / sum(counts), 2))

    X_exp = X_train.copy()
    y_exp = y_train.copy()
    if X_train.shape[0] > exp_dataset_threshold:
        X_train, X_exp, y_train, y_exp = train_test_split(
            X_train,
            y_train,
            test_size=exp_dataset_ratio,
            stratify=y_train,
            random_state=random_state,
        )
    else:
        if verbose:
            warnings.warn("Blackbox and Explanation sets are the same")

    return (
        X_train,
        y_train,
        None,
        None,
        X_test,
        y_test,
        X_exp,
        y_exp,
        None,
        None,
        X_test,
        y_test,
    )


def load_uea_dataset(
    name,
    verbose=True,
    exp_dataset_threshold=10000,
    exp_dataset_ratio=0.3,
    random_state=0,
):
    data_loader = UCR_UEA_datasets()
    X_train, y_train, X_test, y_test = data_loader.load_dataset(name)
    y_train = y_train.astype("int")
    y_test = y_test.astype("int")

    assert len(X_train.shape) == 3
    assert X_train.shape[1] != 1

    label_encoder = False
    if y_train.min() != 0:
        label_encoder = True
        le = LabelEncoder()
        le.fit(y_train)
        y_train = le.transform(y_train)
        y_test = le.transform(y_test)

    if verbose:
        print("DATASET INFO:")
        if label_encoder:
            print("Label Encoding:", le.classes_)
        print("X_train: ", X_train.shape)
        print("y_train: ", y_train.shape)
        unique, counts = np.unique(y_train, return_counts=True)
        print("\nCLASSES BALANCE")
        for i, label in enumerate(unique):
            print(label, ": ", round(counts[i] / sum(counts), 2))
        print()
        print("X_test: ", X_test.shape)
        print("y_test: ", y_test.shape)
        unique, counts = np.unique(y_test, return_counts=True)
        print("\nCLASSES BALANCE")
        for i, label in enumerate(unique):
            print(label, ": ", round(counts[i] / sum(counts), 2))

    X_exp = X_train.copy()
    y_exp = y_train.copy()
    if X_train.shape[0] > exp_dataset_threshold:
        X_train, X_exp, y_train, y_exp = train_test_split(
            X_train,
            y_train,
            test_size=exp_dataset_ratio,
            stratify=y_train,
            random_state=random_state,
        )
    else:
        if verbose:
            warnings.warn("Blackbox and Explanation sets are the same")

    return (
        X_train,
        y_train,
        None,
        None,
        X_test,
        y_test,
        X_exp,
        y_exp,
        None,
        None,
        X_test,
        y_test,
    )


def load_generali_subsampled(random_state=0, verbose=True, start=0, end=2490):
    folder = get_project_root() / "datasets" / "cached" / "generali_subsample"
    X_acc_test_scaled_subsampling = np.load(
        folder / "X_acc_test_scaled_subsampling.npy"
    )
    X_pos_test_scaled_subsampling = np.load(
        folder / "X_pos_test_scaled_subsampling.npy"
    )
    y_test_subsampling = np.load(folder / "y_test_subsampling.npy")
    y = y_test_subsampling
    X = list()
    for dim in range(X_acc_test_scaled_subsampling.shape[2]):
        X.append(X_acc_test_scaled_subsampling[:, start:end, dim][:, :, np.newaxis])
    X.append(X_pos_test_scaled_subsampling[:, :, np.newaxis])
    if verbose:
        print("X_train: ", [dim.shape for dim in X])
        print("y_train: ", y.shape)
    return X, y_test_subsampling


def load_generali(
    path="", verbose=True, start=0, end=2490, train=True, validation=True, test=True
):
    folder = pathlib.Path(path)

    X_train, y_train, X_val, y_val, X_test, y_test = None, None, None, None, None, None

    if train:
        X_acc_train_scaled_subsampling = np.load(folder / "X_acc_train_scaled.npy")
        X_pos_train_scaled_subsampling = np.load(folder / "X_pos_train_scaled.npy")
        y_train = np.load(folder / "y_train.npy")
        X_train = list()
        for dim in range(X_acc_train_scaled_subsampling.shape[2]):
            X_train.append(
                X_acc_train_scaled_subsampling[:, start:end, dim][:, :, np.newaxis]
            )
        X_train.append(X_pos_train_scaled_subsampling[:, :, np.newaxis])
        if verbose:
            print("X_train: ", [dim.shape for dim in X_train])
            print("y_train: ", y_train.shape)

    if validation:
        X_acc_val_scaled_subsampling = np.load(folder / "X_acc_valid_scaled.npy")
        X_pos_val_scaled_subsampling = np.load(folder / "X_pos_valid_scaled.npy")
        y_val = np.load(folder / "y_valid.npy")
        X_val = list()
        for dim in range(X_acc_val_scaled_subsampling.shape[2]):
            X_val.append(
                X_acc_val_scaled_subsampling[:, start:end, dim][:, :, np.newaxis]
            )
        X_val.append(X_pos_val_scaled_subsampling[:, :, np.newaxis])
        if verbose:
            print("X_val: ", [dim.shape for dim in X_val])
            print("y_val: ", y_val.shape)

    if test:
        X_acc_test_scaled_subsampling = np.load(folder / "X_acc_test_scaled.npy")
        X_pos_test_scaled_subsampling = np.load(folder / "X_pos_test_scaled.npy")
        y_test = np.load(folder / "y_test.npy")
        X_test = list()
        for dim in range(X_acc_test_scaled_subsampling.shape[2]):
            X_test.append(
                X_acc_test_scaled_subsampling[:, start:end, dim][:, :, np.newaxis]
            )
        X_test.append(X_pos_test_scaled_subsampling[:, :, np.newaxis])
        if verbose:
            print("X_test: ", [dim.shape for dim in X_test])
            print("y_test: ", y_test.shape)

    return X_train, y_train, X_val, y_val, X_test, y_test


def build_cbf(n_samples=600, random_state=0, verbose=True):
    X_all, y_all = make_cylinder_bell_funnel(
        n_samples=n_samples, random_state=random_state
    )
    X_all = X_all[:, :, np.newaxis]

    if verbose:
        print("DATASET INFO:")
        print("X SHAPE: ", X_all.shape)
        print("y SHAPE: ", y_all.shape)
        unique, counts = np.unique(y_all, return_counts=True)
        print("\nCLASSES BALANCE")
        for i, label in enumerate(unique):
            print(label, ": ", round(counts[i] / sum(counts), 2))

    # BLACKBOX/EXPLANATION SETS SPLIT
    X_train, X_exp, y_train, y_exp = train_test_split(
        X_all, y_all, test_size=0.3, stratify=y_all, random_state=random_state
    )

    # BLACKBOX TRAIN/TEST SETS SPLIT
    X_train, X_test, y_train, y_test = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=random_state
    )

    # BLACKBOX TRAIN/VALIDATION SETS SPLIT
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=random_state
    )

    # EXPLANATION TRAIN/TEST SETS SPLIT
    X_exp_train, X_exp_test, y_exp_train, y_exp_test = train_test_split(
        X_exp, y_exp, test_size=0.2, stratify=y_exp, random_state=random_state
    )

    # EXPLANATION TRAIN/VALIDATION SETS SPLIT
    X_exp_train, X_exp_val, y_exp_train, y_exp_val = train_test_split(
        X_exp_train,
        y_exp_train,
        test_size=0.2,
        stratify=y_exp_train,
        random_state=random_state,
    )

    if verbose:
        print("\nSHAPES:")
        print("BLACKBOX TRAINING SET: ", X_train.shape)
        print("BLACKBOX VALIDATION SET: ", X_val.shape)
        print("BLACKBOX TEST SET: ", X_test.shape)
        print("EXPLANATION TRAINING SET: ", X_exp_train.shape)
        print("EXPLANATION VALIDATION SET: ", X_exp_val.shape)
        print("EXPLANATION TEST SET: ", X_exp_test.shape)

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        X_exp_train,
        y_exp_train,
        X_exp_val,
        y_exp_val,
        X_exp_test,
        y_exp_test,
    )


def build_ts_syege(path="./datasets/ts_syege/", random_state=0, verbose=True):
    X_all = np.load(path + "ts_syege01.npy")
    X_all = X_all[:, :, np.newaxis]

    if verbose:
        print("DATASET INFO:")
        print("X SHAPE: ", X_all.shape)
        # print("y SHAPE: ", y_all.shape)
        # unique, counts = np.unique(y_all, return_counts=True)
        # print("\nCLASSES BALANCE")
        # for i, label in enumerate(unique):
        # print(label, ": ", round(counts[i] / sum(counts), 2))

    # BLACKBOX/EXPLANATION SETS SPLIT
    X_train, X_exp = train_test_split(X_all, test_size=0.3, random_state=random_state)

    # BLACKBOX TRAIN/TEST SETS SPLIT
    X_train, X_test = train_test_split(
        X_train, test_size=0.2, random_state=random_state
    )

    # BLACKBOX TRAIN/VALIDATION SETS SPLIT
    X_train, X_val = train_test_split(X_train, test_size=0.2, random_state=random_state)

    # EXPLANATION TRAIN/TEST SETS SPLIT
    X_exp_train, X_exp_test = train_test_split(
        X_exp, test_size=0.2, random_state=random_state
    )

    # EXPLANATION TRAIN/VALIDATION SETS SPLIT
    X_exp_train, X_exp_val = train_test_split(
        X_exp_train, test_size=0.2, random_state=random_state
    )

    if verbose:
        print("\nSHAPES:")
        print("BLACKBOX TRAINING SET: ", X_train.shape)
        print("BLACKBOX VALIDATION SET: ", X_val.shape)
        print("BLACKBOX TEST SET: ", X_test.shape)
        print("EXPLANATION TRAINING SET: ", X_exp_train.shape)
        print("EXPLANATION VALIDATION SET: ", X_exp_val.shape)
        print("EXPLANATION TEST SET: ", X_exp_test.shape)

    return (
        X_train,
        None,
        X_val,
        None,
        X_test,
        None,
        X_exp_train,
        None,
        X_exp_val,
        None,
        X_exp_test,
        None,
    )


def build_multivariate_cbf(n_samples=600, n_features=3, random_state=0, verbose=True):
    X_all = [[], [], []]
    y_all = []
    for i in range(n_features):
        X, y = make_cylinder_bell_funnel(
            n_samples=n_samples, random_state=random_state + i
        )
        X = X[:, :, np.newaxis]
        for label in range(3):
            X_all[label].append(X[np.nonzero(y == label)])
    for i in range(len(X_all)):
        X_all[i] = np.concatenate(X_all[i], axis=2)
    for label in range(3):
        y_all.extend(label for i in range(len(X_all[label])))
    X_all = np.concatenate(X_all, axis=0)
    y_all = np.array(y_all)

    if verbose:
        print("DATASET INFO:")
        print("X SHAPE: ", X_all.shape)
        print("y SHAPE: ", y_all.shape)
        unique, counts = np.unique(y_all, return_counts=True)
        print("\nCLASSES BALANCE")
        for i, label in enumerate(unique):
            print(label, ": ", round(counts[i] / sum(counts), 2))

    # BLACKBOX/EXPLANATION SETS SPLIT
    X_train, X_exp, y_train, y_exp = train_test_split(
        X_all, y_all, test_size=0.3, stratify=y_all, random_state=random_state
    )

    # BLACKBOX TRAIN/TEST SETS SPLIT
    X_train, X_test, y_train, y_test = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=random_state
    )

    # BLACKBOX TRAIN/VALIDATION SETS SPLIT
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=random_state
    )

    # EXPLANATION TRAIN/TEST SETS SPLIT
    X_exp_train, X_exp_test, y_exp_train, y_exp_test = train_test_split(
        X_exp, y_exp, test_size=0.2, stratify=y_exp, random_state=random_state
    )

    # EXPLANATION TRAIN/VALIDATION SETS SPLIT
    X_exp_train, X_exp_val, y_exp_train, y_exp_val = train_test_split(
        X_exp_train,
        y_exp_train,
        test_size=0.2,
        stratify=y_exp_train,
        random_state=random_state,
    )

    if verbose:
        print("\nSHAPES:")
        print("BLACKBOX TRAINING SET: ", X_train.shape)
        print("BLACKBOX VALIDATION SET: ", X_val.shape)
        print("BLACKBOX TEST SET: ", X_test.shape)
        print("EXPLANATION TRAINING SET: ", X_exp_train.shape)
        print("EXPLANATION VALIDATION SET: ", X_exp_val.shape)
        print("EXPLANATION TEST SET: ", X_exp_test.shape)

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        X_exp_train,
        y_exp_train,
        X_exp_val,
        y_exp_val,
        X_exp_test,
        y_exp_test,
    )


def build_rnd_blobs(
    n_ts_per_blob=200, sz=80, d=1, n_blobs=6, random_state=0, verbose=True
):
    X_all, y_all = random_walk_blobs(
        n_ts_per_blob=n_ts_per_blob,
        sz=sz,
        d=d,
        n_blobs=n_blobs,
        random_state=random_state,
    )

    if verbose:
        print("DATASET INFO:")
        print("X SHAPE: ", X_all.shape)
        print("y SHAPE: ", y_all.shape)
        unique, counts = np.unique(y_all, return_counts=True)
        print("\nCLASSES BALANCE")
        for i, label in enumerate(unique):
            print(label, ": ", round(counts[i] / sum(counts), 2))

    # BLACKBOX/EXPLANATION SETS SPLIT
    X_train, X_exp, y_train, y_exp = train_test_split(
        X_all, y_all, test_size=0.3, stratify=y_all, random_state=random_state
    )

    # BLACKBOX TRAIN/TEST SETS SPLIT
    X_train, X_test, y_train, y_test = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=random_state
    )

    # BLACKBOX TRAIN/VALIDATION SETS SPLIT
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=random_state
    )

    # EXPLANATION TRAIN/TEST SETS SPLIT
    X_exp_train, X_exp_test, y_exp_train, y_exp_test = train_test_split(
        X_exp, y_exp, test_size=0.2, stratify=y_exp, random_state=random_state
    )

    # EXPLANATION TRAIN/VALIDATION SETS SPLIT
    X_exp_train, X_exp_val, y_exp_train, y_exp_val = train_test_split(
        X_exp_train,
        y_exp_train,
        test_size=0.2,
        stratify=y_exp_train,
        random_state=random_state,
    )

    if verbose:
        print("\nSHAPES:")
        print("BLACKBOX TRAINING SET: ", X_train.shape)
        print("BLACKBOX VALIDATION SET: ", X_val.shape)
        print("BLACKBOX TEST SET: ", X_test.shape)
        print("EXPLANATION TRAINING SET: ", X_exp_train.shape)
        print("EXPLANATION VALIDATION SET: ", X_exp_val.shape)
        print("EXPLANATION TEST SET: ", X_exp_test.shape)

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        X_exp_train,
        y_exp_train,
        X_exp_val,
        y_exp_val,
        X_exp_test,
        y_exp_test,
    )


def build_synth(path=None, verbose=True, random_state=0):
    if path is None:
        path = pathlib.Path(get_project_root() / "datasets" / "cached" / "synth")
    X_all = np.load(path / "X.npy")
    y_all = np.load(path / "y.npy").ravel()

    if verbose:
        print("DATASET INFO:")
        print("X SHAPE: ", X_all.shape)
        print("y SHAPE: ", y_all.shape)
        unique, counts = np.unique(y_all, return_counts=True)
        print("\nCLASSES BALANCE")
        for i, label in enumerate(unique):
            print(label, ": ", round(counts[i] / sum(counts), 2))

    # BLACKBOX/EXPLANATION SETS SPLIT
    X_train, X_exp, y_train, y_exp = train_test_split(
        X_all, y_all, test_size=0.3, stratify=y_all, random_state=random_state
    )

    # BLACKBOX TRAIN/TEST SETS SPLIT
    X_train, X_test, y_train, y_test = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=random_state
    )

    # BLACKBOX TRAIN/VALIDATION SETS SPLIT
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=random_state
    )

    # EXPLANATION TRAIN/TEST SETS SPLIT
    X_exp_train, X_exp_test, y_exp_train, y_exp_test = train_test_split(
        X_exp, y_exp, test_size=0.2, stratify=y_exp, random_state=random_state
    )

    # EXPLANATION TRAIN/VALIDATION SETS SPLIT
    X_exp_train, X_exp_val, y_exp_train, y_exp_val = train_test_split(
        X_exp_train,
        y_exp_train,
        test_size=0.2,
        stratify=y_exp_train,
        random_state=random_state,
    )

    if verbose:
        print("\nSHAPES:")
        print("BLACKBOX TRAINING SET: ", X_train.shape)
        print("BLACKBOX VALIDATION SET: ", X_val.shape)
        print("BLACKBOX TEST SET: ", X_test.shape)
        print("EXPLANATION TRAINING SET: ", X_exp_train.shape)
        print("EXPLANATION VALIDATION SET: ", X_exp_val.shape)
        print("EXPLANATION TEST SET: ", X_exp_test.shape)

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        X_exp_train,
        y_exp_train,
        X_exp_val,
        y_exp_val,
        X_exp_test,
        y_exp_test,
    )


def build_har(path=None, verbose=True, random_state=0):
    if path is None:
        path = get_project_root() / "datasets" / "cached" / "har"
    X_train = np.load(path / "X_train.npy")
    y_train = np.load(path / "y_train.npy").ravel()
    X_test = np.load(path / "X_test.npy")
    y_test = np.load(path / "y_test.npy").ravel()
    le = LabelEncoder()
    le.fit(y_train)
    y_train = le.transform(y_train)
    y_test = le.transform(y_test)

    if verbose:
        print("DATASET INFO:")
        print("Label Encoding:", le.classes_)
        print("X_train: ", X_train.shape)
        print("y_train: ", y_train.shape)
        unique, counts = np.unique(y_train, return_counts=True)
        print("\nCLASSES BALANCE")
        for i, label in enumerate(unique):
            print(label, ": ", round(counts[i] / sum(counts), 2))
        print()
        print("X_test: ", X_test.shape)
        print("y_test: ", y_test.shape)
        unique, counts = np.unique(y_test, return_counts=True)
        print("\nCLASSES BALANCE")
        for i, label in enumerate(unique):
            print(label, ": ", round(counts[i] / sum(counts), 2))

    X_exp = X_train.copy()
    y_exp = y_train.copy()
    if verbose:
        warnings.warn("Blackbox and Explanation sets are the same")

    return (
        X_train,
        y_train,
        None,
        None,
        X_test,
        y_test,
        X_exp,
        y_exp,
        None,
        None,
        X_test,
        y_test,
    )


def build_fsdd(
    path="./datasets/fsdd/", verbose=True, label_encoder=True, random_state=0
):
    df = pd.read_csv(path + "fsdd.csv", compression="gzip")

    X_all = (
        df[df["indexes"] >= 5].drop(["indexes", "name"], axis=1).reset_index(drop=True)
    )
    y_all = np.array(X_all["class"])
    X_all = X_all.drop("class", axis=1).values
    X_all = X_all.reshape((X_all.shape[0], X_all.shape[1], 1))

    X_test = (
        df[df["indexes"] < 5].drop(["indexes", "name"], axis=1).reset_index(drop=True)
    )
    y_test = np.array(X_test["class"])
    X_test = X_test.drop("class", axis=1).values
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    if verbose:
        print("X SHAPE: ", X_all.shape)
        print("y SHAPE: ", y_all.shape)
        unique, counts = np.unique(y_all, return_counts=True)
        print("\nCLASSES BALANCE")
        for i, label in enumerate(unique):
            print(label, ": ", round(counts[i] / sum(counts), 2))

    # BLACKBOX/EXPLANATION SETS SPLIT
    X_train, X_exp, y_train, y_exp = train_test_split(
        X_all, y_all, test_size=0.4, stratify=y_all, random_state=random_state
    )

    # BLACKBOX TRAIN/VALIDATION SETS SPLIT
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=random_state
    )

    # EXPLANATION TRAIN/TEST SETS SPLIT
    X_exp_train, X_exp_test, y_exp_train, y_exp_test = train_test_split(
        X_exp, y_exp, test_size=0.2, stratify=y_exp, random_state=random_state
    )

    # EXPLANATION TRAIN/VALIDATION SETS SPLIT
    X_exp_train, X_exp_val, y_exp_train, y_exp_val = train_test_split(
        X_exp_train,
        y_exp_train,
        test_size=0.2,
        stratify=y_exp_train,
        random_state=random_state,
    )

    if verbose:
        print("\nSHAPES:")
        print("BLACKBOX TRAINING SET: ", X_train.shape)
        print("BLACKBOX VALIDATION SET: ", X_val.shape)
        print("BLACKBOX TEST SET: ", X_test.shape)
        print("EXPLANATION TRAINING SET: ", X_exp_train.shape)
        print("EXPLANATION VALIDATION SET: ", X_exp_val.shape)
        print("EXPLANATION TEST SET: ", X_exp_test.shape)

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        X_exp_train,
        y_exp_train,
        X_exp_val,
        y_exp_val,
        X_exp_test,
        y_exp_test,
    )


def build_esr(path="./datasets/esr/", verbose=True, random_state=0):
    X = pd.read_csv(path + "data.csv", index_col=0)
    y = np.array(X["y"])
    y_all = np.ravel(y).astype("int")
    for i in range(2, 6):
        y_all[y_all == i] = 2
    le = LabelEncoder()
    le.fit(y_all)
    y_all = le.transform(y_all)
    X_all = X.drop("y", axis=1).values
    rus = RandomUnderSampler(
        random_state=random_state,
    )
    X_all, y_all = rus.fit_resample(X_all, y_all)
    X_all = X_all.reshape((X_all.shape[0], X_all.shape[1], 1))

    if verbose:
        print("X SHAPE: ", X_all.shape)
        print("y SHAPE: ", y_all.shape)
        unique, counts = np.unique(y_all, return_counts=True)
        print("\nCLASSES BALANCE")
        for i, label in enumerate(unique):
            print(label, ": ", round(counts[i] / sum(counts), 2))

    # BLACKBOX TRAIN/TEST SETS SPLIT
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, stratify=y_all, random_state=random_state
    )

    # BLACKBOX/EXPLANATION SETS SPLIT
    X_train, X_exp, y_train, y_exp = train_test_split(
        X_train, y_train, test_size=0.3, stratify=y_train, random_state=random_state
    )

    # BLACKBOX TRAIN/VALIDATION SETS SPLIT
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=random_state
    )

    # EXPLANATION TRAIN/TEST SETS SPLIT
    X_exp_train, X_exp_test, y_exp_train, y_exp_test = train_test_split(
        X_exp, y_exp, test_size=0.2, stratify=y_exp, random_state=random_state
    )

    # EXPLANATION TRAIN/VALIDATION SETS SPLIT
    X_exp_train, X_exp_val, y_exp_train, y_exp_val = train_test_split(
        X_exp_train,
        y_exp_train,
        test_size=0.2,
        stratify=y_exp_train,
        random_state=random_state,
    )

    if verbose:
        print("SHAPES:")
        print("BLACKBOX TRAINING SET: ", X_train.shape)
        print("BLACKBOX VALIDATION SET: ", X_val.shape)
        print("BLACKBOX TEST SET: ", X_test.shape)
        print("EXPLANATION TRAINING SET: ", X_exp_train.shape)
        print("EXPLANATION VALIDATION SET: ", X_exp_val.shape)
        print("EXPLANATION TEST SET: ", X_exp_test.shape)

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        X_exp_train,
        y_exp_train,
        X_exp_val,
        y_exp_val,
        X_exp_test,
        y_exp_test,
    )


def build_gunpoint(random_state=0, verbose=True, label_encoder=True):
    X_all, X_test, y_all, y_test = load_gunpoint(return_X_y=True)
    X_all = X_all[:, :, np.newaxis]
    X_test = X_test[:, :, np.newaxis]
    if label_encoder:
        le = LabelEncoder()
        le.fit(y_all)
        y_all = le.transform(y_all)
        y_test = le.transform(y_test)

    if verbose:
        print("DATASET INFO:")
        print("X SHAPE: ", X_all.shape)
        print("y SHAPE: ", y_all.shape)
        unique, counts = np.unique(y_all, return_counts=True)
        print("\nCLASSES BALANCE")
        for i, label in enumerate(unique):
            print(label, ": ", round(counts[i] / sum(counts), 2))

    # BLACKBOX/EXPLANATION SETS SPLIT
    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y_all, test_size=0.2, stratify=y_all, random_state=random_state
    )

    X_exp_train = X_train.copy()
    y_exp_train = y_train.copy()
    X_exp_val = X_val.copy()
    y_exp_val = y_val.copy()
    X_exp_test = X_test.copy()
    y_exp_test = y_test.copy()

    warnings.warn("Blackbox and Explanation sets are the same")

    if verbose:
        print("\nSHAPES:")
        print("BLACKBOX TRAINING SET: ", X_train.shape)
        print("BLACKBOX VALIDATION SET: ", X_val.shape)
        print("BLACKBOX TEST SET: ", X_test.shape)
        print("EXPLANATION TRAINING SET: ", X_exp_train.shape)
        print("EXPLANATION VALIDATION SET: ", X_exp_val.shape)
        print("EXPLANATION TEST SET: ", X_exp_test.shape)
        print("\nBlackbox and Explanation sets are the same!")

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        X_exp_train,
        y_exp_train,
        X_exp_val,
        y_exp_val,
        X_exp_test,
        y_exp_test,
    )


def build_coffee(random_state=0, verbose=True, label_encoder=True):
    X_train, X_test, y_train, y_test = load_coffee(return_X_y=True)
    X_train = X_train[:, :, np.newaxis]
    X_test = X_test[:, :, np.newaxis]
    if label_encoder:
        le = LabelEncoder()
        le.fit(y_train)
        y_train = le.transform(y_train)
        y_test = le.transform(y_test)

    if verbose:
        print("DATASET INFO:")
        print("X SHAPE: ", X_train.shape)
        print("y SHAPE: ", y_train.shape)
        unique, counts = np.unique(y_train, return_counts=True)
        print("\nCLASSES BALANCE")
        for i, label in enumerate(unique):
            print(label, ": ", round(counts[i] / sum(counts), 2))

    X_exp_train = X_train.copy()
    y_exp_train = y_train.copy()
    X_exp_val = X_val = np.array(list())
    y_exp_val = y_val = np.array(list())
    X_exp_test = X_test.copy()
    y_exp_test = y_test.copy()

    warnings.warn(
        "The validation sets are empty, use cross-validation to evaluate models"
    )
    warnings.warn("Blackbox and Explanation sets are the same")

    if verbose:
        print("\nSHAPES:")
        print("BLACKBOX TRAINING SET: ", X_train.shape)
        print("BLACKBOX VALIDATION SET: ", X_val.shape)
        print("BLACKBOX TEST SET: ", X_test.shape)
        print("EXPLANATION TRAINING SET: ", X_exp_train.shape)
        print("EXPLANATION VALIDATION SET: ", X_exp_val.shape)
        print("EXPLANATION TEST SET: ", X_exp_test.shape)

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        X_exp_train,
        y_exp_train,
        X_exp_val,
        y_exp_val,
        X_exp_test,
        y_exp_test,
    )


def build_ecg200(
    path="./datasets/ECG200/", random_state=0, verbose=True, label_encoder=True
):
    X_all = pd.read_csv(path + "ECG200_TRAIN.txt", sep="\s+", header=None)
    y_all = np.array(X_all[0])
    X_all = np.array(X_all.drop([0], axis=1))
    X_all = X_all[:, :, np.newaxis]

    X_test = pd.read_csv(path + "ECG200_TEST.txt", sep="\s+", header=None)
    y_test = np.array(X_test[0])
    X_test = np.array(X_test.drop([0], axis=1))
    X_test = X_test[:, :, np.newaxis]

    if label_encoder:
        le = LabelEncoder()
        le.fit(y_all)
        y_all = le.transform(y_all)
        y_test = le.transform(y_test)

    if verbose:
        print("DATASET INFO:")
        print("X SHAPE: ", X_all.shape)
        print("y SHAPE: ", y_all.shape)
        unique, counts = np.unique(y_all, return_counts=True)
        print("\nCLASSES BALANCE")
        for i, label in enumerate(unique):
            print(label, ": ", round(counts[i] / sum(counts), 2))

    # BLACKBOX/EXPLANATION SETS SPLIT
    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y_all, test_size=0.2, stratify=y_all, random_state=random_state
    )

    X_exp_train = X_train.copy()
    y_exp_train = y_train.copy()
    X_exp_val = X_val.copy()
    y_exp_val = y_val.copy()
    X_exp_test = X_test.copy()
    y_exp_test = y_test.copy()

    warnings.warn("Blackbox and Explanation sets are the same")

    if verbose:
        print("\nSHAPES:")
        print("BLACKBOX TRAINING SET: ", X_train.shape)
        print("BLACKBOX VALIDATION SET: ", X_val.shape)
        print("BLACKBOX TEST SET: ", X_test.shape)
        print("EXPLANATION TRAINING SET: ", X_exp_train.shape)
        print("EXPLANATION VALIDATION SET: ", X_exp_val.shape)
        print("EXPLANATION TEST SET: ", X_exp_test.shape)

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        X_exp_train,
        y_exp_train,
        X_exp_val,
        y_exp_val,
        X_exp_test,
        y_exp_test,
    )


def build_EthanolConcentration(verbose=True):
    X_train, X_test, y_train, y_test = fetch_uea_dataset(
        "EthanolConcentration", return_X_y=True
    )
    # X_train = X_train.transpose(0, 2, 1)
    # X_test = X_test.transpose(0, 2, 1)
    X_test = minmax_scale(
        X_test.transpose(0, 2, 1), minimum=X_train.min(), maximum=X_train.max()
    )
    X_train = minmax_scale(X_train.transpose(0, 2, 1))
    le = LabelEncoder()
    le.fit(y_train)
    y_train = le.transform(y_train)
    y_test = le.transform(y_test)
    if verbose:
        print("DATASET INFO:")
        print("Label Encoding:", le.classes_)
        print("X_train: ", X_train.shape)
        print("y_train: ", y_train.shape)
        unique, counts = np.unique(y_train, return_counts=True)
        print("\nCLASSES BALANCE")
        for i, label in enumerate(unique):
            print(label, ": ", round(counts[i] / sum(counts), 2))
        print()
        print("X_test: ", X_test.shape)
        print("y_test: ", y_test.shape)
        unique, counts = np.unique(y_test, return_counts=True)
        print("\nCLASSES BALANCE")
        for i, label in enumerate(unique):
            print(label, ": ", round(counts[i] / sum(counts), 2))

    X_exp = X_train.copy()
    y_exp = y_train.copy()
    warnings.warn("Blackbox and Explanation sets are the same")

    return (
        X_train,
        y_train,
        None,
        None,
        X_test,
        y_test,
        X_exp,
        y_exp,
        None,
        None,
        X_test,
        y_test,
    )


def build_ArticularyWordRecognition(verbose=True):
    X_train, X_test, y_train, y_test = fetch_uea_dataset(
        "ArticularyWordRecognition", return_X_y=True
    )
    X_train = X_train.transpose(0, 2, 1)
    X_test = X_test.transpose(0, 2, 1)
    le = LabelEncoder()
    le.fit(y_train)
    y_train = le.transform(y_train)
    y_test = le.transform(y_test)
    if verbose:
        print("DATASET INFO:")
        print("Label Encoding:", le.classes_)
        print("X_train: ", X_train.shape)
        print("y_train: ", y_train.shape)
        unique, counts = np.unique(y_train, return_counts=True)
        print("\nCLASSES BALANCE")
        for i, label in enumerate(unique):
            print(label, ": ", round(counts[i] / sum(counts), 2))
        print()
        print("X_test: ", X_test.shape)
        print("y_test: ", y_test.shape)
        unique, counts = np.unique(y_test, return_counts=True)
        print("\nCLASSES BALANCE")
        for i, label in enumerate(unique):
            print(label, ": ", round(counts[i] / sum(counts), 2))

    X_exp = X_train.copy()
    y_exp = y_train.copy()
    warnings.warn("Blackbox and Explanation sets are the same")

    return (
        X_train,
        y_train,
        None,
        None,
        X_test,
        y_test,
        X_exp,
        y_exp,
        None,
        None,
        X_test,
        y_test,
    )


def build_FingerMovements(verbose=True):
    X_train, X_test, y_train, y_test = fetch_uea_dataset(
        "FingerMovements", return_X_y=True
    )
    X_train = X_train.transpose(0, 2, 1)
    X_test = X_test.transpose(0, 2, 1)
    le = LabelEncoder()
    le.fit(y_train)
    y_train = le.transform(y_train)
    y_test = le.transform(y_test)
    if verbose:
        print("DATASET INFO:")
        print("Label Encoding:", le.classes_)
        print("X_train: ", X_train.shape)
        print("y_train: ", y_train.shape)
        unique, counts = np.unique(y_train, return_counts=True)
        print("\nCLASSES BALANCE")
        for i, label in enumerate(unique):
            print(label, ": ", round(counts[i] / sum(counts), 2))
        print()
        print("X_test: ", X_test.shape)
        print("y_test: ", y_test.shape)
        unique, counts = np.unique(y_test, return_counts=True)
        print("\nCLASSES BALANCE")
        for i, label in enumerate(unique):
            print(label, ": ", round(counts[i] / sum(counts), 2))

    X_exp = X_train.copy()
    y_exp = y_train.copy()
    warnings.warn("Blackbox and Explanation sets are the same")

    return (
        X_train,
        y_train,
        None,
        None,
        X_test,
        y_test,
        X_exp,
        y_exp,
        None,
        None,
        X_test,
        y_test,
    )


def build_ERing(verbose=True):
    X_train, y_train = load_from_tsfile_to_dataframe(
        get_project_root() / "datasets" / "cached" / "ERing" / "ERing_TRAIN.ts"
    )
    X_test, y_test = load_from_tsfile_to_dataframe(
        get_project_root() / "datasets" / "cached" / "ERing" / "ERing_TEST.ts"
    )
    X_train = convert_sktime_to_numpy(X_train)
    X_test = convert_sktime_to_numpy(X_test)
    le = LabelEncoder()
    le.fit(y_train)
    y_train = le.transform(y_train)
    y_test = le.transform(y_test)
    if verbose:
        print("DATASET INFO:")
        print("Label Encoding:", le.classes_)
        print("X_train: ", X_train.shape)
        print("y_train: ", y_train.shape)
        unique, counts = np.unique(y_train, return_counts=True)
        print("\nCLASSES BALANCE")
        for i, label in enumerate(unique):
            print(label, ": ", round(counts[i] / sum(counts), 2))
        print()
        print("X_test: ", X_test.shape)
        print("y_test: ", y_test.shape)
        unique, counts = np.unique(y_test, return_counts=True)
        print("\nCLASSES BALANCE")
        for i, label in enumerate(unique):
            print(label, ": ", round(counts[i] / sum(counts), 2))

    X_exp = X_train.copy()
    y_exp = y_train.copy()
    warnings.warn("Blackbox and Explanation sets are the same")

    return (
        X_train,
        y_train,
        None,
        None,
        X_test,
        y_test,
        X_exp,
        y_exp,
        None,
        None,
        X_test,
        y_test,
    )


def build_LSST(verbose=True):
    data_loader = UCR_UEA_datasets()
    X_train, y_train, X_test, y_test = data_loader.load_dataset("LSST")
    X_test = minmax_scale(X_test, minimum=X_train.min(), maximum=X_train.max())
    X_train = minmax_scale(X_train)
    le = LabelEncoder()
    le.fit(y_train)
    y_train = le.transform(y_train)
    y_test = le.transform(y_test)
    if verbose:
        print("DATASET INFO:")
        print("Label Encoding:", le.classes_)
        print("X_train: ", X_train.shape)
        print("y_train: ", y_train.shape)
        unique, counts = np.unique(y_train, return_counts=True)
        print("\nCLASSES BALANCE")
        for i, label in enumerate(unique):
            print(label, ": ", round(counts[i] / sum(counts), 2))
        print()
        print("X_test: ", X_test.shape)
        print("y_test: ", y_test.shape)
        unique, counts = np.unique(y_test, return_counts=True)
        print("\nCLASSES BALANCE")
        for i, label in enumerate(unique):
            print(label, ": ", round(counts[i] / sum(counts), 2))

    X_exp = X_train.copy()
    y_exp = y_train.copy()
    warnings.warn("Blackbox and Explanation sets are the same")

    return (
        X_train,
        y_train,
        None,
        None,
        X_test,
        y_test,
        X_exp,
        y_exp,
        None,
        None,
        X_test,
        y_test,
    )


if __name__ == "__main__":
    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        X_exp_train,
        y_exp_train,
        X_exp_val,
        y_exp_val,
        X_exp_test,
        y_exp_test,
    ) = load_ucr_dataset("TwoLeadECG")
    # from lasts.datasets.loader import dataset_loader
    # from lasts.blackboxes.loader import cached_blackbox_loader
    # X_train, y_train, X_test, y_test = dataset_loader("Libras")()
    # clf = cached_blackbox_loader("Libras_rocket.joblib")

    # from lasts.utils import make_path
    #
    # name = "Libras"
    # data_loader = UCR_UEA_datasets()
    # X_train, y_train, X_test, y_test = data_loader.load_dataset(name)
    # y_train = y_train.astype("int")
    # y_test = y_test.astype("int")
    #
    #
    # print(name, X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    #
    # folder = make_path(get_project_root() / "datasets" / "cached" / name)
    # # np.save(folder / "X_train.npy", X_train)
    # # np.save(folder / "X_test.npy", X_test)
    # np.save(folder / "y_train.npy", y_train)
    # np.save(folder / "y_test.npy", y_test)
