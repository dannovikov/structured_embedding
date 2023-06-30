from Bio import SeqIO
import numpy as np
import pickle
import torch

fasta_file = "data/sequences.fasta"

# Read in the fasta file
seqs = {}
for record in SeqIO.parse(fasta_file, "fasta"):
    seqs[record.id] = record.seq

# create one-hot encoding of length 4*L for each sequence
ohe_seqs = {}
for key in seqs.keys():
    seq = seqs[key]
    ohe_seq = np.zeros((4, len(seq)))
    for i in range(len(seq)):
        if seq[i] == "A":
            ohe_seq[0, i] = 1
        elif seq[i] == "C":
            ohe_seq[1, i] = 1
        elif seq[i] == "G":
            ohe_seq[2, i] = 1
        elif seq[i] == "T":
            ohe_seq[3, i] = 1
    ohe_seqs[key] = ohe_seq

# create integer encoding of length L for each sequence
int_seqs = {}
for key in seqs.keys():
    seq = seqs[key]
    int_seq = np.zeros((len(seq)))
    for i in range(len(seq)):
        if seq[i] == "A":
            int_seq[i] = 1
        elif seq[i] == "C":
            int_seq[i] = 2
        elif seq[i] == "G":
            int_seq[i] = 3
        elif seq[i] == "T":
            int_seq[i] = 4
        else:
            int_seq[i] = 5
    int_seqs[key] = int_seq


def handle_ohe(ohe_seqs):
    # create labels dictionary by extracting the subtypes from the sequence ids
    labels = {}
    for key in seqs.keys():
        subtype = key.split(".")[0]
        labels[key] = subtype

    # create pytorch tensors from the one-hot encodings
    N = len(ohe_seqs.keys())
    L = 0
    for key in ohe_seqs.keys():
        s = ohe_seqs[key]
        if s.shape[1] > L:
            L = s.shape[1]
    X = torch.zeros((N, 4, L))
    print(X.shape)
    for i, (key, s) in enumerate(ohe_seqs.items()):
        padded_seq = torch.zeros((4, L))
        padded_seq[:, : s.shape[1]] = torch.from_numpy(s)
        X[i, :, :] = padded_seq

    # create pytorch tensor from the labels using integer encoding
    y = torch.zeros(N)
    subtypes = list(set(labels.values()))
    for i, key in enumerate(ohe_seqs.keys()):
        y[i] = subtypes.index(labels[key])

    subtype_labels_map = {i: subtypes[i] for i in range(len(subtypes))}
    matrix_row_id_map = {i: key for i, key in enumerate(ohe_seqs.keys())}

    print(X)
    print(y)
    # print(subtype_labels_map)
    # print(matrix_row_id_map)

    # save the tensors and labels to disk
    torch.save(X, "data/X.pt")
    torch.save(y, "data/y.pt")
    with open("data/subtype_labels_map.pkl", "wb") as f:
        pickle.dump(subtype_labels_map, f)

    with open("data/matrix_row_id_map.pkl", "wb") as f:
        pickle.dump(matrix_row_id_map, f)


def handle_int(int_seqs):
    # create labels dictionary by extracting the subtypes from the sequence ids
    labels = {}
    for key in seqs.keys():
        subtype = key.split(".")[0]
        labels[key] = subtype

    # create pytorch tensors from the one-hot encodings
    N = len(int_seqs.keys())
    L = 0
    for key in int_seqs.keys():
        s = int_seqs[key]
        if s.shape[0] > L:
            L = s.shape[0]
    X = torch.zeros((N, L))
    print(X.shape)
    for i, (key, s) in enumerate(int_seqs.items()):
        padded_seq = torch.zeros((L))
        padded_seq[: s.shape[0]] = torch.from_numpy(s)
        X[i, :] = padded_seq

    # create pytorch tensor from the labels using integer encoding
    y = torch.zeros(N)
    subtypes = list(set(labels.values()))
    for i, key in enumerate(int_seqs.keys()):
        y[i] = subtypes.index(labels[key])

    subtype_labels_map = {i: subtypes[i] for i in range(len(subtypes))}
    matrix_row_id_map = {i: key for i, key in enumerate(int_seqs.keys())}

    print(X)
    print(y)
    # print(subtype_labels_map)
    # print(matrix_row_id_map)

    # save the tensors and labels to disk
    torch.save(X, "data/X.pt")
    torch.save(y, "data/y.pt")
    with open("data/subtype_labels_map.pkl", "wb") as f:
        pickle.dump(subtype_labels_map, f)

    with open("data/matrix_row_id_map.pkl", "wb") as f:
        pickle.dump(matrix_row_id_map, f)


handle_int(int_seqs)
