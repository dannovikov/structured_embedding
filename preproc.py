from Bio import SeqIO
import numpy as np
import pickle
import torch
from tqdm import tqdm

fasta_file = "data/sequences.fasta"

# Read in the fasta file
seqs = {}
max_len = 0
for record in SeqIO.parse(fasta_file, "fasta"):
    seqs[record.id] = record.seq
    if len(record.seq) > max_len:
        max_len = len(record.seq)

# Create integer encoding tensor, and a dictionary mapping ids to rows in the matrix.
N = len(seqs.keys())
L = max_len
map_row_to_seqid = {}
X = torch.zeros((N, L))

for row, (id, seq) in tqdm(enumerate(seqs.items()), desc="Integer Encoding", total=len(seqs.keys())):
    map_row_to_seqid[row] = id
    for col, nucl in enumerate(seq):
        nucl = nucl.upper()
        if nucl not in ["A", "C", "G", "T"]:
            X[row][col] = 4
        elif nucl == "A":
            X[row][col] = 0
        elif nucl == "C":
            X[row][col] = 1
        elif nucl == "G":
            X[row][col] = 2
        elif nucl == "T":
            X[row][col] = 3

# create the labels tensor
labels = {}
for id in seqs.keys():
    subtype = id.split(".")[0]
    labels[id] = subtype

# create pytorch tensor from the labels
y = torch.zeros(N)
subtypes = list(set(labels.values()))
for i, key in enumerate(seqs.keys()):
    y[i] = subtypes.index(labels[key])

# dictinoary mapping label id integers to subtype strings
map_label_to_subtype = {i: subtypes[i] for i in range(len(subtypes))}

print(X)
print(y)


torch.save(X, "data/X.pt")
torch.save(y, "data/y.pt")
with open("data/map_label_to_subtype.pkl", "wb") as f:
    pickle.dump(map_label_to_subtype, f)

with open("data/map_row_to_seqid.pkl", "wb") as f:
    pickle.dump(map_row_to_seqid, f)
