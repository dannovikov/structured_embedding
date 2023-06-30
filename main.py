import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pickle
import wandb

wandb.init(project="seqvae")
from tqdm import tqdm

EPOCHS = 20
BATCH_SIZE = 4
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "./data"
EMBED_DIM = 32
NUM_WORKERS = 8

class SeqVAE(nn.Module):
    def __init__(
        self,
        nucl_alphabet_size=5,
        nucl_embedding_dim=EMBED_DIM,
        n_embedding_heads=8,
        n_embedding_layers=16,
        hidden_dim=256,
        latent_dim=2,
    ):
        super().__init__()
        # For embedding the sequences (each given as a integer encoded L-vector) into a continuous space
        self.embedding = nn.Embedding(nucl_alphabet_size, nucl_embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=nucl_embedding_dim,
            nhead=n_embedding_heads,
            dim_feedforward=hidden_dim,
            activation="relu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_embedding_layers)

        # Rather than encoding to a single point, VAE encodes to a gaussian of points centered at a certain location in the latent space
        # This gaussian is represented by its parameters, the mean vector mu and diagonal covariance matrix sigma.
        self.mu = nn.Linear(nucl_embedding_dim, latent_dim)
        # By tradition, we learn the log of sigma instead of sigma itself (makes math easier a little easier) (https://stats.stackexchange.com/a/486205)
        self.logvar = nn.Linear(nucl_embedding_dim, latent_dim)

        # For decoding the latent space back into the original embedding space
        self.latent_to_embedding = nn.Linear(latent_dim, nucl_embedding_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=nucl_embedding_dim,
            nhead=n_embedding_heads,
            dim_feedforward=hidden_dim,
            activation="relu",
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_embedding_layers)

        # For decoding the embedding space back into the original sequence space
        self.embedding_to_nucl = nn.Linear(nucl_embedding_dim, 1)

    def reparameterize(self, mu, logvar):
        # We sample from the gaussian using the reparameterization trick
        # https://stats.stackexchange.com/a/16338
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # x is a batch of sequences, each represented as a 4xL one-hot matrix
        # We first embed the sequences into a continuous space
        # print(x.shape, "original")
        x = self.embedding(x)
        # print(x.shape)
        x = self.encoder(x)
        # print(x.shape, "after encoder")
        # We then estimate the parameters of the gaussian in the latent space
        mu = self.mu(x)
        logvar = self.logvar(x)

        # Now sample this gaussian to get points in the latent space
        z = self.reparameterize(mu, logvar)

        # Decode the samples back into the original sequence space
        # print(z.shape, "new sample")
        z = self.latent_to_embedding(z)
        # print(z.shape, "embedding sized")
        x = self.decoder(tgt=z, memory=x)
        # print(x.shape, "decoded")
        x = self.embedding_to_nucl(x).squeeze()
        # print(x.shape, "nucl sized")
        return x, mu, logvar

    def generate(self, z):
        # z is a batch of points in the latent space
        # We decode the latent space back into the original embedding space
        z = self.latent_to_embedding(z)
        # We then decode the embedding space back into the original sequence space
        x = self.decoder(z)
        x = self.embedding_to_nucl(x)
        return x

    def encode(self, x):
        x = self.embedding(x)
        x = self.encoder(x)
        mu = self.mu(x)
        logvar = self.logvar(x)
        z = self.reparameterize(mu, logvar)
        return z


class KLDivergenceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.long()
        self.y = y

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def main():
    data = get_data()
    model = SeqVAE()
    model, train_stats = train(model, data)
    save(model)
    plot_latent_space(model, data)
    new_seqs = generate(model, data)
    for seq in new_seqs:
        print(seq)
    # validation = evaluate(new_seqs, data)


def get_data():
    X = torch.load(f"{DATA_DIR}/X.pt")
    y = torch.load(f"{DATA_DIR}/y.pt")
    with open(f"{DATA_DIR}/map_label_to_subtype.pkl", "rb") as f:
        map_label_to_subtype = pickle.load(f)
    with open(f"{DATA_DIR}/map_row_to_seqid.pkl", "rb") as f:
        map_row_to_seqid = pickle.load(f)
    result = {
        "X": X,
        "y": y,
        "map_label_to_subtype": map_label_to_subtype,
        "map_row_to_seqid": map_row_to_seqid,
    }
    return result


def train(model, data):
    """
    model is the initialized model
    data is a dictionary where keys X and y contain the data and label tensors respectively,
    and two more keys for metadata dicts.
    """
    model = model.to(DEVICE)
    train_loader, test_loader = _get_data_loaders(data)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    recon_loss = nn.MSELoss()
    kld_loss = KLDivergenceLoss()
    # train_stats = {"loss": [], "accuracy": []}
    for epoch in range(EPOCHS):
        avg_acc = 0
        for i, (x, _) in tqdm(enumerate(train_loader), desc=f"Epoch {epoch}", total=len(train_loader)):
            model.train()
            x = x.to(DEVICE)
            optimizer.zero_grad()
            x_pred, mu, logvar = model(x)
            # print(f"{x_pred.shape=}, {x.shape=}")
            loss = recon_loss(x_pred, x.float()) + kld_loss(mu, logvar)
            loss.backward()
            optimizer.step()
            # train_stats["loss"].append(loss.item())
            model.eval()
            try:
                with torch.no_grad():
                    acc = _get_accuracy(x_pred, x.float())
                    avg_acc += acc
            except:
                pass
            wandb.log({"loss": loss.item()})
        # train_stats["accuracy"].append(avg_acc / len(train_loader))
        wandb.log({"accuracy": avg_acc / len(train_loader)})
        


        model.eval()
        with torch.no_grad():
            avg_acc = 0
            for i, (x, _) in enumerate(test_loader):
                x = x.to(DEVICE)
                x_pred, mu, logvar = model(x)
                loss = recon_loss(x_pred, x.float()) + kld_loss(mu, logvar)
                try:
                    acc = _get_accuracy(x_pred, x.float())
                    avg_acc += acc
                except:
                    pass
                wandb.log({"val_loss": loss.item()})
            wandb.log({"val_accuracy": avg_acc / len(test_loader)})
    # return model, train_stats
    return model, None


def plot_latent_space(model, data):
    pass


def generate(model, data):
    # for each sequence, generate 10 new similar sequences
    model.eval()
    new_seqs = []
    for seq in data["X"]:
        seq = seq.to(DEVICE)
        z = model.encode(seq)
        genned_seqs = {seq: []}
        for i in range(10):
            new_seq = model.generate(z)
            genned_seqs[seq].append(new_seq)
        new_seqs.append(genned_seqs)
    return new_seqs


def evaluate(new_seqs, data):
    pass


def save(model):
    torch.save(model.state_dict(), "saved_models/model.pt")


def _get_data_loaders(data):
    """
    data['X'] and ['y'] are the data and label tensors respectively
    Create datasets and dataloaders for training and testing
    """
    dataset = SeqDataset(data["X"], data["y"])
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers = NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True,num_workers = NUM_WORKERS)
    return train_loader, test_loader


def _get_accuracy(x_pred, x):
    """
    x_pred is the predicted sequence
    x is the true sequence
    Compute their hamming distance.
    return 1- normalized hamming distance
    """

    # return sum of all times the predicted sequence is not equal to the true sequence
    # TODO: test
    # return 1 - (x_pred != x).sum() / x.shape[0]
    corr = 0
    inco = 0
    tota = 0
    for seq in range(x.shape[0]):
        for i in range(x.shape[1]):
            if int(x[seq][i]) == int(x_pred[seq][i]):
                corr += 1
            else:
                inco += 1
            tota += 1
    assert corr + inco == tota
    return corr / tota


if __name__ == "__main__":
    main()
