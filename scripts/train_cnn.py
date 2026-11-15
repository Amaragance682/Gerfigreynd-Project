import argparse 
import time
from pathlib import Path
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1] # jumps up one folder into the root for relative file addresses
DATA = ROOT / "data"
FS = 64 # samples/second
N_SAMPLES = 901 * FS # the split which accounts for 0-901 seconds
WIN = 16*FS # samples per CNN input window

def load_split(split, device): 
    seg = torch.load(DATA / f"segments_h1k_{split}.pt", weights_only=True)
    ids = [t[4].removesuffix(".edf") for t in seg] # id assignment
    ages = torch.tensor([t[3] for t in seg], dtype=torch.float32)
    X = torch.empty(len(ids), 20, N_SAMPLES, dtype=torch.float16) # each crop is converted to float32 but float16 halves the memory
    lengths = torch.empty(len(ids), dtype=torch.long)
    for i, rec_id in enumerate(ids): # for every id we load in rec_id and iterate over ids.length
        x = torch.load(DATA / "pre_processed" / f"{rec_id}.pt", weights_only=True)[:, :N_SAMPLES]
        X[i] = x
        lengths[i] = int((x != 0).any(dim=0).nonzero()[-1]) + 1 # last sample where any channel is non-zero
    return ids, X.to(device), lengths.to(device), ages.to(device)


def global_scale(X, lengths):
    # root mean square error over the signal of every recording (42.41 µV)
    # only ever computed on train and is used to divide every split, running it on val/test would leak a bit of info to train
    total, n = 0.0, 0
    for x, length in zip(X, lengths.tolist()):
        total += x[:, :length].double().square().sum().item()
        n += x.shape[0] * length
    return (total / n) ** 0.5

def random_starts(lengths, idx, generator):
    # A random start for each recording in `idx`, so that the window ends inside the real signal. 
    # This prevents overfitting due to an epoch seeing a different 16 s piece of each child instead of the same one
    # we also seed this with 'generator' for report reproducability
    u = torch.rand(len(idx), device=lengths.device, generator=generator)
    return (u * (lengths[idx] - WIN + 1)).long()

def crop(X, idx, starts):
    # three index tensors, broadcast to (batch, channels, WIN). This function allows me to process a whole batch in one GPU operation
    # The WIN samples from starts in each recording in idx. Format: (len(idx), 20, WIN).
    rows = idx[:, None, None] # which recording are we targeting?
    chans = torch.arange(X.shape[1], device=X.device)[None, :, None] # all 20 channels
    times = (starts[:, None] + torch.arange(WIN, device=X.device))[:, None, :]
    return X[rows, chans, times]

def epoch_batches(n_children, crops_per_child, batch_size, generator):
    # Child indices for one epoch: each child `crops_per_child` times, shuffled, in batches.
    order = torch.arange(n_children, device=generator.device).repeat(crops_per_child)
    order = order[torch.randperm(len(order), device=generator.device, generator=generator)]
    return order.split(batch_size)

def eval_windows(X, lengths):
    # All non-overlapping WIN-sample windows of each recording's real signal, and whose they are.
    windows, owner = [], []
    for i, length in enumerate(lengths.tolist()):
        k = length // WIN
        windows.append(X[i, :, : k * WIN].reshape(X.shape[1], k, WIN).transpose(0, 1))
        owner.append(torch.full((k,), i, device=X.device))
    return torch.cat(windows), torch.cat(owner)



class ConvBlock(nn.Module):
    
    def __init__(self, n_in, n_out):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(n_in, n_out, (1, 3), padding=(0, 1)), nn.BatchNorm2d(n_out), nn.ReLU(),
            nn.Conv2d(n_out, n_out, (1, 3), padding=(0, 1)), nn.BatchNorm2d(n_out), nn.ReLU(),
            nn.MaxPool2d((1, 2)),
        )
    def forward(self, x):
        return self.layers(x)



# sample figure 3 for the project:

class BrainAgeCNN(nn.Module):
    # figure 3 of the assignment along with a linear head that predicts log(age):
    #  (B, 20, WIN) -> 3 ConvBlocks -> fully connected layer per channel -> max over channels -> z -> log(age)

    def __init__(self, widths=(16, 32, 64), n_features=64, dropout=0.3):
        super().__init__()
        blocks, n_in = [], 1
        for w in widths:
            blocks.append(ConvBlock(n_in, w))
            n_in = w
        self.blocks = nn.Sequential(*blocks)
        t_out = WIN // 2 ** len(widths)

        self.fc = nn.Sequential(nn.Conv2d(n_in, n_features, (1, t_out)), nn.ReLU())
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(n_features, 1))

    def features(self, x):
        x = self.blocks(x.unsqueeze(1))     # (B, 1, 20, WIN) -> (B, widths[-1], 20, t_out)
        x = self.fc(x)                      # (B, n_features, 20, 1)
        return x.amax(dim=(2, 3))           # MaxPool2D (C, 1) in figure 3: max over the 20 channels

    def forward(self, x):
        # (B, 20, WIN) -> predicted log(age), (B,)
        return self.head(self.features(x)).squeeze(1) 










# A.I. generated data checker pre model training, sanity checks, overfit checks, add any checks here in the future related to model training
def check_data(device):
    # Shape and content checks for the loader (plan task T11).
    t0 = time.time()
    ids, X, lengths, ages = load_split("train", device)
    print(f"train: X {tuple(X.shape)} {X.dtype} on {X.device}, {X.numel() * 2 / 1e9:.2f} GB, "
          f"loaded in {time.time() - t0:.0f} s")
    print(f"real signal: {lengths.min().item() / FS:.0f}-{lengths.max().item() / FS:.0f} s, "
          f"{(lengths < N_SAMPLES).sum().item()} recordings end in zero padding")

    labels = pd.read_csv(DATA / "anon_codex_age_sex.csv", index_col="Anon. Filename")
    csv_ages = torch.tensor(labels.loc[[f"{i}.edf" for i in ids], "Quantized Age (years)"].to_numpy(),
                            dtype=torch.float32, device=device)
    assert torch.allclose(ages, csv_ages), "split ages differ from the label file"
    print(f"ages match the label file; {ages.min().item():.2f}-{ages.max().item():.2f} years")

    scale = global_scale(X, lengths)
    print(f"global scale (train RMS): {scale:.2f} µV")

    gen = torch.Generator(device=device).manual_seed(0)
    batches = epoch_batches(len(ids), crops_per_child=8, batch_size=64, generator=gen)
    idx = batches[0]
    starts = random_starts(lengths, idx, gen)
    crops = crop(X, idx, starts).float() / scale
    print(f"one epoch = {len(batches)} batches; batch {tuple(crops.shape)} {crops.dtype}, "
          f"std {crops.std().item():.2f}, targets log(age) {tuple(torch.log(ages[idx]).shape)}")

    # Each crop must be the right slice of its own recording, inside the non-padded part.
    assert (starts + WIN <= lengths[idx]).all(), "a crop runs into the zero padding"
    for b, (i, s) in enumerate(zip(idx.tolist(), starts.tolist())):
        assert torch.equal(crop(X, idx[b:b + 1], starts[b:b + 1])[0], X[i, :, s:s + WIN]), f"crop {b} is wrong"
    counts = torch.bincount(torch.cat(batches), minlength=len(ids))
    assert (counts == 8).all(), "not every child appears 8 times per epoch"
    print("crops are the right slices, inside the real signal; every child appears 8 times")

    if device.type == "cuda": # my own system specific method
        torch.cuda.synchronize()
    t0 = time.time()
    for idx in batches:
        crop(X, idx, random_starts(lengths, idx, gen)).float() / scale
    if device.type == "cuda":
        torch.cuda.synchronize()
    print(f"cropping a whole epoch takes {time.time() - t0:.2f} s")

    _, Xv, lv, _ = load_split("val", device)
    windows, owner = eval_windows(Xv, lv)
    assert len(windows) == (lv // WIN).sum().item()
    assert torch.equal(windows[0], Xv[0, :, :WIN]) and torch.equal(windows[1], Xv[0, :, WIN:2 * WIN])
    last = (owner == len(lv) - 1).nonzero()[-1].item()
    k = (lv[-1] // WIN).item()
    assert torch.equal(windows[last], Xv[-1, :, (k - 1) * WIN:k * WIN])
    print(f"val: {len(lv)} children, {len(windows)} eval windows {tuple(windows.shape[1:])}, "
          f"{len(windows) / len(lv):.1f} per child; windows match the recordings")
    if device.type == "cuda":
        print(f"GPU memory in use: {torch.cuda.memory_allocated() / 1e9:.2f} GB")


# forward + backward pass checks for the CNN and time recording per training step
def check_model(device):
    torch.manual_seed(0) # manual seeding for reproducability
    model = BrainAgeCNN().to(device)
    for name, part in [("conv blocks", model.blocks), ("fully connected", model.fc), ("head", model.head)]:
        print(f"{name}: {sum(p.numel() for p in part.parameters()):,} parameters")

    x = torch.randn(64, 20, WIN, device=device)
    z, y = model.features(x), model(x)
    print(f"input {tuple(x.shape)} → z {tuple(z.shape)} → prediction {tuple(y.shape)}")
    y.square().mean().backward()
    grad = model.blocks[0].layers[0].weight.grad
    assert grad is not None and grad.abs().sum() > 0, "no gradient reaches the first layer"
    print("gradients reach the first layer")

    model.eval()
    with torch.no_grad():
        shuffled = x[:, torch.randperm(20, device=device)]
        assert torch.allclose(model(x), model(shuffled), atol=1e-5), "channel order changes the prediction"
    print("shuffling the 20 channels gives the same prediction: channels are filtered alone, then max-pooled")

    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    target = torch.randn(64, device=device)
    for step in range(25):
        if step == 5: # don't time the warm-up steps
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
        loss = nn.functional.huber_loss(model(x), target)
        opt.zero_grad()
        loss.backward()
        opt.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
        print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB") # edge case for my own computer
    step_s = (time.time() - t0) / 20
    print(f"one training step (batch 64): {step_s * 1000:.0f} ms, so about {step_s * 104:.0f} s per epoch")





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="run the data checks and stop")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # cuda for nvidia gpus, cpu fallback
    if args.check:
        check_data(device)
        check_model(device)