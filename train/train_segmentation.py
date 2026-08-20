import argparse
import yaml
import os
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from models.unet_spm import UNetSPM
from models.loss import SegLoss

class SegDataset(Dataset):
    def __init__(self, npz_list):
        self.files = npz_list
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        d = np.load(self.files[idx])
        img = d["image"].astype(np.float32)
        mask = d["mask"].astype(np.float32)
        img = torch.from_numpy(img).unsqueeze(0)
        mask = torch.from_numpy(mask).unsqueeze(0)
        return img, mask

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", type=str, required=True)
    args = parser.parse_args()
    with open(args.config,"r",encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetSPM(in_channels=1,num_classes=1).to(device)
    criterion = SegLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])
    os.makedirs(cfg["ckpt_dir"], exist_ok=True)
    # --------- pseudo dataloader code, replace with your real file list --------
    train_set = SegDataset([])
    val_set = SegDataset([])
    train_loader = DataLoader(train_set, batch_size=cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_set, batch_size=cfg["batch_size"], shuffle=False)

    for epoch in range(cfg["epochs"]):
        model.train()
        for imgs, masks in train_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            logits = model(imgs)
            loss = criterion(logits, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # validation loop omitted, add your own validation
        torch.save(model.state_dict(), os.path.join(cfg["ckpt_dir"],f"seg_ep{epoch}.pth"))

if __name__ == "__main__":
    main()