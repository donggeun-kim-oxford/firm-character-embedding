import os
import glob
import torch
import logging


def save_checkpoint(model, optimizer, epoch, phase, checkpoint_dir, name_prefix="checkpoint_saint"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    filename = f"{name_prefix}_epoch_{epoch}_{phase}.pt"
    ckpt_path = os.path.join(checkpoint_dir, filename)
    ckpt_data = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "phase": phase}
    torch.save(ckpt_data, ckpt_path)
    logging.info(f"Saved checkpoint => {ckpt_path}")


def find_latest_checkpoint(checkpoint_dir, name_prefix="checkpoint_saint"):
    pattern = os.path.join(checkpoint_dir, f"{name_prefix}_epoch_*_*.pt")
    ckpts = glob.glob(pattern)
    if not ckpts:
        return None
    ckpts.sort(key=os.path.getmtime, reverse=True)
    return ckpts[0]


def load_checkpoint(model, optimizer, checkpoint_dir, name_prefix="checkpoint_saint"):
    ckpt_path = find_latest_checkpoint(checkpoint_dir, name_prefix)
    if ckpt_path is None:
        logging.info("No checkpoint found. Starting from scratch.")
        return 0, "none"
    logging.info(f"Loading checkpoint => {ckpt_path}")
    ckpt_data = torch.load(ckpt_path, map_location=__import__('embed_train.config').CONFIG["device"])
    model.load_state_dict(ckpt_data["model"])
    optimizer.load_state_dict(ckpt_data["optimizer"])
    epoch = ckpt_data["epoch"]
    phase = ckpt_data["phase"]
    logging.info(f"Resumed from epoch={epoch}, phase='{phase}'")
    return epoch, phase

