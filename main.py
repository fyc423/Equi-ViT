from torch.utils.tensorboard import SummaryWriter
import sys
import os, math, torch, argparse, random
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


from nct_crc import NCT_CRC
from transforms import get_nct_crc_transforms
from vit_wrapper import HFViTWrapper


def parse_args():
    ap = argparse.ArgumentParser()
    # Model / checkpoint
    ap.add_argument("--vit_ckpt", default="google/vit-base-patch16-224",
                    help="HuggingFace ViT checkpoint id or local path")
    ap.add_argument("--random_init", action="store_true",
                    help="Initialize from config instead of loading pretrained weights")

    # Patch and kernel size arguments
    ap.add_argument("--patch_size", type=int, default=16,
                    help="Patch size for ViT patch embedding (default: 16)")
    ap.add_argument("--image_size", type=int, default=224, help="Input image size (height/width) in pixels")
    ap.add_argument("--kernel_size1", type=int, required=True,
                    help="Kernel size for first GMRConv2d in patch embedding (required)")
    ap.add_argument("--kernel_size2", type=int, required=True,
                    help="Kernel size for second GMRConv2d in patch embedding (required)")

    # Training 
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--out", default="runs/vit_vanilla")
    ap.add_argument("--seed", type=int, default=42)

    # Augment/transform knobs 
    ap.add_argument("--fix_rotate", action="store_true")
    ap.add_argument("--degree", type=float, default=0.0)

    #Other
    ap.add_argument("--dataset", type=str, default="nct_crc", choices=["nct_crc"],
                    help="Dataset to use: 'nct_crc'")

    return ap.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False  # keep fast kernels
    torch.backends.cudnn.benchmark = True


def main():
    args = parse_args()
    # Debug: show raw argv and parsed args
    try:
        print("[DEBUG] sys.argv:", " ".join(sys.argv))
        print("[DEBUG] parsed args:", {k: getattr(args, k) for k in vars(args)})
    except Exception:
        pass
    writer = SummaryWriter(log_dir=os.path.join(args.out, "tensorboard"))
    os.makedirs(args.out, exist_ok=True)
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- Data ----
    if args.dataset == "nct_crc":
        train_tf, val_tf = get_nct_crc_transforms(args)
        train_ds = NCT_CRC("data/NCT-CRC-HE-100K", transform=train_tf)
        val_ds   = NCT_CRC("data/CRC-VAL-HE-7K",   transform=val_tf)
        num_classes = 9
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size*2, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True, drop_last=False)

    # ---- Model (prefer no-env; pass use_gmr directly). Fallback if wrapper doesn't accept it. ----
    def build_model():
        try:
            return HFViTWrapper(
                num_labels=num_classes,
                checkpoint=args.vit_ckpt,
                pretrained=not args.random_init,
                use_gmrx2=True,
                patch_size=args.patch_size,
                kernel_size1=args.kernel_size1,
                kernel_size2=args.kernel_size2,
                image_size=args.image_size,
            ).to(device)
        except TypeError:
            print("[ERROR] HFViTWrapper does not accept the provided arguments. Please update your wrapper to support GMRx2 and kernel sizes.")
            raise

    model = build_model()
    start_epoch = 1
    best_acc = 0.0

    # ---- Optim + sched (original) ----
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = max(1, len(train_loader))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps_per_epoch * args.epochs)

    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    # ---- Eval ----
    import numpy as np
    def evaluate(val_loader):
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                    logits = model(x, interpolate_pos_encoding=False)
                    pred = logits.argmax(1)
                correct += (pred == y).sum().item()
                total   += y.numel()
        return correct / max(total, 1)

    # ---- Pretty config print ----
    print("========== CONFIG ==========")
    print(f"device:          {device}")
    print(f"checkpoint:      {args.vit_ckpt}")
    print(f"pretrained:      {not args.random_init}")
    print("use_gmrx2:       True")
    print(f"epochs:          {args.epochs}")
    print(f"batch_size:      {args.batch_size}")
    print(f"lr:              {args.lr}")
    print(f"weight_decay:    {args.weight_decay}")
    print(f"out dir:         {args.out}")
 
    print("============================")

    global_step = 0

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                logits = model(x, interpolate_pos_encoding=args.interpolate_pos_encoding)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            # Log training loss and LR per batch
            writer.add_scalar("Loss/train", loss.item(), global_step)
            for i, pg in enumerate(optimizer.param_groups):
                writer.add_scalar(f"LR/group_{i}", pg['lr'], global_step)
            global_step += 1

        print(f"\n==== Epoch {epoch} Rotation Inference ====")
        rotation_angles = range(0, 360, 30)
        rotation_accuracies = {}
        if args.dataset == "nct_crc":
            for angle in rotation_angles:
                print(f"Testing with rotation: {angle} degrees")
                args.degree = angle
                _, val_tf_rot = get_nct_crc_transforms(args)
                val_ds_rot = NCT_CRC("data/CRC-VAL-HE-7K", transform=val_tf_rot)
                val_loader_rot = DataLoader(val_ds_rot, batch_size=args.batch_size*2, shuffle=False,
                                           num_workers=args.num_workers, pin_memory=True, drop_last=False)
                acc = evaluate(val_loader_rot)
                rotation_accuracies[angle] = acc
                print(f"Accuracy at {angle} degrees: {acc:.4f}")
                # Log validation accuracy for each rotation angle
                writer.add_scalar(f"Accuracy/val_{angle}", acc, epoch)
            print("Rotation Accuracies:")
            for angle, acc in rotation_accuracies.items():
                print(f"{angle} deg: {acc:.4f}")
            # Save best model based on 0-degree accuracy (original behavior)
            acc_0 = rotation_accuracies.get(0, 0.0)
            writer.add_scalar("Accuracy/val_nct_crc_0", acc_0, epoch)
            if acc_0 > best_acc:
                best_acc = acc_0
                torch.save(model.state_dict(), os.path.join(args.out, "vit_best.pt"))


    print("Best val acc:", best_acc)

    # Close TensorBoard writer
    writer.close()

    best_model_path = os.path.join(args.out, "vit_best.pt")
    if os.path.isfile(best_model_path):
        print(f"Loading best model from {best_model_path}")
        model.load_state_dict(torch.load(best_model_path, map_location=device))


if __name__ == "__main__":
    main()
