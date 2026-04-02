#!/usr/bin/env python3
"""Train reconstruction models for DAEL route2.

This script trains feature extractors used by [route2/trainer.py](route2/trainer.py):
- AE (default): saves `ae_rec_{noise}.pth`
- Denoising VAE (new skeleton): saves `vae_rec_{noise}.pth`
- Dual-latent VAE (fluctuation-aware): saves `dualvae_rec_{noise}.pth`
"""

from __future__ import annotations

import argparse
import os
import sys
import random
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

current_file = Path(__file__).resolve()
triad_root = current_file.parents[3]
if str(triad_root) not in sys.path:
    sys.path.append(str(triad_root))

from da_models.dael.route2 import dael_da
from da_models.dael.route2 import dael_utils


DEFAULT_TARGET_CELLS = ["Monocytes", "Unknown", "CD4Tcells", "Bcells", "NK", "CD8Tcells"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class DenoisingVAE(dael_da.BaseModel):
    """A stronger reconstruction backbone than plain AE in many noisy settings.

    This model combines denoising + variational regularization, which often
    improves robustness to domain shift compared to deterministic AE.
    """

    def __init__(self, option_list, seed=42):
        super().__init__(seed=seed)
        self.seed = seed
        self.batch_size = option_list["batch_size"]
        self.feature_num = option_list["feature_num"]
        self.latent_dim = option_list["latent_dim"]
        self.num_epochs = option_list["epochs"]
        self.lr = option_list["learning_rate"]
        self.early_stop = option_list["early_stop"]
        self.beta = option_list.get("vae_beta", 1e-3)

        self.enc = nn.Sequential(
            nn.Linear(self.feature_num, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.GELU(),
        )
        self.mu = nn.Linear(512, self.latent_dim)
        self.logvar = nn.Linear(512, self.latent_dim)
        self.dec = nn.Sequential(
            nn.Linear(self.latent_dim, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1024),
            nn.GELU(),
            nn.Linear(1024, self.feature_num),
        )

    def encode(self, x):
        h = self.enc(x)
        return self.mu(h), self.logvar(h)

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        rec = self.dec(z)
        return rec, z, mu, logvar

    def loss(self, x, rec, mu, logvar):
        rec_loss = F.mse_loss(rec, x)
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return rec_loss + self.beta * kl, rec_loss.detach(), kl.detach()


class DualLatentVAE(dael_da.BaseModel):
    """Disentangle stable/noise-sensitive latent spaces.

    - z_stable: domain-invariant backbone for robust reconstruction.
    - z_noise: intentionally keeps noise-dependent fluctuation information.
    """

    def __init__(self, option_list, seed=42):
        super().__init__(seed=seed)
        self.seed = seed
        self.batch_size = option_list["batch_size"]
        self.feature_num = option_list["feature_num"]
        self.latent_dim = option_list["latent_dim"]
        self.num_epochs = option_list["epochs"]
        self.lr = option_list["learning_rate"]
        self.early_stop = option_list["early_stop"]
        self.beta_stable = option_list.get("beta_stable", 1e-3)
        self.beta_noise = option_list.get("beta_noise", 5e-4)
        self.lambda_noise = option_list.get("lambda_noise", 0.2)

        hidden = 1024
        trunk_out = 512

        self.trunk = nn.Sequential(
            nn.Linear(self.feature_num, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, trunk_out),
            nn.GELU(),
        )

        self.mu_stable = nn.Linear(trunk_out, self.latent_dim)
        self.logvar_stable = nn.Linear(trunk_out, self.latent_dim)
        self.mu_noise = nn.Linear(trunk_out, self.latent_dim)
        self.logvar_noise = nn.Linear(trunk_out, self.latent_dim)

        self.noise_head = nn.Sequential(
            nn.Linear(self.latent_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

        self.dec = nn.Sequential(
            nn.Linear(self.latent_dim * 2, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1024),
            nn.GELU(),
            nn.Linear(1024, self.feature_num),
        )

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    @staticmethod
    def kl_loss(mu, logvar):
        return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    def forward(self, x):
        h = self.trunk(x)
        mu_s, logvar_s = self.mu_stable(h), self.logvar_stable(h)
        mu_n, logvar_n = self.mu_noise(h), self.logvar_noise(h)

        z_s = self.reparameterize(mu_s, logvar_s)
        z_n = self.reparameterize(mu_n, logvar_n)

        rec = self.dec(torch.cat([z_s, z_n], dim=1))
        noise_pred = self.noise_head(z_n).squeeze(1)
        return rec, z_s, z_n, mu_s, logvar_s, mu_n, logvar_n, noise_pred

    def loss(self, x, rec, mu_s, logvar_s, mu_n, logvar_n, noise_pred, noise_level):
        rec_loss = F.mse_loss(rec, x)
        kl_s = self.kl_loss(mu_s, logvar_s)
        kl_n = self.kl_loss(mu_n, logvar_n)

        target_noise = torch.full_like(noise_pred, float(noise_level))
        noise_loss = F.mse_loss(noise_pred, target_noise)

        total = (
            rec_loss
            + self.beta_stable * kl_s
            + self.beta_noise * kl_n
            + self.lambda_noise * noise_loss
        )
        return total, rec_loss.detach(), kl_s.detach(), kl_n.detach(), noise_loss.detach()


def parse_list(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def train_one_noise(train_data, target_cells, args, noise, device):
    option_list = {
        "batch_size": args.batch_size,
        "feature_num": train_data.shape[1],
        "latent_dim": args.latent_dim,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "early_stop": args.early_stop,
        "vae_beta": args.vae_beta,
        "beta_stable": args.beta_stable,
        "beta_noise": args.beta_noise,
        "lambda_noise": args.lambda_noise,
        "SaveResultsDir": args.save_dir,
    }

    if args.model == "ae":
        model = dael_da.AE(option_list, seed=args.seed).to(device)
        ckpt_prefix = "ae_rec"
    elif args.model == "vae":
        model = DenoisingVAE(option_list, seed=args.seed).to(device)
        ckpt_prefix = "vae_rec"
    elif args.model == "dualvae":
        model = DualLatentVAE(option_list, seed=args.seed).to(device)
        ckpt_prefix = "dualvae_rec"
    else:
        raise ValueError(f"Unknown model: {args.model}")

    model.aug_dataloader(train_data, batch_size=args.batch_size, noise=noise, target_cells=target_cells)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)

    best_loss = float("inf")
    stale = 0
    save_path = os.path.join(args.save_dir, f"{ckpt_prefix}_{noise}.pth")

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        n_batch = 0

        for batch_x, _ in model.aug_loader:
            x = batch_x.to(device)
            optimizer.zero_grad()

            if args.model == "ae":
                rec, _ = model(x)
                loss = F.mse_loss(rec, x)
            elif args.model == "vae":
                rec, _, mu, logvar = model(x)
                loss, rec_loss, kl = model.loss(x, rec, mu, logvar)
            else:
                rec, _, _, mu_s, logvar_s, mu_n, logvar_n, noise_pred = model(x)
                loss, rec_loss, kl_s, kl_n, noise_loss = model.loss(
                    x, rec, mu_s, logvar_s, mu_n, logvar_n, noise_pred, noise
                )

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batch += 1

        epoch_loss /= max(n_batch, 1)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            stale = 0
            torch.save(model.state_dict(), save_path)
        else:
            stale += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            if args.model == "vae":
                print(
                    f"[noise={noise}] epoch={epoch + 1} loss={epoch_loss:.6f} "
                    f"(last_rec={float(rec_loss):.6f}, last_kl={float(kl):.6f})"
                )
            elif args.model == "dualvae":
                print(
                    f"[noise={noise}] epoch={epoch + 1} loss={epoch_loss:.6f} "
                    f"(last_rec={float(rec_loss):.6f}, last_kl_s={float(kl_s):.6f}, "
                    f"last_kl_n={float(kl_n):.6f}, last_noise={float(noise_loss):.6f})"
                )
            else:
                print(f"[noise={noise}] epoch={epoch + 1} mse={epoch_loss:.6f}")

        if stale >= args.early_stop:
            print(f"[noise={noise}] early stop at epoch {epoch + 1}")
            break

    print(f"[noise={noise}] best_loss={best_loss:.6f} -> {save_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train reconstruction backbones for DAEL route2")
    parser.add_argument("--h5ad-path", required=True, type=str)
    parser.add_argument("--save-dir", required=True, type=str)
    parser.add_argument("--source-list", default="donorA,donorC,data6k,data8k,sdy67", type=str)
    parser.add_argument("--target-cells", default=",".join(DEFAULT_TARGET_CELLS), type=str)
    parser.add_argument("--n-samples", default=None, type=int)
    parser.add_argument("--n-vtop", default=1000, type=int)
    parser.add_argument("--noise-list", default="0.0,0.1,1.0", type=str)

    parser.add_argument("--model", choices=["ae", "vae", "dualvae"], default="ae")
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--latent-dim", default=256, type=int)
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--learning-rate", default=1e-3, type=float)
    parser.add_argument("--early-stop", default=30, type=int)
    parser.add_argument("--vae-beta", default=1e-3, type=float)
    parser.add_argument("--beta-stable", default=1e-3, type=float)
    parser.add_argument("--beta-noise", default=5e-4, type=float)
    parser.add_argument("--lambda-noise", default=0.2, type=float)
    parser.add_argument("--seed", default=42, type=int)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    set_seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)
    source_list = parse_list(args.source_list)
    target_cells = parse_list(args.target_cells)
    noise_list = [float(x) for x in parse_list(args.noise_list)]

    train_data, gene_names = dael_utils.prep_daeldg(
        h5ad_path=args.h5ad_path,
        source_list=source_list,
        n_samples=args.n_samples,
        n_vtop=args.n_vtop,
    )
    _ = gene_names

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}, model={args.model}, noises={noise_list}")

    for noise in noise_list:
        train_one_noise(train_data, target_cells, args, noise, device)


if __name__ == "__main__":
    main()
