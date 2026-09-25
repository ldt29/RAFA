import os
import random
from functools import partial
from typing import Dict, List, Literal, Tuple, Union

import lightning as L
import numpy as np
import torch
from jaxtyping import Bool, Float
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from loguru import logger
from torch import Tensor
from omegaconf import OmegaConf

from proteinfoundation.flow_matching.product_space_flow_matcher import (
    ProductSpaceFlowMatcher,
)
from proteinfoundation.nn.local_latents_transformer import LocalLatentsTransformer
from proteinfoundation.nn.losses.structure_losses import (
    backbone_bond_loss,
    interface_distance_loss,
    ca_bond_angle_loss,
    ca_dihedral_loss,
    align_rmsd_loss,
    interface_align_loss,
    soft_fnat_loss,
    steric_clash_loss,
)
from proteinfoundation.nn.local_latents_transformer_unindexed import LocalLatentsTransformerMotifUidx
from proteinfoundation.partial_autoencoder.autoencoder import AutoEncoder
from proteinfoundation.utils.coors_utils import nm_to_ang, trans_nm_to_atom37
from proteinfoundation.utils.pdb_utils import (
    create_full_prot,
    to_pdb,
)


@rank_zero_only
def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)


class Proteina(L.LightningModule):
    def __init__(self, cfg_exp, store_dir=None, autoencoder_ckpt_path=None):
        super().__init__()
        self.save_hyperparameters()
        self.cfg_exp = cfg_exp
        self.inf_cfg = None  # Only for inference runs
        self.validation_output_lens = {}
        self.validation_output_data = []
        self.store_dir = store_dir if store_dir is not None else "./tmp"
        self.val_path_tmp = os.path.join(self.store_dir, "val_samples")
        create_dir(self.val_path_tmp)

        self.metric_factory = None

        if autoencoder_ckpt_path is not None:
            # Allow adding new keys
            logger.info(f"Manually setting autoencoder_ckpt_path to {autoencoder_ckpt_path}")
            OmegaConf.set_struct(cfg_exp, False)
            # Update the configuration with the new key-value pair
            cfg_exp.autoencoder_ckpt_path = autoencoder_ckpt_path
            # Re-enable struct mode if needed
            OmegaConf.set_struct(cfg_exp, True)

        self.autoencoder, latent_dim = self.load_autoencoder(cfg_exp, freeze_params=True)
        
        # Add right latent dimensionality in the config file, needed to instantiate the flow matcher below
        if latent_dim is not None:
            self.latent_dim = latent_dim
        else:
            self.latent_dim = cfg_exp.product_flowmatcher.local_latents.get("dim", 8)
            
        if self.autoencoder is not None:
            try:
                cfg_exp.product_flowmatcher.local_latents.dim = self.latent_dim
            except:
                OmegaConf.set_struct(cfg_exp, False)
                # Update the configuration with the new key-value pair
                cfg_exp.product_flowmatcher.local_latents.dim = self.latent_dim
                # Re-enable struct mode if needed
                OmegaConf.set_struct(cfg_exp, True)

        self.fm = ProductSpaceFlowMatcher(cfg_exp)
        logger.info(f"cfg_exp.nn: {cfg_exp.nn}")

        # Neural network
        if cfg_exp.nn.name == "local_latents_transformer":
            self.nn = LocalLatentsTransformer(**cfg_exp.nn, latent_dim=self.latent_dim)
        elif cfg_exp.nn.name == "local_latents_transformer_motif_uidx":
            self.nn = LocalLatentsTransformerMotifUidx(**cfg_exp.nn, latent_dim=self.latent_dim)
        else:
            raise IOError(f"Wrong nn selected for CAFlow {cfg_exp.nn.name}")

        # Scaling laws stuff
        self.nflops = 0
        self.nsamples_processed = 0
        self.nparams = sum(p.numel() for p in self.nn.parameters() if p.requires_grad)

        self.nn_ag = None

        # Antibody design mode
        self.ab_design_mode = cfg_exp.training.get("ab_design_mode", False)
        if self.ab_design_mode:
            logger.info("Antibody design mode enabled")

        # The teacher and student share this module.  The teacher-only
        # privileged adapter lives inside ``self.nn`` and is enabled by the
        # corresponding NN config flag; the student path below explicitly
        # sanitizes native target inputs before every public forward.
        self.teacher_training_cfg = cfg_exp.get("teacher_training", {})
        self.teacher_training_enabled = bool(
            self.teacher_training_cfg.get("enabled", False)
        )
        if self.teacher_training_enabled:
            logger.info(
                f"Teacher/student training enabled: {self.teacher_training_cfg}"
            )

    def apply_cdr_mask(self, batch: Dict) -> Dict:
        """
        Ensure correct masks are set for antibody design training.

        After corrupt_batch, batch["mask"] == mask_dict["coords"][...,0,0] == ab_mask (ab-only).
        This function simply ensures batch["full_mask"] (ab+ag, for attention) is present.
        The collate_fn pre-sets it; this is a safety fallback.

        Args:
            batch: Training batch (post corrupt_batch)

        Returns:
            Batch with full_mask guaranteed to be set
        """
        if not self.ab_design_mode:
            return batch

        if "full_mask" not in batch:
            # Fallback: recover from chain_type if collate_fn didn't set it
            chain_type = batch.get("chain_type")
            if chain_type is not None:
                batch["full_mask"] = (chain_type > 0)
            else:
                batch["full_mask"] = batch["mask"].clone()

        # batch["mask"] is already ab-only (set by collate_fn and confirmed by corrupt_batch)
        return batch

    def load_autoencoder(self, cfg_exp, freeze_params=True):
        """Loads autoencoder, if required."""
        if ("autoencoder_ckpt_path" in cfg_exp):
            # for new runs trained with refactored codebase
            ae_ckp_path = cfg_exp.autoencoder_ckpt_path
        elif ("autoencoder_ckpt_path" in cfg_exp.product_flowmatcher.local_latents):
            # for old runs trained with old codebase
            ae_ckp_path = cfg_exp.product_flowmatcher.local_latents.autoencoder_ckpt_path
        else:
            raise ValueError("No autoencoder checkpoint path provided")

        if ae_ckp_path is None:
            # If no checkpoint path provided but nn_ae config exists, create a new autoencoder
            if hasattr(cfg_exp, 'nn_ae') and cfg_exp.nn_ae is not None:
                logger.info("Creating new autoencoder from config (no checkpoint provided)")
                # AutoEncoder expects config to have nn_ae inside it
                ae_config = OmegaConf.create({'nn_ae': cfg_exp.nn_ae})
                autoencoder = AutoEncoder(ae_config)
                latent_dim = cfg_exp.nn_ae.get('latent_z_dim', 8)
                if freeze_params:
                    # Don't freeze during training - we want to train it
                    pass
                return autoencoder, latent_dim
            return None, None

        logger.info(f"Loading autoencoder from {ae_ckp_path}")
        autoencoder = AutoEncoder.load_from_checkpoint(ae_ckp_path, strict=False)
        if freeze_params:
            for param in autoencoder.parameters():
                param.requires_grad = False
        return autoencoder, autoencoder.latent_dim

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            [p for p in self.parameters() if p.requires_grad], lr=self.cfg_exp.opt.lr
        )
        return optimizer

    def on_save_checkpoint(self, checkpoint):
        """Adds additional variables to checkpoint."""
        checkpoint["nflops"] = self.nflops
        checkpoint["nsamples_processed"] = self.nsamples_processed

    def on_load_checkpoint(self, checkpoint):
        """Loads additional variables from checkpoint."""
        try:
            self.nflops = checkpoint["nflops"]
            self.nsamples_processed = checkpoint["nsamples_processed"]
        except:
            logger.info("Failed to load nflops and nsamples_processed from checkpoint")
            self.nflops = 0
            self.nsamples_processed = 0

    def call_nn(
        self,
        batch: Dict[str, torch.Tensor],
        n_recycle: int,
        nn_override=None,
    ) -> Dict[str, torch.Tensor]:
        """
        Calls NN with recycling. Should this be here or in the NN? Possibly better here,
        in case we want to recycle using decoder for some approach, etc, and this is akin
        to self conditioning, also here.
        Also, if we want to recycle clean sample predictions... Then we'd need this here,
        as the nn does not know about relations between v, x1, ...
        """
        nn_model = self.nn if nn_override is None else nn_override

        # First call
        nn_out = nn_model(batch)

        # Recycle n_recycle times detaching gradients and updating input
        # Note that recycling is supported by the codebase, but the models provided 
        # with the La-Proteina paper do not use it, nor were trained with it.
        for _ in range(n_recycle):
            x_1_pred = self.fm.nn_out_to_clean_sample_prediction(
                batch=batch, nn_out=nn_out
            )
            batch[f"x_recycle"] = {dm: x_1_pred[dm].detach() for dm in x_1_pred}
            nn_out = nn_model(batch)

        # Final prediction
        return nn_out

    def _teacher_batch(self, batch: Dict) -> Dict:
        """Make the privileged teacher view, with optional input noise.

        Noise is applied only to the clean information consumed by the
        privileged encoder.  The original batch remains the loss target, so
        this regularizes the teacher without changing the FM supervision.
        """
        teacher_batch = dict(batch)
        teacher_batch["use_privileged_info"] = True
        noise_cfg = self.teacher_training_cfg.get("privileged_noise", {})
        if not bool(noise_cfg.get("enabled", False)):
            return teacher_batch

        target_mask = batch.get("mask")
        if target_mask is None:
            target_mask = batch.get("full_mask")
        if target_mask is None:
            raise ValueError("privileged noise requires a target structure mask")
        target_mask = target_mask.bool()

        clean = batch.get("x_1")
        if clean is None:
            raise ValueError("privileged teacher requires clean x_1")
        noisy_clean = {
            key: value.clone() if torch.is_tensor(value) else value
            for key, value in clean.items()
        }

        ca_std = float(noise_cfg.get("ca_std_nm", 0.0))
        if ca_std > 0.0 and "bb_ca" in noisy_clean:
            noise = torch.randn_like(noisy_clean["bb_ca"]) * ca_std
            noisy_clean["bb_ca"] = noisy_clean["bb_ca"] + noise * target_mask.unsqueeze(-1)

        latent_std = float(noise_cfg.get("latent_std", 0.0))
        if latent_std > 0.0 and "local_latents" in noisy_clean:
            noise = torch.randn_like(noisy_clean["local_latents"]) * latent_std
            noisy_clean["local_latents"] = (
                noisy_clean["local_latents"] + noise * target_mask.unsqueeze(-1)
            )
        teacher_batch["x_1"] = noisy_clean

        seq_dropout = float(noise_cfg.get("seq_dropout", 0.0))
        if seq_dropout > 0.0:
            residue_type = batch.get("residue_type", batch.get("seq"))
            if torch.is_tensor(residue_type):
                noisy_seq = residue_type.clone()
                drop = (
                    torch.rand_like(noisy_seq.float()) < seq_dropout
                ) & target_mask
                noisy_seq[drop] = -1
                teacher_batch["residue_type"] = noisy_seq
                if "seq" in batch:
                    teacher_batch["seq"] = noisy_seq

        teacher_batch["privileged_noise_applied"] = True
        return teacher_batch

    def _student_batch(self, batch: Dict) -> Dict:
        """Build the public input view used by the student.

        The flow state and partner/antigen condition remain visible.  Native
        target coordinates are never exposed; for antibody data, framework
        sequence remains public while the native CDR sequence is hidden.  For
        general-protein samples the target sequence is hidden according to the
        sequence-generation contract.
        """
        if not self.teacher_training_enabled:
            return batch

        student_batch = {
            key: value for key, value in batch.items() if not key.startswith("gt_")
        }
        student_batch.pop("x_1", None)
        student_batch["use_privileged_info"] = False

        target_mask = batch.get("mask")
        if target_mask is None:
            target_mask = batch.get("full_mask")
        if target_mask is None:
            return student_batch
        target_mask = target_mask.bool()

        # Hide clean target coordinates while retaining the fixed partner or
        # antigen coordinates as public conditioning.
        for key in ("coords_nm", "coords"):
            value = batch.get(key)
            if torch.is_tensor(value) and value.ndim == 4:
                student_batch[key] = value.masked_fill(
                    target_mask.unsqueeze(-1).unsqueeze(-1), 0.0
                )

        task_type = batch.get("task_type", "antibody")
        is_general = task_type == "general_protein"
        if is_general:
            # General-pair samples expose the partner structure/sequence, but
            # only randomly mask the target sequence.  The target structure is
            # still withheld from the public path and represented by x_t.
            sequence_mask = batch.get(
                "sequence_mask", batch.get("cdr_mask", torch.zeros_like(target_mask))
            )
            hide_seq = sequence_mask.bool() & target_mask
        else:
            native_cdr = batch.get("target_cdr_mask", batch.get("native_cdr_mask"))
            if native_cdr is None:
                native_cdr = batch.get("cdr_mask", torch.zeros_like(target_mask))
            hide_seq = native_cdr.bool() & target_mask

        for key in ("residue_type", "seq"):
            value = batch.get(key)
            if torch.is_tensor(value) and value.ndim == 2:
                student_batch[key] = value.masked_fill(hide_seq, -1)

        # Optional folding/inverse-folding features are privileged because
        # they can otherwise re-introduce the clean target through coords/seq.
        student_batch["use_ca_coors_nm_feature"] = False
        student_batch["use_residue_type_feature"] = False

        # Do not let recycling mutate the original nested diffusion state.
        for key in ("x_t", "x_sc", "x_recycle"):
            value = student_batch.get(key)
            if isinstance(value, dict):
                student_batch[key] = {
                    subkey: subvalue.clone() if torch.is_tensor(subvalue) else subvalue
                    for subkey, subvalue in value.items()
                }
        return student_batch

    @staticmethod
    def _masked_mse_per_sample(
        pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        mask_f = mask.float()
        denom = (mask_f.sum(dim=-1) * pred.shape[-1]).clamp(min=1.0)
        err = ((pred - target) ** 2) * mask_f.unsqueeze(-1)
        return err.sum(dim=(-1, -2)) / denom

    def _compute_teacher_distill_losses(
        self,
        batch: Dict,
        student_out: Dict[str, torch.Tensor],
        teacher_out: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        distill_weight = float(self.teacher_training_cfg.get("distill_weight", 1.0))
        if distill_weight == 0.0:
            return {}

        mask = batch["mask"].bool()
        t_min = float(self.cfg_exp.loss.get("t_min_struct", 0.0))
        t_gate = (batch["t"]["bb_ca"] >= t_min).float()
        velocity_weight = float(
            self.teacher_training_cfg.get("velocity_distill_weight", 1.0)
        )
        losses: Dict[str, torch.Tensor] = {}
        for data_mode in ("bb_ca", "local_latents"):
            loss = self._masked_mse_per_sample(
                student_out[data_mode]["v"],
                teacher_out[data_mode]["v"].detach(),
                mask,
            )
            losses[f"teacher_distill_{data_mode}_v"] = (
                distill_weight * velocity_weight * t_gate * loss
            )
            losses[f"teacher_distill_{data_mode}_v_raw_justlog"] = loss.detach()
        return losses

    def compute_teacher_student_losses(
        self,
        batch: Dict,
        student_out: Dict[str, torch.Tensor],
        n_recycle: int = 0,
    ) -> Dict[str, torch.Tensor]:
        """Compute teacher auxiliary losses plus student←teacher distillation."""
        teacher_out = self.call_nn(self._teacher_batch(batch), n_recycle=n_recycle)
        losses: Dict[str, torch.Tensor] = {}
        teacher_losses = self.fm.compute_loss(batch=batch, nn_out=teacher_out)
        if self.ab_design_mode:
            teacher_losses.update(self._compute_structure_losses(batch, teacher_out))
        for key, value in teacher_losses.items():
            if "_justlog" in key:
                losses[f"teacher_{key}"] = value.detach()
            else:
                losses[f"teacher_{key}"] = value
                losses[f"teacher_{key}_raw_justlog"] = value.detach()
        losses.update(self._compute_teacher_distill_losses(batch, student_out, teacher_out))
        return losses

    def predict_for_sampling(
        self,
        batch: Dict,
        mode: Literal["full", "ucond"],
        n_recycle: int,
        nn_override=None,
    ) -> Tuple[Union[Dict[str, torch.Tensor], float, None]]:
        """
        This function predicts clean samples for multiple models:
        x_pred, the 'original' model, if mode == full
        x_pred_ucond, the unconditional model, , if mode == ucond

        TODO: Need to update to include autoguidance again

        These predictions will later be used to sample with guidance and autoguidance.

        Args:
            batch: Dict
            mode: str

        Returns:
            x_pred (tensor) for the requested mode
        """
        # Bug fix: simulation_step zeros antigen positions via _apply_mask(x, ab_mask).
        # The model was trained with true antigen Ca in x_t, so restore them here
        # at every ODE step to match the training distribution.
        if self.ab_design_mode and "x_t" in batch:
            _ct = batch.get("chain_type")
            _cnm = batch.get("coords_nm")
            if _ct is not None and _cnm is not None:
                _ct2 = _ct if _ct.dim() == 2 else _ct.unsqueeze(0)
                _cnm2 = _cnm if _cnm.dim() == 4 else _cnm.unsqueeze(0)
                _ag_mask = (_ct2 == 3)  # [b, n]
                if _ag_mask.any():
                    _ag_ca = _cnm2[:, :, 1, :]  # [b, n, 3] Ca in nm
                    batch["x_t"]["bb_ca"] = torch.where(
                        _ag_mask.unsqueeze(-1), _ag_ca, batch["x_t"]["bb_ca"]
                    )
                    if "local_latents" in batch["x_t"]:
                        batch["x_t"]["local_latents"] = torch.where(
                            _ag_mask.unsqueeze(-1),
                            torch.zeros_like(batch["x_t"]["local_latents"]),
                            batch["x_t"]["local_latents"],
                        )

        model_batch = self._student_batch(batch)
        if mode == "full":
            nn_out = self.call_nn(
                model_batch, n_recycle=n_recycle, nn_override=nn_override
            )
        elif mode == "ucond":
            assert "cath_code" in batch or "x_motif" in batch, "Only support CFG when cath_code or x_motif is provided"
            uncond_batch = model_batch.copy()
            if "cath_code" in uncond_batch:
                uncond_batch.pop("cath_code")
            if "x_motif" in uncond_batch:
                uncond_batch.pop("x_motif")
            nn_out = self.call_nn(
                uncond_batch, n_recycle=n_recycle, nn_override=nn_override
            )
        else:
            raise IOError(f"Wrong {mode} passed to `predict_for_sampling`")

        return nn_out

    def training_step(self, batch: Dict, batch_idx: int):
        """
        Computes training loss for batch of samples.

        Args:
            batch: Data batch.

        Returns:
            Training loss averaged over batch dimension.
        """
        val_step = batch_idx == -1  # validation step is indicated with batch_idx -1
        log_prefix = "validation_loss" if val_step else "train"

        # Add clean samples for all data modes / spaces we are working on
        batch = self.add_clean_samples(batch)

        # Corrupt the batch
        batch = self.fm.corrupt_batch(batch)  # adds x_1, t, x_0, x_t, mask
        bs, n = batch["mask"].shape

        # Apply CDR mask for antibody design mode
        if self.ab_design_mode:
            batch = self.apply_cdr_mask(batch)
            # Restore antigen positions in x_t to their true coordinates (fixed condition).
            # corrupt_batch zeros out antigen positions (mask_dict covers antibody only),
            # but the nn attends over all positions via full_mask — antigen must be visible.
            ag_mask = (batch["chain_type"] == 3)  # [b, n]
            if ag_mask.any():
                ag_ca = batch["coords_nm"][:, :, 1, :]  # [b, n, 3] CA in nm
                batch["x_t"]["bb_ca"] = torch.where(
                    ag_mask.unsqueeze(-1), ag_ca, batch["x_t"]["bb_ca"]
                )
                # Zero antigen local_latents: corrupt_batch sets them to (1-t)*noise
                # but they should always be 0 (AE never encodes antigen).
                if "local_latents" in batch["x_t"]:
                    batch["x_t"]["local_latents"] = torch.where(
                        ag_mask.unsqueeze(-1),
                        torch.zeros_like(batch["x_t"]["local_latents"]),
                        batch["x_t"]["local_latents"],
                    )

        # Handle conditioning variables
        batch = self.handle_self_cond(
            batch
        )  # self conditioning, adds ["x_sc"] to batch prob 0.5
        batch = self.handle_folding_n_inverse_folding(
            batch
        )  # folding and inverse folding iterations

        # Number of recycling steps
        n_recycle = self.handle_recycling()

        # The student sees the sanitized public view.  The original batch is
        # retained for the clean FM/structure targets and teacher branch.
        nn_out = self.call_nn(self._student_batch(batch), n_recycle=n_recycle)
        losses = self.fm.compute_loss(
            batch=batch,
            nn_out=nn_out,
        )  # Dict[str, Tensor w.batch shape [*]]
        # After compute_loss, nn_out["bb_ca"]["x_1"] is populated via
        # nn_out_add_clean_sample_prediction (x_t + (1-t)*v).

        # ── Priority A: CA coordinate structure losses ────────────────────────
        if self.ab_design_mode:
            struct_losses = self._compute_structure_losses(batch, nn_out)
            losses.update(struct_losses)

        if self.teacher_training_enabled:
            losses.update(
                self.compute_teacher_student_losses(
                    batch, student_out=nn_out, n_recycle=n_recycle
                )
            )

        self.log_losses(bs=bs, losses=losses, log_prefix=log_prefix, batch=batch)
        train_loss = sum([torch.mean(losses[k]) for k in losses if "_justlog" not in k])

        self.log(
            f"{log_prefix}/loss",
            train_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            batch_size=bs,
            sync_dist=True,
            add_dataloader_idx=False,
        )

        if not val_step:  # Don't log these for val step
            self.log_train_loss_n_prog_bar(bs, train_loss)
            self.update_n_log_flops(bs, n)
            self.update_n_log_nsamples_processed(bs)
            self.log_nparams()

        return train_loss

    # ── Priority A: CA coordinate structure losses ────────────────────────────

    def _compute_structure_losses(
        self,
        batch: Dict,
        nn_out: Dict,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute auxiliary structural losses on the predicted clean CA coords.

        Requires self.ab_design_mode = True and nn_out["bb_ca"]["x_1"] to be
        already populated (done by ProductSpaceFlowMatcher.compute_loss).

        Loss A-1 (backbone_bond_loss): CA-CA virtual bond length ≈ 3.8 Å.
        Loss A-2 (interface_distance_loss): CDR residues should be within
            contact distance of their ground-truth epitope neighbours.

        Both losses are gated by t > t_min_struct so that high-noise samples
        (where x_1_pred is meaningless) do not pollute the signal.
        The gate is applied per sample, so partial batches still contribute.

        Config keys (under cfg_exp.loss):
            bond_weight:       float  (default 0.0 → disabled)
            angle_weight:      float  (default 0.0 → disabled)
            dihedral_weight:   float  (default 0.0 → disabled)
            interface_weight:  float  (default 0.0 → disabled)
            contact_weight:    float  (default 0.0 → disabled)  # Priority B
            t_min_struct:      float  (default 0.4)

        Returns:
            Dict of named losses shaped [B], following the same convention as
            fm.compute_loss — the caller does torch.mean(loss) when summing.
        """
        loss_cfg = self.cfg_exp.get("loss", {})
        bond_weight        = float(loss_cfg.get("bond_weight", 0.0))
        angle_weight       = float(loss_cfg.get("angle_weight", 0.0))
        dihedral_weight    = float(loss_cfg.get("dihedral_weight", 0.0))
        interface_weight   = float(loss_cfg.get("interface_weight", 0.0))
        contact_weight     = float(loss_cfg.get("contact_weight", 0.0))
        # New docking-aware losses (L1-L4)
        align_weight       = float(loss_cfg.get("align_weight", 0.0))
        iface_align_weight = float(loss_cfg.get("iface_align_weight", 0.0))
        fnat_weight        = float(loss_cfg.get("fnat_weight", 0.0))
        clash_train_weight = float(loss_cfg.get("clash_train_weight", 0.0))
        t_min_struct       = float(loss_cfg.get("t_min_struct", 0.4))

        all_zero = (
            bond_weight == 0.0 and angle_weight == 0.0 and dihedral_weight == 0.0
            and interface_weight == 0.0 and contact_weight == 0.0
            and align_weight == 0.0 and iface_align_weight == 0.0
            and fnat_weight == 0.0 and clash_train_weight == 0.0
        )
        if all_zero:
            return {}

        # ── Predicted and ground-truth clean CA coords ────────────────────────
        # nn_out["bb_ca"]["x_1"] = x_t + (1-t)*v  [B, N, 3]  (training-only)
        if "bb_ca" not in nn_out or "x_1" not in nn_out["bb_ca"]:
            return {}
        ca_pred = nn_out["bb_ca"]["x_1"]       # [B, N, 3]
        ca_true = batch["x_1"]["bb_ca"]         # [B, N, 3]

        # ── Masks and metadata ────────────────────────────────────────────────
        ab_mask     = batch["mask"].bool()                   # [B, N]  ab only
        chain_type  = batch.get("chain_type")                # [B, N]
        cdr_mask    = batch.get("cdr_mask")                  # [B, N]
        epitope_mask = batch.get("epitope_mask")             # [B, N]

        # ── Per-sample t-gate: only supervise near-clean samples ─────────────
        t = batch["t"]["bb_ca"]                              # [B]
        t_gate = (t > t_min_struct).float()                  # [B]  in {0, 1}
        # All auxiliary losses use t_linear_weight: 0 when t < t_min_struct,
        # then ramps linearly with t (stronger supervision near clean samples).
        t_linear_weight = t_gate * t                         # [B]  ∈ [0, 1]

        losses: Dict[str, torch.Tensor] = {}

        # ── A-1: Backbone bond length ─────────────────────────────────────────
        if bond_weight > 0.0 and chain_type is not None:
            bl = backbone_bond_loss(ca_pred, ab_mask, chain_type)  # [B]
            losses["struct_bond_loss"] = bond_weight * t_linear_weight * bl
            losses["struct_bond_loss_raw_justlog"] = bl.detach()

        # ── A-3: CA pseudo-bond angle ─────────────────────────────────────────
        if angle_weight > 0.0 and chain_type is not None:
            al = ca_bond_angle_loss(ca_pred, ca_true, ab_mask, chain_type)  # [B]
            losses["struct_angle_loss"] = angle_weight * t_linear_weight * al
            losses["struct_angle_loss_raw_justlog"] = al.detach()

        # ── A-4: CA pseudo-dihedral ───────────────────────────────────────────
        if dihedral_weight > 0.0 and chain_type is not None:
            dl = ca_dihedral_loss(ca_pred, ca_true, ab_mask, chain_type)  # [B]
            losses["struct_dihedral_loss"] = dihedral_weight * t_linear_weight * dl
            losses["struct_dihedral_loss_raw_justlog"] = dl.detach()

        # ── A-2: Ab-Epitope interface distance ───────────────────────────────
        paratope_mask = batch.get("paratope_mask")
        if (
            interface_weight > 0.0
            and epitope_mask is not None
            and paratope_mask is not None
        ):
            il = interface_distance_loss(
                ca_pred,
                ca_true,
                paratope_mask.bool(),
                epitope_mask.bool(),
            )  # [B]
            losses["struct_interface_loss"] = interface_weight * t_linear_weight * il
            losses["struct_interface_loss_raw_justlog"] = il.detach()

        # ── Priority B: Contact Head (BCE + log-distance regression) ──────────
        if (
            contact_weight > 0.0
            and "contact_logits" in nn_out
            and epitope_mask is not None
        ):
            contact_logits = nn_out["contact_logits"]          # [B, N, N]
            log_dist_pred  = nn_out.get("log_dist_pred")       # [B, N, N] or None

            gt_dists = torch.cdist(ca_true, ca_true)           # [B, N, N]  (nm)
            contact_gt = (gt_dists < 0.8).float()              # [B, N, N]

            pair_mask = (
                ab_mask.unsqueeze(-1) & epitope_mask.bool().unsqueeze(-2)
            ).float()  # [B, N, N]
            n_pairs = pair_mask.sum(dim=(-1, -2)).clamp(min=1.0)  # [B]

            pos_w = torch.tensor(20.0, device=ca_pred.device, dtype=ca_pred.dtype)
            bce = torch.nn.functional.binary_cross_entropy_with_logits(
                contact_logits, contact_gt, reduction="none", pos_weight=pos_w
            )  # [B, N, N]
            contact_bce = (bce * pair_mask).sum(dim=(-1, -2)) / n_pairs  # [B]
            losses["contact_bce_loss"] = contact_weight * t_linear_weight * contact_bce
            losses["contact_bce_loss_raw_justlog"] = contact_bce.detach()

            if log_dist_pred is not None:
                log_dist_gt = torch.log(gt_dists.clamp(min=1e-3))   # [B, N, N]
                dist_reg = torch.nn.functional.smooth_l1_loss(
                    log_dist_pred, log_dist_gt, reduction="none"
                )  # [B, N, N]
                contact_dist = (dist_reg * pair_mask).sum(dim=(-1, -2)) / n_pairs  # [B]
                losses["contact_dist_loss"] = contact_weight * t_linear_weight * contact_dist
                losses["contact_dist_loss_raw_justlog"] = contact_dist.detach()

        # ── L1: Kabsch-aligned full-antibody RMSD ─────────────────────────────
        if align_weight > 0.0:
            al = align_rmsd_loss(ca_pred, ca_true, ab_mask)   # [B]
            losses["align_rmsd_loss"] = align_weight * t_linear_weight * al
            losses["align_rmsd_loss_raw_justlog"] = al.detach()

        # ── L2: Interface-aligned RMSD (iRMS, 10 Å paratope) ─────────────────
        if iface_align_weight > 0.0 and paratope_mask is not None:
            il2 = interface_align_loss(ca_pred, ca_true, paratope_mask.bool())
            losses["iface_align_loss"] = iface_align_weight * t_linear_weight * il2
            losses["iface_align_loss_raw_justlog"] = il2.detach()

        # ── L3: Soft fnat loss (differentiable contact recovery, 5 Å CA) ─────
        if fnat_weight > 0.0 and chain_type is not None:
            ag_mask_l3 = (chain_type == 3).bool()
            fl = soft_fnat_loss(ca_pred, ca_true, ab_mask, ag_mask_l3)
            losses["soft_fnat_loss"] = fnat_weight * t_linear_weight * fl
            losses["soft_fnat_loss_raw_justlog"] = fl.detach()

        # ── L4: Steric clash loss (pred-ab CA vs true-ag CA, 3.5 Å) ──────────
        if clash_train_weight > 0.0 and chain_type is not None:
            ag_mask_l4 = (chain_type == 3).bool()
            cl = steric_clash_loss(ca_pred, ca_true, ab_mask, ag_mask_l4)
            losses["steric_clash_loss"] = clash_train_weight * t_linear_weight * cl
            losses["steric_clash_loss_raw_justlog"] = cl.detach()

        return losses

    def add_clean_samples(self, batch: Dict) -> Dict:
        """
        Adds clean sample for all data modes / spaces we are working on. For instance, if we have two
        data modes, bb_ca and local_latents, it adds the clean data to the batch
        x_1 = {
            "bb_ca": Corresponding tensor with clean bb_ca coordinates, shape [b, n, 3]
            "local_latents": Corresponding tensor with clean local_latents, shape [b, n, d]
        }

        Args:
            batch: Batch to add clean samples to.

        Returns:
            Batch with clean sample added.
        """
        batch["x_1"] = {
            dm: self._get_clean_sample(batch, dm)
            for dm in self.cfg_exp.product_flowmatcher
        }
        return batch

    def _get_clean_sample(self, batch: Dict, dm: str) -> torch.Tensor:
        """
        Gets clean sample for a given data mode.

        Args:
            batch: Batch to get clean sample from.
            dm: Data mode to get clean sample for.

        Returns:
            Clean sample for the given data mode.
        """
        if dm == "bb_ca":
            return batch["coords_nm"][:, :, 1, :]  # [b, n, 3]
        elif dm == "local_latents":
            if self.ab_design_mode:
                # Slice out antibody-only residues for AE encoding.
                # The AE was trained on single-chain proteins; antigen residues
                # would corrupt the attention and produce bad latents.  General
                # protein pairs also have variable target lengths, so encode
                # each target independently instead of assuming one padded
                # target length for the whole batch.
                ab_mask = batch["mask_dict"]["coords"][..., 0, 0]  # [b, n] bool, antibody only
                b, n = ab_mask.shape
                target_lengths = ab_mask.sum(dim=1).tolist()
                if not target_lengths or max(target_lengths) == 0:
                    return torch.zeros(
                        b,
                        n,
                        self.latent_dim,
                        device=batch["coords_nm"].device,
                        dtype=batch["coords_nm"].dtype,
                    )

                encoded_targets = []
                for i, target_len_value in enumerate(target_lengths):
                    target_len = int(target_len_value)
                    # Keep tensor fields batch-shaped for the AE, then trim the
                    # padded residue axis to this sample's target chain.
                    target_batch = {}
                    for key, value in batch.items():
                        if isinstance(value, torch.Tensor) and value.ndim >= 1 and value.shape[0] == b:
                            item = value[i : i + 1]
                            if item.ndim >= 2 and item.shape[1] == n:
                                item = item[:, :target_len]
                            target_batch[key] = item
                    target_batch["mask_dict"] = {
                        key: value[i : i + 1, :target_len]
                        for key, value in batch["mask_dict"].items()
                        if isinstance(value, torch.Tensor) and value.ndim >= 2
                    }
                    encoded_batch = self.autoencoder.encode(target_batch)
                    # Use posterior mean (deterministic) instead of stochastic
                    # z sample.  With kl.weight=1e-4 the posterior scale may
                    # be large, making the per-step z target noisy.
                    encoded_z = (
                        encoded_batch["mean"]
                        if "mean" in encoded_batch
                        else encoded_batch["z_latent"]
                    )
                    encoded_targets.append(encoded_z[0, :target_len])

                latent_dim = encoded_targets[0].shape[-1]
                z_full = torch.zeros(
                    b,
                    n,
                    latent_dim,
                    device=encoded_targets[0].device,
                    dtype=encoded_targets[0].dtype,
                )
                for i, target_latents in enumerate(encoded_targets):
                    z_full[i, : target_latents.shape[0]] = target_latents
                return z_full
            else:
                encoded_batch = self.autoencoder.encode(batch)
                return encoded_batch["z_latent"]
        else:
            raise ValueError(
                f"Loading clean samples from data mode {dm} not supported."
            )


    def handle_self_cond(self, batch: Dict) -> Dict:
        n_recycle = self.cfg_exp.training.get(
            "n_recycle", 0
        )
        if random.random() > 0.5 and self.cfg_exp.training.self_cond:
            nn_out = self.call_nn(self._student_batch(batch), n_recycle=n_recycle)
            x_1_pred = self.fm.nn_out_to_clean_sample_prediction(
                batch=batch, nn_out=nn_out
            )
            batch["x_sc"] = {k: x_1_pred[k].detach() for k in x_1_pred}

        return batch

    def handle_recycling(self):
        n_recycle = self.cfg_exp.training.get("n_recycle", 0)
        if n_recycle == 0:
            return 0
        return random.randint(0, n_recycle)  # 0 and n_recycle included

    def handle_folding_n_inverse_folding(self, batch: Dict) -> Dict:
        """
        With 15% probability either a folding or inverse folding iteration.
        If one such iteration (ie 15% of the times), with 50% probability set
        set folding_mode to true, otherwise set inverse_folding_mode to true.

        For inverse folding, we just provide CA.

        Applies to the whole batch.

        With 85% probability sets both to false.

        Adds entries 'folding_mode' and 'inverse_folding_ca_mode' to batch, with
        values being boolean variables (True or False).
        """
        batch["use_ca_coors_nm_feature"] = False
        batch["use_residue_type_feature"] = False
        prob = self.cfg_exp.training.get("p_folding_n_inv_folding_iters", 0.0)
        r1 = random.random()  # float
        if r1 < prob:  # with p=prob
            r2 = random.random()
            if r2 < 0.5:  # with p=0.5
                batch["use_ca_coors_nm_feature"] = True
            else:
                batch["use_residue_type_feature"] = True
        return batch

    def log_losses(
        self,
        bs: int,
        losses: Dict[str, Float[torch.Tensor, "b"]],
        log_prefix: str,
        batch: Dict,
    ):
        for k in losses:
            log_name = k[: -len("_justlog")] if k.endswith("_justlog") else k

            self.log(
                f"{log_prefix}/loss_{log_name}",
                torch.mean(losses[k]),
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                batch_size=bs,
                sync_dist=True,
                add_dataloader_idx=False,
            )

            if self.cfg_exp.training.get("p_folding_n_inv_folding_iters", 0.0) > 0.0:
                # Log also for folding and inverse folding iters
                # divides by p_aux to account for the fact that for most steps loss will be just zero
                p_aux = self.cfg_exp.training["p_folding_n_inv_folding_iters"] / 2
                loss = torch.mean(losses[k])  # [b]

                f_inv_fold = batch["use_ca_coors_nm_feature"] * 1.0 / p_aux
                self.log(
                    f"{log_prefix}_invfold_ca_iter/loss_{log_name}",
                    loss * f_inv_fold,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                    batch_size=bs,
                    sync_dist=True,
                    add_dataloader_idx=False,
                )

                f_fold = batch["use_residue_type_feature"] * 1.0 / p_aux
                self.log(
                    f"{log_prefix}_fold_iter/loss_{log_name}",
                    loss * f_fold,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                    batch_size=bs,
                    sync_dist=True,
                    add_dataloader_idx=False,
                )

    def log_train_loss_n_prog_bar(self, b: int, train_loss: torch.Tensor):
        self.log(
            f"train_loss",
            train_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=b,
            sync_dist=True,
            add_dataloader_idx=False,
        )

    def log_nparams(self):
        self.log(
            "scaling/nparams",
            self.nparams * 1.0,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            batch_size=1,
            sync_dist=True,
        )  # constant line

    def update_n_log_nsamples_processed(self, b: int):
        self.nsamples_processed = self.nsamples_processed + b * self.trainer.world_size
        self.log(
            "scaling/nsamples_processed",
            self.nsamples_processed * 1.0,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            batch_size=1,
            sync_dist=True,
        )

    def update_n_log_flops(self, b: int, n: int):
        """
        Updates and logs flops, if available
        """
        try:
            nflops_step = self.nn.nflops_computer(
                b, n
            )  # nn should implement this function if we want to see nflops
        except:
            nflops_step = None

        if nflops_step is not None:
            self.nflops = (
                self.nflops + nflops_step * self.trainer.world_size
            )  # Times number of processes so it logs sum across devices
            self.log(
                "scaling/nflops",
                self.nflops * 1.0,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
                batch_size=1,
                sync_dist=True,
            )

    def validation_step(self, batch: Dict, batch_idx: int):
        """
        Validation step.

        Args:
            batch: batch from dataset (see last argument)
            batch_idx: batch index (unused)
        """
        self.validation_step_data(batch, batch_idx)

    def validation_step_data(self, batch: Dict, batch_idx: int):
        """Evaluates the training loss on validation data."""
        with torch.no_grad():
            loss = self.training_step(batch, batch_idx=-1)
            self.validation_output_data.append(loss.item())

    def on_validation_epoch_end(self):
        """
        Takes the samples produced in the validation step, stores them as pdb files, and computes validation metrics.
        It also cleans results.
        """
        self.on_validation_epoch_end_data()

    def on_validation_epoch_end_data(self):
        self.validation_output_data = []

    def configure_inference(self, inf_cfg, nn_ag):
        """Sets inference config with all sampling parameters required by the method (dt, etc)
        and autoguidance network (or None if not provided)."""
        self.inf_cfg = inf_cfg
        self.nn_ag = nn_ag

    def predict_step(self, batch: Dict, batch_idx: int) -> List[Tuple[torch.tensor]]:
        """
        Makes predictions. Should call set_inf_cfg before calling this.

        Args:
            batch: data batch, contains all info for the samples to generate (nsamples, nres, dt, etc.)
                The full batch is passed through to the prediction functions.

        Returns:
            List of tuples. Each tuple represents one of the generated samples, has two elements:
                coordinates tensor of shape [n, 37, 3], and aatype tensor of shape [n].
        """
        self_cond = self.inf_cfg.args.self_cond

        nsteps = self.inf_cfg.args.nsteps
        guidance_w = self.inf_cfg.args.get("guidance_w", 1.0)
        ag_ratio = self.inf_cfg.args.get("ag_ratio", 0.0)
        save_trajectory_every = 0

        # For ab_design inference: full_mask covers ab+antigen, mask covers antibody only
        if self.ab_design_mode:
            chain_type = batch.get("chain_type", None)
            if chain_type is not None:
                # full_mask: all valid residues (ab + antigen) — used for attention/features
                batch["full_mask"] = (chain_type > 0)
                # mask: antibody-only — used for noise sampling and loss
                batch["mask"] = (chain_type > 0) & (chain_type < 3)

        fn_predict_for_sampling = partial(
            self.predict_for_sampling, n_recycle=self.inf_cfg.get("n_recycle", 0)
        )
        gen_samples, extra_info = self.fm.full_simulation(
            batch=batch,
            predict_for_sampling=fn_predict_for_sampling,
            nsteps=nsteps,
            nsamples=batch["nsamples"],
            n=batch["coords_nm"].shape[1],  # full sequence length (ab + antigen)
            self_cond=self_cond,
            sampling_model_args=self.inf_cfg.model,
            device=self.device,
            save_trajectory_every=save_trajectory_every,
            guidance_w=guidance_w,
            ag_ratio=ag_ratio,
        )
        # Dict with the data_modes as keys, and values with batch shape b
        # extra_info is a dict with additional things, including
        # "mask", whose value is boolean of shape [nsamples, n]

        # Bug fix: for ab_design, restore true antigen Ca and zero antigen latents
        # before calling the AE decoder in sample_formatting.
        # full_simulation only updates antibody positions (mask=ab_only), so antigen
        # positions in gen_samples still hold the initial Gaussian noise — passing that
        # noise to a transformer AE decoder corrupts the antibody reconstruction.
        if self.ab_design_mode and "local_latents" in gen_samples:
            _coords_nm = batch.get("coords_nm")   # [B, n_full, 37, 3] or [n_full, 37, 3]
            _chain_type = batch.get("chain_type")  # [B, n_full] or [n_full]
            if _coords_nm is not None and _chain_type is not None:
                _B = gen_samples["bb_ca"].shape[0]  # batch size (= nsamples in ab_design)
                # Ensure [B, N, 37, 3] / [B, N] — expand if squeezed to single sample
                if _coords_nm.dim() == 3:
                    _coords_nm = _coords_nm.unsqueeze(0).expand(_B, -1, -1, -1)
                if _chain_type.dim() == 1:
                    _chain_type = _chain_type.unsqueeze(0).expand(_B, -1)
                _ag_mask = (_chain_type == 3)  # [B, n_full]
                if _ag_mask.any():
                    _true_ag_ca = _coords_nm[:, :, 1, :]  # [B, n_full, 3]
                    gen_samples["bb_ca"] = torch.where(
                        _ag_mask.unsqueeze(-1), _true_ag_ca, gen_samples["bb_ca"]
                    )
                    # Antigen latents should be zero (AE never trained on antigen)
                    gen_samples["local_latents"] = gen_samples["local_latents"].clone()
                    for _b in range(_B):
                        gen_samples["local_latents"][_b, _ag_mask[_b]] = 0.0

        # Format the generated samples back to proteins
        sample_prots = self.sample_formatting(
            x=gen_samples,
            extra_info=extra_info,
            ret_mode="coors37_n_aatype",
        )
        # Dict with keys `coors` (a37), `residue_type`, and `mask`,
        # shapes [b, n, 37, 3], [b, n], [b, n]

        generation_list = []
        nres = batch.get("nres", sample_prots["coors"].shape[1])

        # For ab_design: build complex (antibody + antigen) for saving
        if self.ab_design_mode:
            # Each sample in the batch may have a different antigen — extract per-sample.
            full_coords_nm = batch.get("coords_nm")   # [B, N, 37, 3] (collate pads to max_len)
            full_chain_type = batch.get("chain_type") # [B, N]
            full_seq = batch.get("residue_type", batch.get("seq"))  # [B, N]

            B_pred = sample_prots["coors"].shape[0]   # = batch_size

            if full_coords_nm is not None and full_chain_type is not None:
                # Guard: if full_simulation squeezed batch dim, restore it
                if full_coords_nm.dim() == 3:
                    full_coords_nm = full_coords_nm.unsqueeze(0).expand(B_pred, -1, -1, -1)
                if full_chain_type.dim() == 1:
                    full_chain_type = full_chain_type.unsqueeze(0).expand(B_pred, -1)
                if full_seq is not None and full_seq.dim() == 1:
                    full_seq = full_seq.unsqueeze(0).expand(B_pred, -1)

                for i in range(B_pred):
                    ct_i  = full_chain_type[i]   # [N]
                    crd_i = full_coords_nm[i]    # [N, 37, 3]
                    seq_i = full_seq[i] if full_seq is not None else None  # [N]

                    # Per-sample antigen extraction
                    ag_mask_i  = (ct_i == 3)
                    ag_coors_i = crd_i[ag_mask_i] * 10.0  # nm → Å, [n_ag, 37, 3]
                    ag_seq_i   = seq_i[ag_mask_i] if seq_i is not None else None
                    ag_chain_i = ct_i[ag_mask_i]           # all 3s

                    # Per-sample antibody length (may differ across batch due to padding)
                    ab_mask_i = (ct_i > 0) & (ct_i < 3)
                    n_ab_i    = int(ab_mask_i.sum().item())

                    ab_coors   = sample_prots["coors"][i, :n_ab_i]         # [n_ab, 37, 3] Å
                    ab_seq_i   = sample_prots["residue_type"][i, :n_ab_i]  # [n_ab]
                    ab_chain_i = ct_i[ab_mask_i]                           # [n_ab], values 1 or 2

                    # Concatenate antibody + antigen; chain_index: heavy=0, light=1, antigen=2
                    complex_coors = torch.cat([ab_coors, ag_coors_i], dim=0)
                    complex_seq   = torch.cat([ab_seq_i, ag_seq_i], dim=0) if ag_seq_i is not None else ab_seq_i
                    complex_chain = torch.cat([ab_chain_i - 1, ag_chain_i - 1], dim=0)

                    generation_list.append((complex_coors, complex_seq, complex_chain))
                return generation_list

        for i in range(sample_prots["coors"].shape[0]):
            generation_list.append(
                (sample_prots["coors"][i, :nres], sample_prots["residue_type"][i, :nres])
            )
        return generation_list

    def sample_formatting(
        self,
        x: Dict[str, Tensor],
        extra_info: Dict[str, Tensor],
        ret_mode: str,
    ):
        """
        Given a batch of b samples x produced by the flow matcher, it returns the samples in the requested format (ret_mode).

        Supports `ret_modes` for:
            - `samples` returns the original sample from the flow matcher, a dictionary[str, Tensor].
            for the data modalities, each with batch shape b.
            - `atom37` returns an Tensor of shape [b, n, 37, 3] just for coordinates.
            - `pdb_string` returns a list of dictionaries {"pdb_string": str, "nres": int}, with one dictionary per sample.
            - `coors37_n_aatype` returns a dictionary with keys `coors` (atom37), `residue_type`, and `mask`, and
            values with shapes [b, n, 37, 3] float, [b, n] int, [b, n] boolean, respectively.

        Args:
            x: sample.
            extra_info: a dict with additional things, including:
                - "mask", whose value is boolean of shape [nsamples, n]
                - ...
            ret_mode: target format, for now only supports atom37.

        Returns:
            Sample x in the requested format.
        """
        data_modes = sorted([dm for dm in self.cfg_exp.product_flowmatcher])
        if data_modes == ["bb_ca"]:
            return self._format_sample_bb_ca(
                x=x, ret_mode=ret_mode, mask=extra_info["mask"]
            )
        elif data_modes == ["bb_ca", "local_latents"]:
            return self._format_sample_local_latents(
                x=x, ret_mode=ret_mode, mask=extra_info["mask"]
            )
        else:
            raise NotImplementedError(f"Format {ret_mode} not implemented")

    def _format_sample_bb_ca(
        self,
        x: Dict[str, torch.Tensor],
        ret_mode: str,
        mask: Bool[torch.Tensor, "b n"],
    ):
        if ret_mode == "samples":
            return x

        if ret_mode == "atom37":
            return trans_nm_to_atom37(x["bb_ca"].float())

        elif ret_mode == "coors37_n_aatype":
            coors = (
                trans_nm_to_atom37(x["bb_ca"].float()) * mask[..., None, None]
            )  # [b, n, 37, 3]
            residue_type = torch.zeros_like(coors)[..., 0, 0] * mask  # [b, n]
            return {
                "coors": coors,  # [b, n, 37, 3]
                "residue_type": residue_type.long(),  # [b, n]
                "mask": mask,  # [b, n]
            }

        elif ret_mode == "pdb_string":
            pdb_strings = []

            coors = (
                trans_nm_to_atom37(x["bb_ca"]).float().detach().cpu().numpy()
            )  # [b, n, 37, 3]
            residue_type = np.zeros_like(coors[:, :, 0, 0])  # [b, n]
            atom37_mask = np.zeros_like(coors[:, :, :, 0])  # [b, n, 37]
            atom37_mask[:, :, 1] = 1.0  # [b, n, 37]
            atom37_mask = atom37_mask * mask[..., None]  # [b, n, 37]
            n = coors.shape[-3]

            for i in range(coors.shape[0]):
                prot = create_full_prot(
                    atom37=coors[i, ...],
                    atom37_mask=atom37_mask[i, ...],
                    aatype=residue_type[i, ...],
                )
                pdb_string = to_pdb(prot=prot)
                pdb_strings.append(
                    {
                        "pdb_string": pdb_string,
                        "nres": n,
                    }
                )
            return pdb_strings

        else:
            raise NotImplementedError(
                f"{ret_mode} format for data modes `[bb_ca]` not implemented"
            )

    def _format_sample_local_latents(
        self,
        x: Dict[str, torch.Tensor],
        ret_mode: str,
        mask: Bool[torch.Tensor, "b n"],
    ):
        """
        Given a batch of b samples consisting on `bb_ca` and `local_latents` this
        returns formatted samples.

        Note: This calls the decoder from the autoencoder, since it needs to go from
        local latent variables to the actual coordinates and sequence.

        Note: The self.autoencoder.decode function (used here) returns a dictoinary like
        {
            "coors_nm": [b, n, 37, 3], already masked
            "residue_type": [b, n], already masked, careful with 0s
            "residue_mask": [b, n]
            "atom_mask": [b, n, 37]
        }

        Args:
            x: sample.
            extra_info: a dict with additional things, including:
                - "mask", whose value is boolean of shape [nsamples, n]
                - ...
            ret_mode: target format, for now only supports atom37.

        Returns:
            Sample x in the requested format.
        """
        output_decoder = self.autoencoder.decode(
            z_latent=x["local_latents"], ca_coors_nm=x["bb_ca"], mask=mask
        )

        if ret_mode == "samples":
            return x

        elif ret_mode == "coors37_n_aatype":
            return {
                "coors": nm_to_ang(output_decoder["coors_nm"]),  # [b, n, 37, 3]
                "residue_type": output_decoder["residue_type"],  # [b, n]
                "mask": output_decoder["residue_mask"],  # [b, n]
            }

        elif ret_mode == "pdb_string":
            pdb_strings = []

            coors_atom_37 = (
                nm_to_ang(output_decoder["coors_nm"]).float().detach().cpu().numpy(),
            )  # [b, n, 37, 3]
            residue_type = output_decoder["residue_type"]  # [b, n]
            atom_mask = output_decoder["atom_mask"]  # [b, n, 37]
            n = coors_atom_37.shape[-3]

            for i in range(atom_mask.shape[0]):
                prot = create_full_prot(
                    atom37=coors_atom_37[i, ...],
                    atom37_mask=atom_mask[i, ...],
                    aatype=residue_type[i, ...],
                )
                pdb_string = to_pdb(prot=prot)
                pdb_strings.append(
                    {
                        "pdb_string": pdb_string,
                        "nres": n,
                    }
                )
            return pdb_strings

        else:
            raise NotImplementedError(
                f"{ret_mode} format for data modes `[bb_ca, latent_locals]` not implemented"
            )
