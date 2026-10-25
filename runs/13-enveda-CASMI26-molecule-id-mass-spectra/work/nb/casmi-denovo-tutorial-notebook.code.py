# Use the Dependency Manager
# pip install lightning
# pip install tokenizers
# pip install rdkit==2026.3.3

# %%

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from functools import partial
from sklearn.model_selection import train_test_split

from rdkit.Chem import MolFromSmiles, MolToSmiles, MolToInchiKey
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator
import rdkit.rdBase as rkrb
import rdkit.RDLogger as rkl

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import TransformerEncoderLayer, TransformerEncoder
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from tokenizers import Tokenizer, models, trainers
from tokenizers.processors import TemplateProcessing

import lightning as L
from lightning.pytorch import loggers

tqdm.pandas()

# %%

# Competition paths. The input directory is mounted read-only; everything we write goes to
# /kaggle/working, which is what Kaggle collects as the notebook's output.
COMP_DIR = '/kaggle/input/competitions/enveda-CASMI26-molecule-id-mass-spectra'
TRAIN_PATH = os.path.join(COMP_DIR, 'train.parquet')
TEST_PATH = os.path.join(COMP_DIR, 'test.parquet')
SAMPLE_SUBMISSION_PATH = os.path.join(COMP_DIR, 'sample_submission.csv')

WORKING_DIR = '/kaggle/working'
SUBMISSION_PATH = os.path.join(WORKING_DIR, 'submission.csv')

print(os.listdir(COMP_DIR))

# %%

# disable rdkit warning logs
# necessary because we'll be attempting to decode lots of invalid smiles during training
logger = rkl.logger()
logger.setLevel(rkl.ERROR)
rkrb.DisableLog("rdApp.error") 

# %%

NEEDED = ['ingest_lib', 'inchikey14', 'normalized_smiles', 'precursor_mz',
        'ms2_mzs', 'ms2_normalized_intensities']
casmi_all_df = pd.read_parquet(TRAIN_PATH, columns=NEEDED,
                             filters=[('ingest_lib', '!=', 'enveda-180')])

# split into train, val
all_structs = casmi_all_df.inchikey14.unique()
train_structs, val_structs = train_test_split(all_structs, test_size=500, random_state=0)
casmi_train_df = casmi_all_df[casmi_all_df.inchikey14.isin(train_structs)]
casmi_val_df = casmi_all_df[casmi_all_df.inchikey14.isin(val_structs)]

# Cap the training set so a Kaggle session finishes. Measured at ~840 ms/step on a T4 at batch 128
# and MAX_LEN 128, so 200k spectra is ~1,560 steps, roughly 22 minutes. All 1.39M spectra would be
# ~2.5 hours of training before validation. Set to None to use everything.
MAX_TRAIN_SPECTRA = 200_000
if MAX_TRAIN_SPECTRA is not None and len(casmi_train_df) > MAX_TRAIN_SPECTRA:
    casmi_train_df = casmi_train_df.sample(n=MAX_TRAIN_SPECTRA, random_state=0)

# One spectrum per val structure. Validation generates autoregressively with no KV cache, so
# validating all 8,447 val spectra costs more than the training epoch itself.
casmi_val_df = casmi_val_df.drop_duplicates(subset=['inchikey14'], keep='first')

print('split: spectra count, structure count')
print('train:', len(casmi_train_df), casmi_train_df.inchikey14.nunique())
print('val:', len(casmi_val_df), casmi_val_df.inchikey14.nunique())

# %%

# data processing params
# 128, not 512. The collator pads each batch to its longest spectrum, and although the median
# spectrum has only 20 peaks, 95% of batches contain at least one long one -- so at MAX_LEN=512
# nearly every batch ran 512-length attention over mostly padding. Measured on a T4: batch 128 at
# seq 512 needs 16.0 GB and OOMs a 14.56 GB card; at seq 128 it needs 6.4 GB. Peaks are sorted by
# intensity before truncation, so this keeps every peak for 89.5% of spectra and the 127 most
# intense for the rest. Raise to 256 (9.8 GB, 94.8% complete) if you want more of the tail.
MAX_LEN = 128
BPE_PAD_ID = 0
BPE_BOS_ID = 1
BPE_EOS_ID = 2
VOCAB_SIZE = 512

# %%

# process data
## spectrum transform
def process_spectra(df):
    def _process_spectrum(row):
        # sort mzs, intensities
        sort_mask = np.argsort(row.ms2_normalized_intensities)[::-1]
        sorted_mzs = row.ms2_mzs[sort_mask]
        sorted_intensities = row.ms2_normalized_intensities[sort_mask]
    
        # truncate to max len
        truncated_mzs = sorted_mzs[:MAX_LEN - 1] # note: we use MAX_LEN - 1 to account for the precursor, which will be prepended
        truncated_intensities = sorted_intensities[:MAX_LEN - 1]
        
        # normalize intensities
        normalized_intensities = truncated_intensities / truncated_intensities.max()
    
        return truncated_mzs, normalized_intensities

    mzs, ints = zip(*df.progress_apply(_process_spectrum, axis=1))
    df['processed_mzs'] = mzs
    df['processed_intensities'] = ints
    return df

## transform spectra
print('processing spectra...')
casmi_train_df = process_spectra(casmi_train_df)
casmi_val_df = process_spectra(casmi_val_df)

## smiles bpe tokenizer
def train_bpe_tokenizer(structures):
    os.environ["TOKENIZERS_PARALLELISM"] = "true"     # turn on tokenizers parallelism to speed up bpe training
    
    full_alphabet = list(set(list("".join(structures))))
    tokenizer = Tokenizer(models.BPE())
    special_tokens = [
        ("<pad>", BPE_PAD_ID),
        ("<s>", BPE_BOS_ID),
        ("</s>", BPE_EOS_ID),
    ]
    tokenizer.post_processor = TemplateProcessing(
        single="<s> $A </s>", special_tokens=special_tokens
    )
    special_tokens = ["<pad>", "<s>", "</s>"]
    
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        initial_alphabet=full_alphabet,
        special_tokens=special_tokens,
        show_progress=True,
    )
    train_iterator = structures
    tokenizer.train_from_iterator(train_iterator, trainer)
    
    os.environ["TOKENIZERS_PARALLELISM"] = "false"     # turn off tokenizers parallelism
    return tokenizer

def tokenize_df_smiles(df, tokenizer):
    df['bpe_tokenized_smiles'] = df.normalized_smiles.progress_apply(
        lambda x: np.array(tokenizer.encode(x).ids, dtype=int)
    )
    return df

## tokenize smiles
print('training bpe...')
bpe_train_structures = list(casmi_train_df.normalized_smiles.unique())
tokenizer = train_bpe_tokenizer(bpe_train_structures)

print('processing smiles...')
casmi_train_df = tokenize_df_smiles(casmi_train_df, tokenizer=tokenizer)
casmi_val_df = tokenize_df_smiles(casmi_val_df, tokenizer=tokenizer)

# utility for decoding bpe smiles
def decode_tokenized_smiles(bpe_smiles_tokens, tokenizer=tokenizer):
    return tokenizer.decode(bpe_smiles_tokens).replace(' ', '')

# %%

# datamodule params
PRECURSOR_INTENSITY = 2.0
MZ_PAD_ID = 0
INTENSITY_PAD_ID = 0
BATCH_SIZE = 128
# A Kaggle T4 session has 4 vCPUs. This was defined but never passed to a DataLoader, which is
# what Lightning's 'does not have many workers' warning was about.
NUM_WORKERS = 3

# %%

class PandasDataset(Dataset):
    def __init__(self, df, column_list, shuffle=True):
        # Prediction keeps the original row order so rows stay aligned with their molecule_id.
        self.df = df[column_list].sample(frac=1, random_state=0) if shuffle else df[column_list]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.df.iloc[idx].to_dict()

# %%

# define datamodule class
class CASMIDataModule(L.LightningDataModule):
    def __init__(self, batch_size, predict_batch_size=None):
        super().__init__()
        self.batch_size = batch_size
        # Generation expands the batch by n_samples, so prediction needs a far smaller batch than
        # training. See the predict cell for the arithmetic.
        self.predict_batch_size = predict_batch_size or batch_size

    def setup(self, stage):
        columns = ['precursor_mz', 'processed_mzs', 'processed_intensities', 'bpe_tokenized_smiles', 'normalized_smiles']
        if stage == "fit":
            self.train_df = PandasDataset(casmi_train_df, column_list=columns)
            self.val_df = PandasDataset(casmi_val_df, column_list=columns)
        elif stage == "validate":
            self.val_df = PandasDataset(casmi_val_df, column_list=columns)
        elif stage == "predict":
            # test.parquet carries no labels, so only the spectrum columns exist. molecule_id rides
            # along because scoring is per molecule, not per spectrum, and a molecule's spectra have
            # to be grouped back together afterwards.
            self.predict_df = PandasDataset(
                casmi_test_df,
                column_list=['molecule_id', 'precursor_mz', 'processed_mzs', 'processed_intensities'],
                shuffle=False,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_df,
            batch_size=self.batch_size,
            collate_fn=self.get_collator('fit'),
            num_workers=NUM_WORKERS,
            persistent_workers=NUM_WORKERS > 0,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_df,
            batch_size=self.batch_size,
            collate_fn=self.get_collator('validate'),
            num_workers=NUM_WORKERS,
            persistent_workers=NUM_WORKERS > 0,
            pin_memory=True,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_df,
            batch_size=self.predict_batch_size,
            collate_fn=self.get_collator('predict'),
            num_workers=NUM_WORKERS,
            persistent_workers=NUM_WORKERS > 0,
            pin_memory=True,
        )

    def get_collator(self, stage):
        return partial(self._collator, stage=stage)

    def _collator(self, data, stage='fit'):
        mzs = [[row['precursor_mz']] + row['processed_mzs'].tolist() for row in data]  # prepend precursor to m/zs
        ints = [[PRECURSOR_INTENSITY] + row['processed_intensities'].tolist() for row in data]  # prepend 2. to intensities
        if stage == 'fit' or stage == 'validate':
            labels = [row['bpe_tokenized_smiles'] for row in data]

        # collate peaks
        max_peak_len = max([len(x) for x in mzs])
        mzs = [list(x) + [MZ_PAD_ID] * (max_peak_len - len(x)) for x in mzs]
        ints = [list(x) + [INTENSITY_PAD_ID] * (max_peak_len - len(x)) for x in ints]
        mz_array = torch.tensor(mzs)
        intensity_array = torch.tensor(ints)
        attention_mask = torch.where(mz_array == MZ_PAD_ID, 0, 1)

        # collate labels
        batch = {}
        if stage == 'fit' or stage == 'validate':
            max_smiles_len = max([len(x) for x in labels])
            labels = [list(x) + [BPE_PAD_ID] * (max_smiles_len - len(x)) for x in labels]
            label_array = torch.tensor(labels)

            batch['mzs'] = mz_array
            batch['intensities'] = intensity_array
            batch['attention_mask'] = attention_mask
            batch['structure_tokens'] = label_array # tokenized labels for training
        if stage == 'validate':
            batch['smiles'] = [row['normalized_smiles'] for row in data] # include original smiles for validation metrics
        else:
            batch = batch | {
                'mzs': mz_array,
                'intensities': intensity_array,
                'attention_mask': attention_mask,
            }
        if stage == 'predict':
            batch['molecule_id'] = [row['molecule_id'] for row in data]
        return batch

# define datamodule
datamodule = CASMIDataModule(batch_size=BATCH_SIZE)
datamodule.setup('fit')

# %%

# confirm the dataloader works
val_dl = datamodule.val_dataloader()
for batch in val_dl:
    break
batch

# %%

# define tanimoto similarity for metrics

## we'll use Morgan radius 2 fps to define "tanimoto similarity" in this notebook
### note: radius 2, counts=True, sparse (no hashing)
fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2)

def tanimoto_smiles(smiles1, smiles2):
    mol1, mol2 = MolFromSmiles(smiles1), MolFromSmiles(smiles2)
    if mol1 is None or mol2 is None:
        return None
    fp1, fp2 = fp_gen.GetSparseFingerprint(mol1), fp_gen.GetSparseFingerprint(mol2)
    return DataStructs.TanimotoSimilarity(fp1, fp2)

# %%

# define module classes

### peak embedder
class PeakEmbedder(nn.Module):
    """embed (m/z, intensity) peak pairs with sinusoidal embeddings from Voronov et al"""
    def __init__(self, d_model, dropout, sin_dim=None, mz_log_lims=(-2., 3.), mz_log_power=1.0):
        super().__init__()
        sin_dim = sin_dim if sin_dim is not None else d_model
        self.dropout = dropout

        wavelength = torch.pow(
            10,
            (mz_log_lims[1] - mz_log_lims[0]) * torch.pow(
                torch.linspace(0, 1, int(sin_dim / 2)), 
                mz_log_power,
            ) + mz_log_lims[0],
        )
        frequency = 2 * np.pi / wavelength
        self._frequency = nn.Parameter(frequency, requires_grad=False)
        self._ff_block_1 = nn.Sequential(*[
            nn.Linear(sin_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout),
        ])
        self._ff_block_2 = nn.Sequential(*[
            nn.Linear(d_model+1, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout),
        ])

    def forward(self, mz_tensor, intensity_tensor):
        # "mz embedding" embeds just the m/z value
        omega_mz = self._frequency.view(
            *(1 for _ in range(mz_tensor.ndim)), -1
        ) * mz_tensor.unsqueeze(-1)
        sin = torch.sin(omega_mz)
        cos = torch.cos(omega_mz)
        mz_vecs = torch.cat([sin, cos], dim=-1)
        mz_embeds = self._ff_block_1(mz_vecs)

        # "peak embedding" embeds both the m/z and intensity
        peak_embeds = torch.cat([mz_embeds, intensity_tensor.unsqueeze(2)], dim=2)
        return self._ff_block_2(peak_embeds)

        
### spectrum encoder
class SpectrumEncoder(nn.Module):
    def __init__(self, embed_dim, n_heads, n_layers, dim_feedforward=None, dropout=0.1, activation='relu'):
        super().__init__()
        dim_feedforward = dim_feedforward if dim_feedforward is not None else 4 * embed_dim
        self.encoder = TransformerEncoder(
            TransformerEncoderLayer(
                embed_dim,
                n_heads,
                dim_feedforward=dim_feedforward,
                batch_first=True,
                dropout=dropout,
                activation=activation,
            ),
            n_layers,
        )
        self.init_weights()

    def forward(self, sequence_input, attention_mask):
        pad_mask = (attention_mask == 0)
        return self.encoder(sequence_input, src_key_padding_mask=pad_mask)

    def init_weights(self):
        for layer in self.encoder.layers:
            for name, mod in layer.named_modules():
                if isinstance(mod, nn.Linear):
                    nn.init.xavier_uniform_(mod.weight)
                    if mod.bias is not None:
                        nn.init.constant_(mod.bias, 0.0)
                elif isinstance(mod, nn.LayerNorm):
                    nn.init.constant_(mod.weight, 1.0)
                    if mod.bias is not None:
                        nn.init.constant_(mod.bias, 0.0)


### smiles decoder
class SmilesDecoder(nn.Module):
    def __init__(
        self, 
        embed_dim, 
        vocab_size, 
        n_layers, 
        n_heads, 
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        dim_feedforward=None, 
        dropout=0.1, 
        activation='gelu',
        validate_n_samples=10,
        predict_n_samples=25
    ):
        super().__init__()
        self.bos_token_id = bos_token_id
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.embed_dim = embed_dim
        dim_feedforward = dim_feedforward if dim_feedforward is not None else 4 * self.embed_dim
        self.n_samples_per_stage = {'validate': validate_n_samples, 'predict': predict_n_samples}
        
        self.wte = nn.Embedding(vocab_size, self.embed_dim, padding_idx=self.pad_token_id)
        self.decoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=self.embed_dim,
                dim_feedforward=dim_feedforward,
                nhead=self.n_heads,
                dropout=dropout,
                activation=activation,
                batch_first=True,
            ),
            num_layers=self.n_layers,
        )
        self.lm_head = nn.Linear(self.embed_dim, vocab_size, bias=False)
        self.init_weights()

    def init_weights(self):
        torch.nn.init.zeros_(self.lm_head.weight) # zero out classifier weights to start
        torch.nn.init.normal_(self.wte.weight, mean=0.0, std=1.0)
        # if self.wte.weight.device.type == "cuda": # save memory by casting embeddings to bf16
        #     self.wte.to(dtype=torch.bfloat16)
        for layer in self.decoder.layers:
            for name, mod in layer.named_modules():
                if isinstance(mod, nn.Linear):
                    nn.init.xavier_uniform_(mod.weight)
                    if mod.bias is not None:
                        nn.init.constant_(mod.bias, 0.0)
                elif isinstance(mod, nn.LayerNorm):
                    nn.init.constant_(mod.weight, 1.0)
                    if mod.bias is not None:
                        nn.init.constant_(mod.bias, 0.0)

    def forward(self, idx, encoder_outputs, encoder_attention_mask, structure_tokens=None):
        encoder_pad_mask = (encoder_attention_mask == 0)
        softcap = 15
        tgt = self.wte(idx)
        
        if structure_tokens is not None:
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.shape[1])
        else:
            tgt_mask = None
            
        x = self.decoder(
            tgt=tgt,
            memory=encoder_outputs,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=encoder_pad_mask,
        )
        logits = self.lm_head(x)
        logits = softcap * torch.tanh(logits / softcap)
        logits = logits.float()

        output_dict = {
            'logits': logits,
        }
        if structure_tokens is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                structure_tokens.view(-1),
                ignore_index=self.pad_token_id,
                reduction='mean'
            )
            output_dict['loss'] = loss
        return output_dict
    
    @torch.inference_mode()
    def generate(
        self, 
        encoder_states,
        encoder_mask,
        n_samples=1,
        max_new_tokens=50,
        temperature=1.0,
    ):
        """simple sampling based generation with temperature control"""
        B, S, D = encoder_states.shape

        idx = torch.ones_like(encoder_states[:, :1, 0]).long() * self.bos_token_id
        logprob = torch.zeros_like(encoder_states[:, :1, 0])

        idx = idx.expand(-1, n_samples).reshape(B * n_samples, -1)
        idx_next = idx
        logprob_expanded = logprob.expand(-1, n_samples).reshape(B * n_samples, -1)
        eos_generated = idx[:, -1] == self.eos_token_id

        encoder_states = (
            encoder_states.unsqueeze(1)
            .expand(-1, n_samples, -1, -1)
            .reshape(1, B * n_samples, S, D)
            .squeeze(0)
        )
        encoder_mask = (
            encoder_mask.unsqueeze(1)
            .expand(-1, n_samples, -1)
            .reshape(1, B * n_samples, S)
            .squeeze(0)
        )
        for _ in range(max_new_tokens):
            # forward the model to get the logits for the index in the sequence
            logits = self.forward(
                idx_next,
                encoder_states,
                encoder_mask,
                structure_tokens=None,
            )['logits']
            # pluck the logits at the final step and scale by desired temperature
            rescaled_logits = (logits[:, -1, :] / temperature).log_softmax(dim=-1)
            logits = logits[:, -1, :].log_softmax(dim=-1)

            if eos_generated.sum() > 0:
                logits[eos_generated, :] = -float("Inf")
                logits[eos_generated, self.eos_token_id] = 0

                rescaled_logits[eos_generated, :] = -float("Inf")
                rescaled_logits[eos_generated, self.eos_token_id] = 0

            # sample from the distribution
            idx_next = torch.multinomial(rescaled_logits.softmax(dim=-1), num_samples=1)
            token_logits_next = torch.take_along_dim(logits, idx_next, dim=1)

            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)
            logprob_expanded = torch.cat((logprob_expanded, token_logits_next), dim=1)
            eos_generated = idx[:, -1] == self.eos_token_id
            if eos_generated.sum() == len(eos_generated):
                break

            if (idx == self.eos_token_id).max(axis=-1).values.sum() == idx.shape[0]:
                break

        _, L = idx.shape
        return idx.reshape(B, n_samples, L), logprob_expanded.sum(dim=-1).reshape(
            B, n_samples
        )

# %%

# orchestrate modules into a single Pytorch Lightning Model

class DeNovoLightningModel(L.LightningModule):
    def __init__(self, kwargs):
        super().__init__()
        self.peak_embedder = PeakEmbedder(**kwargs['peak_embedder'])
        self.spectrum_encoder = SpectrumEncoder(**kwargs['spectrum_encoder'])
        self.smiles_decoder = SmilesDecoder(**kwargs['smiles_decoder'])
        self.optimizer_config = kwargs['optimizer']

    @classmethod
    def load_from_ckpt(cls, ckpt_path, cfg, **kwargs):
        cp = torch.load(
            load_config.ckpt_path,
            map_location=torch.device("cpu"),
            weights_only=False,
        )
        obj = cls(cp["hyper_parameters"])
        obj.load_state_dict(cp["state_dict"], strict=False)
        return obj
    
    def forward(self, batch, stage, return_encoder_states=False):
        encoded_spectra = self.spectrum_encoder(
            self.peak_embedder(batch['mzs'], batch['intensities']),
            batch['attention_mask']
        )
        outputs = {}
        if stage in ('fit', 'validate'):
            outputs = self.smiles_decoder(
                batch['structure_tokens'],
                encoded_spectra,
                batch['attention_mask'],
                structure_tokens=batch['structure_tokens']
            )
        if stage in ('validate', 'predict'):
            tokens, scores = self.smiles_decoder.generate(
                encoded_spectra,
                batch['attention_mask'],
                n_samples=self.smiles_decoder.n_samples_per_stage[stage],
                temperature=1., # experiment with this
            )
            outputs['generated_tokens'] = tokens
            outputs['generated_scores'] = scores
        if return_encoder_states:
            outputs['encoder_states'] = encoded_spectra
        return outputs

    def training_step(self, batch):
        outputs = self(batch, "fit")
        loss = outputs['loss']
        self.log(
            "train_loss",
            loss,
            sync_dist=True,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            logger=True,
            batch_size=batch['mzs'].shape[0],
        )
        return loss

    def validation_step(self, batch):
        forward_outputs = self(batch, "validate", return_encoder_states=True)
        loss = forward_outputs['loss']
        generated_tokens, generated_scores = self.smiles_decoder.generate(
            forward_outputs['encoder_states'],
            batch['attention_mask'],
            n_samples=10,
            temperature=1.0,
        )
        generation_metrics = self.compute_val_metrics(
            generated_tokens, 
            generated_scores,
            batch['smiles']
        )
            
        metrics_dict = {
            "val_loss": loss,
            **{f"val_{k}": v for k, v in generation_metrics.items()},
        }
        self.log_dict(
            metrics_dict,
            add_dataloader_idx=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=(batch['mzs'].shape[0]),
        )

    def predict_step(self, batch):
        # forward() already generates for stage='predict'; generating a second time here just did
        # the same expensive sampling twice.
        outputs = self(batch, "predict")
        return batch['molecule_id'], outputs['generated_tokens'], outputs['generated_scores']
    
    def compute_val_metrics(self, generated_tokens, generated_scores, labels):
        decoded_smiles = [
            [decode_tokenized_smiles(tokens.tolist()) for tokens in token_seqs]
            for token_seqs in generated_tokens
        ]
        score_masks = [scores.argsort(descending=True) for scores in generated_scores]
        sorted_valid_smiles = [
            [smiles_seq[idx] for idx in mask if smiles_seq[idx] != '' and MolFromSmiles(smiles_seq[idx]) is not None]
            for smiles_seq, mask in zip(decoded_smiles, score_masks)
        ]
        top_valid_smiles = [
            smiles_seq[0] 
            if len(smiles_seq) > 0 else None
            for smiles_seq in sorted_valid_smiles
        ]
        
        def _sm_to_ikey(smiles):
            return MolToInchiKey(MolFromSmiles(smiles)).split('-')[0]
        ikey_match = [
            _sm_to_ikey(smiles) == _sm_to_ikey(label)
            if smiles is not None else False
            for smiles, label in zip(top_valid_smiles, labels)
        ]
        tanimotos = [
            tanimoto_smiles(smiles, label)
            if smiles is not None else 0.0
            for smiles, label in zip(top_valid_smiles, labels)
        ]
        
        frac_ikey_match = np.mean(ikey_match)
        frac_valid_smiles = np.mean([x is not None for x in top_valid_smiles])
        mean_tanimoto = np.mean(tanimotos)

        return {
            'valid_smiles': frac_valid_smiles,
            'tanimoto': mean_tanimoto,
            'exact match': frac_ikey_match,
        }
    
    def configure_optimizers(self):
        return AdamW(
            self.parameters(),
            lr=self.optimizer_config['lr'],
        )

# %%

# model hyperparams
EMBED_DIM = 768
DIM_FEEDFORWARD = 3072
DROPOUT = 0.1

ENCODER_N_HEADS = 12
ENCODER_N_LAYERS = 6
ENCODER_ACTIVATION = 'gelu'

DECODER_N_HEADS = 12
DECODER_N_LAYERS = 6
DECODER_ACTIVATION = 'gelu'

N_SAMPLES_VAL = 10
N_SAMPLES_PRED = 25

LEARNING_RATE = 5e-5

# %%

model_params = {
    'peak_embedder': {
        'd_model': EMBED_DIM, 
        'dropout': DROPOUT, 
    },
    'spectrum_encoder': {
        'embed_dim': EMBED_DIM,
        'n_heads': ENCODER_N_HEADS,
        'n_layers': ENCODER_N_LAYERS,
        'dim_feedforward': DIM_FEEDFORWARD,
        'dropout': DROPOUT,
        'activation': ENCODER_ACTIVATION,
    },
    'smiles_decoder': {
        'embed_dim': EMBED_DIM, 
        'vocab_size': VOCAB_SIZE, 
        'n_layers': DECODER_N_LAYERS, 
        'n_heads': DECODER_N_HEADS, 
        'pad_token_id': BPE_PAD_ID,
        'bos_token_id': BPE_BOS_ID,
        'eos_token_id': BPE_EOS_ID,
        'dim_feedforward': DIM_FEEDFORWARD, 
        'dropout': DROPOUT, 
        'activation': DECODER_ACTIVATION,
        'validate_n_samples': N_SAMPLES_VAL,
        'predict_n_samples': N_SAMPLES_PRED,
    },
    'optimizer': {
        'lr': LEARNING_RATE,
    },
}
model = DeNovoLightningModel(model_params)

# %%

# confirm the forward pass works
# Slice the batch first. The model is still on CPU here -- Lightning only moves it inside fit() --
# so this runs in fp32 and keeps the whole autograd graph alive for the backward cell below.
# At batch=128 with 512 peaks that is ~66 GB of activations, which OOMs a Kaggle session; 8 rows
# is ~0. Training itself is fine at the full batch: it runs on the GPU in mixed precision and
# frees activations every step.
SMOKE_TEST_ROWS = 8
smoke_batch = {k: v[:SMOKE_TEST_ROWS] for k, v in batch.items()}

outputs = model(smoke_batch, 'fit')
outputs['logits'].shape

# %%

# confirm the backward pass works
outputs['loss'].backward()

# %%

# # confirm the validation step works (including metrics)
# model.validation_step(batch)

# %%

# logger
RUN_NAME = 'tutorial_notebook'
LOGGER_PATH = os.path.join(WORKING_DIR, 'lightning_logs')
logger = loggers.TensorBoardLogger(
    save_dir=LOGGER_PATH, 
    name=RUN_NAME,
)

# %%

# checkpoint
CHECKPOINT_METRIC_TO_MONITOR = 'val_tanimoto' 
CHECKPOINT_MONITOR_MODE = 'max' # or 'min'

CHECKPOINT_DIR = os.path.join(WORKING_DIR, 'model_checkpoints')
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, RUN_NAME)

checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
    dirpath=CHECKPOINT_PATH,
    monitor=CHECKPOINT_METRIC_TO_MONITOR,
    mode=CHECKPOINT_MONITOR_MODE,
    every_n_epochs=1,
    save_top_k=5,
    save_last=True,
    filename=f"{{epoch}}-{{step}}-{{{CHECKPOINT_METRIC_TO_MONITOR}:.3f}}",
)

# %%

MAX_EPOCHS = 1

PRECISION = 'bf16-mixed' if torch.cuda.is_bf16_supported() else '16-mixed'

trainer = L.Trainer(
  max_epochs=MAX_EPOCHS,
  devices=1,
  accelerator='auto',
  precision=PRECISION,
  log_every_n_steps=100,
  check_val_every_n_epoch=1,
  accumulate_grad_batches=3,
  logger=[logger],
  callbacks=[checkpoint_callback],
  num_sanity_val_steps=2,
)

# %%

# train
# RESUME_FROM_CHECKPOINT_PATH = os.path.join(CHECKPOINT_PATH, 'ckpt_name.ckpt')
trainer.fit(
    model,
    datamodule=datamodule,
    # ckpt_path=RESUME_FROM_CHECKPOINT_PATH,
    weights_only=False,
)

# %%

# Load the test spectra and run them through exactly the same processing as the training spectra.
casmi_test_df = pd.read_parquet(TEST_PATH)
print('test:', len(casmi_test_df), 'spectra,', casmi_test_df.molecule_id.nunique(), 'molecules')

casmi_test_df = process_spectra(casmi_test_df)

# generate() expands the batch by n_samples: it materialises a real copy of the encoder memory at
# (batch * n_samples, peaks, embed_dim). At the training batch of 128 with 25 samples that is 3,200
# sequences, which overruns a T4 even at MAX_LEN=128. 32 gives 800 sequences: ~0.3 GB for the
# encoder copy and ~0.2 GB per decoder layer of cross-attention.
PREDICT_BATCH_SIZE = 32
datamodule.predict_batch_size = PREDICT_BATCH_SIZE
datamodule.setup('predict')

# %%

# One device for prediction: multi-GPU splits batches across processes, which would make stitching
# the results back together needlessly awkward for a few thousand spectra.
predict_trainer = L.Trainer(accelerator='auto', devices=1, precision='bf16-mixed', logger=False)
predictions = predict_trainer.predict(model, dataloaders=datamodule.predict_dataloader())

# %%

# Build the submission.
#
# Scoring is per molecule, not per spectrum, so every spectrum of a molecule votes into one ranked
# list: candidates are pooled across the molecule's spectra, dropped if RDKit cannot parse them,
# deduplicated on the first InChIKey block (the metric ignores stereochemistry, so two spellings of
# one skeleton would otherwise waste a slot), and kept in descending generation score.
from collections import defaultdict

N_GUESSES = 25
FALLBACK_SMILES = 'CCO'  # every molecule_id must carry at least one guess

best_by_molecule = defaultdict(dict)  # molecule_id -> {inchikey14: (score, smiles)}
for molecule_ids, token_batch, score_batch in predictions:
    for molecule_id, token_seqs, scores in zip(molecule_ids, token_batch, score_batch):
        for tokens, score in zip(token_seqs, scores):
            smiles = decode_tokenized_smiles(tokens.tolist())
            mol = MolFromSmiles(smiles) if smiles else None
            if mol is None:
                continue
            inchikey14 = MolToInchiKey(mol).split('-')[0]
            candidate = (float(score), MolToSmiles(mol))
            if candidate > best_by_molecule[molecule_id].get(inchikey14, (-np.inf, '')):
                best_by_molecule[molecule_id][inchikey14] = candidate

def top_guesses(molecule_id):
    ranked = sorted(best_by_molecule.get(molecule_id, {}).values(), reverse=True)
    guesses = [smiles for _, smiles in ranked[:N_GUESSES]]
    return ';'.join(guesses) if guesses else FALLBACK_SMILES

# Starting from the sample submission guarantees the right molecule_ids, in the right order.
submission = pd.read_csv(SAMPLE_SUBMISSION_PATH)
submission['smiles'] = submission.molecule_id.apply(top_guesses)
submission.to_csv(SUBMISSION_PATH, index=False)

print(f'wrote {len(submission)} rows to {SUBMISSION_PATH}')
n_guesses = submission.smiles.str.split(';').map(len)
print(f'guesses per molecule: median {int(n_guesses.median())}, max {n_guesses.max()}')
submission.head()

# %%

