# Public 143-case test data

This directory contains the novel 143-case test split used by the release:

- 93 conventional antibody complexes (VH–VL); and
- 50 nanobody complexes (VHH).

The cases are split by the dataset pipeline into `after_20250630_novel_ab/`
and `after_20250630_novel_nb/`. Each case directory contains the standardized
`complex.pdb`, `complex.fasta`, `design.fasta`, `antibody.fasta`,
`antigen.fasta`, and `metadata.json` files. `split_json/test.json` lists all
143 sample names.

`processed/test_processed/part_0.pkl` is the atom37 runtime cache consumed by
`AbGenDataset` and `AntibodyDesignDataset`; `_metainfo` deliberately refers to
the shard by its local filename so the data can be moved between machines.
The cache stores coordinates, atom masks, sequence indices, CDR masks,
paratope flags, antigen epitope flags, and the standardized chain layout.

The metadata is structural provenance only: PDB identifier, release date,
resolution, chain identifiers, antibody type, and contact/QC statistics. It
does not contain names, email addresses, credentials, or other personal
information. The current copy has 98 entries labeled `sabdab` and 45 labeled
`cif_to_data` in `metadata.json`; these labels describe the upstream structural
processing source and should not be interpreted as a claim that every case has
the same upstream database provenance.

The structures and metadata retain third-party PDB/SAbDAb provenance. Before
redistributing this directory, users should check the applicable PDB and
SAbDAb attribution and redistribution terms.

## Use with the release

From the release root, the bundled test data can be used directly for
inference:

```bash
DATASET="$PWD/data/143-case" \
PYTHON_BIN=/path/to/python \
bash scripts/rafa_infer.sh
```

The bundled data is a test split, not a replacement for the full train/valid
dataset required by `scripts/rafa_train_rgt.sh`. See
[`docs/training_data.md`](../../docs/training_data.md) for the complete data
contract and preprocessing description.
