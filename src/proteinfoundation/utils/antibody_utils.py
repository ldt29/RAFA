#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Antibody-specific utilities for La-Proteina.

This module contains VOCAB, IMGT, and Chothia classes adapted from flowab.
"""

from typing import Dict, List, Tuple
import numpy as np


class IMGT:
    """IMGT antibody numbering definition."""
    # heavy chain
    HFR1 = (1, 26)
    HFR2 = (39, 55)
    HFR3 = (66, 104)
    HFR4 = (118, 129)

    H1 = (27, 38)
    H2 = (56, 65)
    H3 = (105, 117)

    # light chain
    LFR1 = (1, 26)
    LFR2 = (39, 55)
    LFR3 = (66, 104)
    LFR4 = (118, 129)

    L1 = (27, 38)
    L2 = (56, 65)
    L3 = (105, 117)

    Hconserve = {
        23: ['CYS'],
        41: ['TRP'],
        104: ['CYS']
    }

    Lconserve = {
        23: ['CYS'],
        41: ['TRP'],
        104: ['CYS']
    }


class Chothia:
    """Chothia antibody numbering definition."""
    # heavy chain
    HFR1 = (1, 25)
    HFR2 = (33, 51)
    HFR3 = (57, 94)
    HFR4 = (103, 113)

    H1 = (26, 32)
    H2 = (52, 56)
    H3 = (95, 102)

    # light chain
    LFR1 = (1, 23)
    LFR2 = (35, 49)
    LFR3 = (57, 88)
    LFR4 = (98, 107)

    L1 = (24, 34)
    L2 = (50, 56)
    L3 = (89, 97)

    Hconserve = {
        92: ['CYS']
    }

    Lconserve = {
        88: ['CYS']
    }


class AminoAcid:
    """Amino acid class."""
    def __init__(self, symbol: str, abrv: str, sidechain: List[str], idx=0):
        self.symbol = symbol
        self.abrv = abrv
        self.idx = idx
        self.sidechain = sidechain

    def __str__(self):
        return f'{self.idx} {self.symbol} {self.abrv} {self.sidechain}'


class VOCAB:
    """
    Vocabulary class for antibody data processing.
    Contains amino acid definitions and atom mappings.
    """

    MAX_ATOM_NUMBER = 14   # 4 backbone atoms + up to 10 sidechain atoms

    def __init__(self):
        self.backbone_atoms = ['N', 'CA', 'C', 'O']
        self.PAD, self.MASK = '#', '*'
        self.BOA, self.BOH, self.BOL = '&', '+', '-' # begin of antigen, heavy chain, light chain

        # Special tokens
        self.special_tokens = [self.PAD, self.MASK, self.BOA, self.BOH, self.BOL]

        # Amino acid symbols
        self.aa_symbols = ['G', 'A', 'V', 'L', 'I', 'F', 'W', 'Y', 'D', 'H',
                          'N', 'E', 'K', 'Q', 'M', 'R', 'S', 'T', 'C', 'P']

        # Full amino acid list including specials
        self.amino_acids = [self.PAD, self.MASK, self.BOA, self.BOH, self.BOL] + self.aa_symbols

        # Symbol to index mapping
        self._symbol_to_idx = {sym: i for i, sym in enumerate(self.amino_acids)}

        # Index to symbol mapping
        self._idx_to_symbol = {i: sym for i, sym in enumerate(self.amino_acids)}

        # Sidechain atoms for each amino acid
        self.sidechain_map = {
            'G': [],
            'A': ['CB'],
            'V': ['CB', 'CG1', 'CG2'],
            'L': ['CB', 'CG', 'CD1', 'CD2'],
            'I': ['CB', 'CG1', 'CG2', 'CD1'],
            'F': ['CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'],
            'W': ['CB', 'CG', 'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2'],
            'Y': ['CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OH'],
            'D': ['CB', 'CG', 'OD1', 'OD2'],
            'H': ['CB', 'CG', 'ND1', 'CD2', 'CE1', 'NE2'],
            'N': ['CB', 'CG', 'OD1', 'ND2'],
            'E': ['CB', 'CG', 'CD', 'OE1', 'OE2'],
            'K': ['CB', 'CG', 'CD', 'CE', 'NZ'],
            'Q': ['CB', 'CG', 'CD', 'OE1', 'NE2'],
            'M': ['CB', 'CG', 'SD', 'CE'],
            'R': ['CB', 'CG', 'CD', 'NE', 'CZ', 'NH1', 'NH2'],
            'S': ['CB', 'OG'],
            'T': ['CB', 'OG1', 'CG2'],
            'C': ['CB', 'SG'],
            'P': ['CB', 'CG', 'CD'],
        }

        # Special sidechain markers
        self.atom_pad = 'p'
        self.atom_mask = 'm'
        self.atom_pos_mask = 'm'
        self.atom_pos_bb = 'b'
        self.atom_pos_pad = 'p'

        # Amino acid abbreviation mapping (PDB 3-letter to 1-letter)
        self.abrv2idx = {
            'GLY': 0, 'ALA': 1, 'VAL': 2, 'LEU': 3, 'ILE': 4,
            'PHE': 5, 'TRP': 6, 'TYR': 7, 'ASP': 8, 'HIS': 9,
            'ASN': 10, 'GLU': 11, 'LYS': 12, 'GLN': 13, 'MET': 14,
            'ARG': 15, 'SER': 16, 'THR': 17, 'CYS': 18, 'PRO': 19,
        }

    def abrv_to_symbol(self, abrv: str) -> str:
        """Convert 3-letter abbreviation to 1-letter symbol."""
        idx = self.abrv_to_idx(abrv)
        if idx is None:
            return None
        return self.aa_symbols[idx]

    def abrv_to_idx(self, abrv: str) -> int:
        """Convert 3-letter abbreviation to index."""
        abrv = abrv.upper()
        return self.abrv2idx.get(abrv, None)

    def symbol_to_idx(self, symbol: str) -> int:
        """Convert amino acid symbol to index."""
        if symbol in self._symbol_to_idx:
            return self._symbol_to_idx[symbol]
        return self._symbol_to_idx[self.MASK]  # Return MASK index for unknown

    def idx_to_symbol(self, idx: int) -> str:
        """Convert index to amino acid symbol."""
        if idx in self._idx_to_symbol:
            return self._idx_to_symbol[idx]
        return self.MASK

    def get_num_amino_acid_type(self) -> int:
        """Get the number of amino acid types (20 standard + 5 specials)."""
        return len(self.amino_acids)

    def get_backbone_atoms(self) -> List[str]:
        """Get backbone atom names."""
        return self.backbone_atoms

    def get_sidechain_atoms(self, aa_symbol: str) -> List[str]:
        """Get sidechain atom names for an amino acid."""
        return self.sidechain_map.get(aa_symbol, [])

    def get_atom_pad_idx(self) -> int:
        """Get index for atom padding token."""
        return self.symbol_to_idx(self.atom_pad)

    def get_atom_mask_idx(self) -> int:
        """Get index for atom mask token."""
        return self.symbol_to_idx(self.atom_mask)

    def get_atom_pos_mask_idx(self) -> int:
        """Get index for atom position mask token."""
        return self.symbol_to_idx(self.atom_pos_mask)

    def get_atom_pos_bb_idx(self) -> int:
        """Get index for backbone atom position token."""
        return self.symbol_to_idx(self.atom_pos_bb)

    def get_atom_pos_pad_idx(self) -> int:
        """Get index for atom position padding token."""
        return self.symbol_to_idx(self.atom_pos_pad)


# Global VOCAB instance
VOCAB = VOCAB()


# =============================================================================
# Contact distance constant
# =============================================================================

CONTACT_DIST = 10.0  # 10 Å — aligns with iRMS interface definition (was 6.6)


# =============================================================================
# Residue, Peptide, Protein classes
# =============================================================================

class Residue:
    """Represents a single residue with coordinates."""

    def __init__(self, symbol: str, coordinate: Dict, residue_id: Tuple) -> None:
        self.symbol = symbol
        self.coordinate = coordinate
        self.id = residue_id
        # Get sidechain atoms (exclude backbone)
        backbone = ['N', 'CA', 'C', 'O', 'CB']
        self.sidechain = [k for k in coordinate.keys() if k not in backbone]

    def get_symbol(self):
        return self.symbol

    def get_id(self):
        return self.id

    def get_backbone_coord_map(self):
        return {k: v for k, v in self.coordinate.items() if k in ['N', 'CA', 'C', 'O']}

    def get_sidechain_coord_map(self):
        backbone = ['N', 'CA', 'C', 'O']
        return {k: v for k, v in self.coordinate.items() if k not in backbone}


class Peptide:
    """Represents a peptide/protein chain."""

    def __init__(self, chain_id: str, residues: List[Residue]) -> None:
        self.id = chain_id
        self.residues = residues
        self.seq = ''.join([r.get_symbol() for r in residues])

    def __len__(self):
        return len(self.residues)

    def get_residue(self, idx: int) -> Residue:
        return self.residues[idx]

    def get_seq(self) -> str:
        return self.seq

    def get_id(self):
        return self.id

    def get_chain(self, idx: int):
        return self.get_residue(idx)


class Protein:
    """Represents a protein with multiple chains."""

    def __init__(self, pdb_id: str, peptides: Dict[str, Peptide]) -> None:
        self.pdb_id = pdb_id
        self.peptides = peptides

    @classmethod
    def from_pdb(cls, pdb_path: str):
        """Load protein structure from PDB file."""
        from Bio.PDB import PDBParser
        import os

        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('anonym', pdb_path)
        pdb_id = structure.header.get('idcode', '').upper().strip()
        if pdb_id == '':
            # deduce from file name
            pdb_id = os.path.split(pdb_path)[1].split('.')[0] + '(filename)'

        peptides = {}
        for chain in structure.get_chains():
            _id = chain.get_id()
            residues = []
            has_non_residue = False
            for residue in chain:
                abrv = residue.get_resname()
                hetero_flag, res_number, insert_code = residue.get_id()
                if hetero_flag != ' ':
                    continue   # residue from glucose or water
                symbol = VOCAB.abrv_to_symbol(abrv)
                if symbol is None:
                    has_non_residue = True
                    break
                # filter Hs because not all data include them
                atoms = { atom.get_id(): list(atom.get_coord()) for atom in residue if atom.element != 'H' }
                residues.append(Residue(
                    symbol, atoms, (res_number, insert_code)
                ))
            if has_non_residue or len(residues) == 0:  # not a peptide
                continue
            peptides[_id] = Peptide(_id, residues)
        return cls(pdb_id, peptides)

    def get_id(self) -> str:
        return self.pdb_id

    def get_chain_names(self) -> List[str]:
        return list(self.peptides.keys())

    def get_chain(self, chain_name: str) -> Peptide:
        return self.peptides.get(chain_name)

    def num_chains(self) -> int:
        return len(self.peptides)


# =============================================================================
# AgAbComplex class
# =============================================================================

class AgAbComplex:
    """Represents an antibody-antigen complex."""

    num_interface_residues = 48

    def __init__(self, antigen: Protein, antibody: Protein, heavy_chain: str, light_chain: str,
                 numbering: str = 'imgt', skip_epitope_cal=False, skip_validity_check=False) -> None:
        self.heavy_chain = heavy_chain
        self.light_chain = light_chain
        self.numbering = numbering

        self.antigen = antigen
        if skip_validity_check:
            self.antibody, self.cdr_pos = antibody, None
        else:
            self.antibody, self.cdr_pos = self._extract_antibody_info(antibody, numbering)
        self.pdb_id = antigen.get_id()

        if skip_epitope_cal:
            self.epitope = None
        else:
            self.epitope = self._cal_epitope()

    @classmethod
    def from_pdb(cls, pdb_path: str, heavy_chain: str, light_chain: str, antigen_chains: List[str],
                 numbering: str = 'imgt', skip_epitope_cal=False, skip_validity_check=False):
        """Create AgAbComplex from PDB file."""
        protein = Protein.from_pdb(pdb_path)
        pdb_id = protein.get_id()

        # Get available chains in the protein
        available_chains = set(protein.get_chain_names())

        ab_peptides = {}
        if heavy_chain in available_chains:
            ab_peptides[heavy_chain] = protein.get_chain(heavy_chain)
        if light_chain in available_chains:
            ab_peptides[light_chain] = protein.get_chain(light_chain)

        # If no antigen chains specified, auto-detect
        if not antigen_chains:
            # Use all chains not used for antibody as antigen
            ab_chain_set = {heavy_chain, light_chain}
            antigen_chains = [c for c in available_chains if c not in ab_chain_set]

        ag_peptides = {}
        for chain in antigen_chains:
            if chain in available_chains and protein.get_chain(chain) is not None:
                ag_peptides[chain] = protein.get_chain(chain)

        # If no antigen was found, try to use any available non-antibody chain
        if not ag_peptides:
            ab_chain_set = {heavy_chain, light_chain}
            for c in available_chains:
                if c not in ab_chain_set:
                    chain_obj = protein.get_chain(c)
                    if chain_obj is not None:
                        ag_peptides[c] = chain_obj
                        break

        antigen = Protein(pdb_id, ag_peptides)
        antibody = Protein(pdb_id, ab_peptides)

        return cls(antigen, antibody, heavy_chain, light_chain, numbering, skip_epitope_cal, skip_validity_check)

    def _extract_antibody_info(self, antibody: Protein, numbering: str):
        numbering = numbering.lower()
        if numbering == 'imgt':
            _scheme = IMGT
        elif numbering == 'chothia':
            _scheme = Chothia
        else:
            raise NotImplementedError(f'Numbering scheme {numbering} not implemented')

        # Get CDR positions
        cdr_pos = {}

        # Heavy chain CDRs
        heavy = antibody.get_chain(self.heavy_chain)
        if heavy:
            for cdr_name, (start, end) in [('H1', _scheme.H1), ('H2', _scheme.H2), ('H3', _scheme.H3)]:
                for i in range(len(heavy)):
                    res = heavy.get_residue(i)
                    if res.get_id()[0] == start:
                        cdr_pos[cdr_name] = [i, i + (end - start)]
                        break

        # Light chain CDRs
        light = antibody.get_chain(self.light_chain)
        if light:
            for cdr_name, (start, end) in [('L1', _scheme.L1), ('L2', _scheme.L2), ('L3', _scheme.L3)]:
                for i in range(len(light)):
                    res = light.get_residue(i)
                    if res.get_id()[0] == start:
                        cdr_pos[cdr_name] = [i, i + (end - start)]
                        break

        return antibody, cdr_pos

    def _cal_epitope(self):
        """Calculate epitope residues."""
        epitope = []

        # Get antibody chains
        ab_chains = []
        if self.antibody:
            hc = self.antibody.get_chain(self.heavy_chain)
            lc = self.antibody.get_chain(self.light_chain)
            if hc:
                ab_chains.append(hc)
            if lc:
                ab_chains.append(lc)

        # Get antigen chains
        ag_chains = []
        for chain_name in self.antigen.get_chain_names():
            ag_chains.append(self.antigen.get_chain(chain_name))

        # Find interacting residues
        for ab_chain in ab_chains:
            for i in range(len(ab_chain)):
                ab_res = ab_chain.get_residue(i)
                ab_coords = ab_res.get_backbone_coord_map()

                for ag_chain in ag_chains:
                    for j in range(len(ag_chain)):
                        ag_res = ag_chain.get_residue(j)
                        ag_coords = ag_res.get_backbone_coord_map()

                        # Check distance
                        for ab_atom in ab_coords.values():
                            for ag_atom in ag_coords.values():
                                dist = np.linalg.norm(np.array(ab_atom) - np.array(ag_atom))
                                if dist < CONTACT_DIST:
                                    epitope.append((ag_res, ag_chain.id, j))
                                    break
                        else:
                            continue
                        break

        return list(set(epitope))

    def get_heavy_chain(self) -> Peptide:
        if self.antibody:
            return self.antibody.get_chain(self.heavy_chain)
        return None

    def get_light_chain(self) -> Peptide:
        if self.antibody:
            return self.antibody.get_chain(self.light_chain)
        return None

    def get_antigen(self) -> Protein:
        from copy import deepcopy
        return deepcopy(self.antigen)

    def get_epitope(self, cdrh3_pos=None):
        from copy import deepcopy
        if cdrh3_pos is not None:
            backup = self.cdr_pos
            self.cdr_pos = {'CDR-H3': [cdrh3_pos[0], cdrh3_pos[1]]}
            epitope = self._cal_epitope()
            self.cdr_pos = backup
            return deepcopy(epitope)
        if self.epitope is None:
            self.epitope = self._cal_epitope()
        return deepcopy(self.epitope)

    def get_cdr_pos(self, cdr: str) -> List[int]:
        if self.cdr_pos and cdr in self.cdr_pos:
            return self.cdr_pos[cdr]
        # Default positions if not found
        defaults = {
            'H1': [27, 38], 'H2': [56, 65], 'H3': [105, 117],
            'L1': [27, 38], 'L2': [56, 65], 'L3': [105, 117]
        }
        return defaults.get(cdr, [0, 0])
