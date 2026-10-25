"""Official CASMI metric identity + MRR@25 scoring (verbatim logic from the V465 notebook,
which embeds the official metric implementation; both answer key and submission pass through
the same canonicalization)."""
from functools import lru_cache
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize
RDLogger.DisableLog('rdApp.*')
_tautomer_canonicalizer = rdMolStandardize.TautomerEnumerator()

def canonicalize_smiles_to_ik14(smiles: str):
    if not isinstance(smiles, str):
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        Chem.RemoveStereochemistry(mol)
        canonical_mol = _tautomer_canonicalizer.Canonicalize(mol)
        inchikey = Chem.MolToInchiKey(canonical_mol)
        return inchikey[:14]
    except (ValueError, RuntimeError, TypeError):
        return None

def heavy_atom_composition(smiles):
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None:
        return None
    counts = {}
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        if sym != 'H':
            counts[sym] = counts.get(sym, 0) + 1
    return tuple(sorted(counts.items()))

@lru_cache(maxsize=300000)
def metric_identity(smiles):
    key = canonicalize_smiles_to_ik14(smiles)
    return (key, heavy_atom_composition(smiles)) if key else None

def first_ranks(smiles):
    ranks = {}
    for rank, smi in enumerate(smiles, 1):
        identity = metric_identity(smi)
        if identity is not None:
            ranks.setdefault(identity, rank)
    return ranks

def mrr_topk(sub_lists, label_smiles, k=25):
    """sub_lists: {molecule_id: [smiles...up to 25]}; label_smiles: {molecule_id: true smiles}.
    Returns (mrr, per_mid dict rank-or-None)."""
    total, per = 0.0, {}
    for mid, true_smi in label_smiles.items():
        target = metric_identity(true_smi)
        if target is None:
            per[mid] = None
            continue
        ranks = first_ranks(sub_lists.get(mid, [])[:k])
        r = ranks.get(target)
        per[mid] = r
        if r:
            total += 1.0 / r
    n = len(label_smiles)
    return total / n if n else 0.0, per

def topk_mrr_from_sub_csv(sub_df, label_dict, k=25):
    lists = {str(r.molecule_id): [s.strip() for s in str(r.smiles).split(';') if s.strip()]
             for r in sub_df.itertuples(index=False)}
    return mrr_topk(lists, {str(kk): v for kk, v in label_dict.items()}, k)
