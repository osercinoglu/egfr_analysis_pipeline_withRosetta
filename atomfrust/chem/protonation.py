"""Protonation and tautomer enumeration at pH 7.4 (plan G2; methods R19, R20, S1.4).

**A different protonation state is a different molecule.** It has a different formal charge,
a different hydrogen-bond pattern and a different REF2015 energy, so it must reach
parametrisation as a distinct identity. Two mechanisms enforce that and neither may be
bypassed:

* :data:`PROTONATION_VERSION` is the middle field of
  :meth:`atomfrust.chem.cache.ParamCache.key` (``inchikey|protonation_version|conformer_version``).
  Bumping it when this module's enumeration changes is a cache miss for every molecule —
  which is the point, because params built under a different protonation model would
  otherwise be reused silently and a run's numbers would not follow from its manifest.
  Build caches through :func:`param_cache` so the stamp cannot be forgotten.
* The chosen state's :attr:`Protomer.state_id` belongs in the ``input_digests`` mapping of
  :func:`atomfrust.settings.regeneration_key`, so two states of one ligand produce two
  regeneration keys and never share a decoy ensemble.

**S1.4 asks for an uncertainty band, not a winner.** The deliverable is *how much the answer
moves* across the plausible states, so :func:`sensitivity_table` is the primary output and
:func:`canonical_state` is the explicitly reduced case that exists only so a point-estimate
run is possible. Ranking is by ``log10`` population penalty relative to the dominant
microspecies: for one acidic site, ``log10([A-]/[HA]) = pH - pKa``, so choosing the minor
form of a site costs ``|pH - pKa|`` in log units and the penalties of independent sites add.
Rank 0 is therefore the dominant state and later ranks are progressively less populated —
the ordering means something rather than being enumeration order.

**Which backend ran is recorded, never inferred.** :attr:`ProtomerSet.method` is one of:

``"dimorphite"``
    Dimorphite-DL enumerated the ionisation states over ``ph ± ph_window``. **This package
    is not installed in this environment, so this path has never executed here.** It is
    tried first when importable and any failure falls back to the path below.
``"rdkit_tautomer"``
    The fallback: RDKit tautomer enumeration plus the small explicit rule set in
    :data:`IONISATION_RULES`. The rules cover four unambiguous groups and *nothing else* —
    imidazoles, anilines, phenols, thiols, tetrazoles, amidines and guanidines are left as
    drawn. Treat a charge of 0 from this path as "no rule fired", not as "predicted neutral".
    Sites are also treated as independent, which over-ionises polybasic ligands: imatinib's
    N-methylpiperazine comes out ``+2`` where the real molecule is ``+1``, because
    protonating one ring nitrogen suppresses the other's pKa and a per-site rule cannot see
    that. The true ``+1`` state is still in the band — it is the rank-1 single-site
    deviation — which is precisely the argument for running the band instead of rank 0.
``"passthrough"``
    Enumeration ran and yielded exactly one state: no rule fired and no distinct tautomer
    exists. The molecule is used as drawn.

A consumer that needs a real pKa model must check ``method`` and refuse; it can never mistake
the fallback for Dimorphite-DL.

**States are deduplicated by InChIKey**, because the InChIKey *is* the cache key: two states
that share one would collide in :class:`~atomfrust.chem.cache.ParamCache` and the second
would silently reuse the first's params. Standard InChI normalises some mobile-hydrogen
tautomers, so this merges exactly the states the cache cannot tell apart — the merge is
forced by the cache design, not a convenience.

RDKit is required here (parsing, InChIKey, tautomers) and its absence raises rather than
being categorised per molecule, unlike :mod:`atomfrust.chem.paramize`: a protomer without an
InChIKey cannot key a cache, so there is no useful degraded result to return.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:  # pragma: no cover
    from atomfrust.chem.cache import ParamCache

__all__ = [
    "PROTONATION_VERSION",
    "IONISATION_RULES",
    "Protomer",
    "ProtomerSet",
    "enumerate_states",
    "canonical_state",
    "sensitivity_table",
    "param_cache",
    "dimorphite_available",
    "rdkit_available",
]

#: Bump when the enumeration changes — a new rule, a different pKa, a different backend.
#: This string is the ``protonation_version`` field of every :class:`ParamCache` key, so a
#: bump invalidates every cached parametrisation on purpose.
PROTONATION_VERSION = "v1"

DEFAULT_PH = 7.4

try:  # pragma: no cover - a property of the environment, not of the tests
    from rdkit import Chem, RDLogger
    from rdkit.Chem.MolStandardize import rdMolStandardize

    RDLogger.DisableLog("rdApp.*")
    RDKIT_IMPORT_ERROR: str | None = None
except Exception as exc:  # pragma: no cover
    Chem = rdMolStandardize = None  # type: ignore[assignment]
    RDKIT_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


def rdkit_available() -> bool:
    return RDKIT_IMPORT_ERROR is None


def dimorphite_available() -> bool:
    """Whether Dimorphite-DL is importable. Never installs anything; absence is normal."""
    import importlib.util

    return importlib.util.find_spec("dimorphite_dl") is not None


# --------------------------------------------------------------------------- rules


@dataclass(frozen=True)
class _Rule:
    """One ionisable group: where the proton moves and at what pKa.

    ``site_index`` selects the reacting atom out of the SMARTS match, so a match reports the
    hydroxyl of an acid rather than its carbon. ``delta`` is the formal-charge change:
    ``-1`` deprotonation (acid), ``+1`` protonation (base). ``pka`` is a single
    representative value for the whole group — deliberately crude, and the reason this path
    is labelled a fallback.
    """

    name: str
    smarts: str
    site_index: int
    delta: int
    pka: float


#: The fallback rule set: four groups whose state at pH 7.4 is not in doubt, and nothing
#: else. Each pKa is a class representative, not a prediction for the specific molecule:
#: acetic acid 4.76 and a benzoic acid 4.2 are both scored at 4.2, which is fine two pH units
#: from the pH and wrong near it. Groups with substituent-dependent pKa in the 5-9 window
#: (imidazole, aniline, phenol, thiol, tetrazole, amidine, guanidine) are deliberately
#: absent: guessing them wrong is worse than leaving the molecule as drawn, and a consumer
#: can see from ``method == "rdkit_tautomer"`` that no real pKa model ran.
IONISATION_RULES: tuple[_Rule, ...] = (
    # -COOH -> -COO(-). pKa 4.2 (aliphatic/aromatic carboxylic acids, 3.8-4.9).
    _Rule("carboxylic_acid", "[CX3](=[OX1])[OX2H1]", 2, -1, 4.2),
    # -SO3H -> -SO3(-). pKa ~ -1; fully ionised anywhere near neutrality.
    _Rule("sulfonic_acid", "[SX4](=[OX1])(=[OX1])[OX2H1]", 3, -1, -1.0),
    # P(=O)-OH -> P(=O)-O(-), once per acidic OH. Scored at the *second* pKa (6.7) for every
    # OH: the first (~2) is far from any pH of interest, and using the tighter value keeps a
    # phosphate near pH 7 in the uncertainty band instead of asserting full ionisation.
    _Rule("phosphate_hydroxyl", "[PX4](=[OX1])[OX2H1]", 2, -1, 6.7),
    # Aliphatic amine -> ammonium. pKa 10.0. Excluded: already-charged N, imines/nitro
    # (N=*), anilines (N on an aromatic ring), amides and thioamides (N-C=O/S/N),
    # sulfonamides and phosphoramides (N-S/P), hydrazines/hydroxylamines (N-N/O/S),
    # cyanamides (N-C#N) and aromatic N (n).
    _Rule(
        "aliphatic_amine",
        "[NX3;H0,H1,H2;!$([N+]);!$([NX3]=*);!$([N][a]);!$([N]C=[O,S,N]);"
        "!$([N][S,P]);!$([N][N,O,S]);!$([N]C#N)]",
        0,
        +1,
        10.0,
    ),
)


@dataclass(frozen=True)
class _Candidate:
    """One state before ranking. ``axis`` separates the two kinds of deviation.

    ``axis=0`` is an ionisation deviation, whose cost ``penalty`` is a real log10 population
    ratio; ``axis=1`` is a tautomer of the canonical ionisation state, whose cost is only
    RDKit's ``ScoreTautomer`` ordering. The two are **not** on a common scale, so they are
    never interleaved: every ionisation state precedes every tautomer variant. That is the
    right call on average — a minor tautomer is usually rarer than a two-to-three log-unit
    ionisation shift — but it is wrong for a near-degenerate pair such as
    2-pyridone/2-hydroxypyridine. Read a returned band as a *set* of states to run; only the
    ``axis=0`` ranks carry a quantitative claim about relative population.
    """

    axis: int
    penalty: float | None
    tautomer_rank: int
    smiles: str
    mol: Any
    origin: str

    @property
    def sort_key(self) -> tuple[int, float, int, str]:
        """Tautomer candidates have no penalty and rank by ``tautomer_rank`` alone."""
        return (self.axis, self.penalty or 0.0, self.tautomer_rank, self.smiles)


@dataclass(frozen=True)
class _Site:
    """A matched ionisable atom in one molecule."""

    rule: str
    atom_index: int
    delta: int
    pka: float

    def fires_at(self, ph: float) -> bool:
        """Acids ionise above their pKa, bases below theirs."""
        return ph >= self.pka if self.delta < 0 else ph <= self.pka

    def penalty(self, ph: float) -> float:
        """``|pH - pKa|``: the log10 population cost of taking this site's minor form."""
        return abs(ph - self.pka)


# --------------------------------------------------------------------------- records


@dataclass(frozen=True)
class Protomer:
    """One microspecies: a protonation state, possibly a tautomer, of the input molecule.

    ``inchikey`` is the identity that reaches the cache, so it is what makes this state a
    different molecule from its siblings. ``is_canonical`` marks the single state a
    point-estimate run would use — exactly one per :class:`ProtomerSet`, always ``rank`` 0.
    ``log10_penalty`` is the population cost relative to that state (0.0 for the canonical
    one, larger = rarer); it is ``None`` for tautomer-only variants, because RDKit's tautomer
    score is an ordering, not a free energy. ``None`` rather than ``nan``: a ``nan`` field
    would make a protomer unequal to itself and break the determinism guarantee that
    :func:`enumerate_states` returns the same states for the same input.
    """

    smiles: str
    inchikey: str
    formal_charge: int
    is_canonical: bool
    rank: int
    log10_penalty: float | None = 0.0
    origin: str = ""

    @property
    def state_id(self) -> str:
        """``"{inchikey}|{PROTONATION_VERSION}"`` — the value to put in the ``input_digests``
        of :func:`atomfrust.settings.regeneration_key`, so that regenerating a run under a
        different state or a different enumeration cannot reuse the stored ensemble."""
        return f"{self.inchikey}|{PROTONATION_VERSION}"


@dataclass(frozen=True)
class ProtomerSet:
    """The states of one molecule at one pH, ordered most-probable-first.

    ``method`` records which backend produced this set (see the module docstring); it is part
    of the result rather than a log line because the fallback and Dimorphite-DL are not
    interchangeable evidence.
    """

    input_smiles: str
    ph: float
    states: tuple[Protomer, ...]
    method: str

    @property
    def canonical(self) -> Protomer:
        return self.states[0]

    @property
    def charges(self) -> tuple[int, ...]:
        return tuple(s.formal_charge for s in self.states)

    @property
    def charge_spread(self) -> int:
        """``max - min`` formal charge. 0 means the band has no charge uncertainty — it may
        still have tautomer uncertainty."""
        charges = self.charges
        return max(charges) - min(charges)

    def __len__(self) -> int:
        return len(self.states)


# --------------------------------------------------------------------------- API


def enumerate_states(
    smiles: str | Any,
    ph: float = DEFAULT_PH,
    max_states: int = 8,
    tautomers: bool = True,
    *,
    ph_window: float = 1.0,
    max_tautomers: int = 64,
) -> ProtomerSet:
    """Enumerate the plausible protonation/tautomer states of one molecule at ``ph``.

    Deterministic: the same SMILES and arguments give the same states in the same order, with
    no RNG anywhere in the path. Ordering is by ``(axis, log10 population penalty, tautomer
    rank, canonical SMILES)`` — ionisation states first and tautomer variants after, for the
    reason given in :class:`_Candidate`, with the SMILES only breaking ties so the result
    cannot depend on RDKit's internal match order.

    ``max_states`` truncates the tail, never the head, so the canonical state survives any
    limit; ``max_states <= 0`` means no limit. ``tautomers=False`` restricts the band to
    ionisation, which is what a caller who has already fixed a tautomer wants.

    ``ph_window`` is only used by the Dimorphite-DL path, where it becomes the ``min_ph``/
    ``max_ph`` bracket: the states that matter are those whose populations swap over a
    plausible pH range, not those at exactly one pH.

    Raises ``ValueError`` for input RDKit cannot parse and ``RuntimeError`` if RDKit is
    absent. Neither is a property of the molecule's chemistry, which is why they raise here
    while :func:`atomfrust.chem.paramize.paramize` categorises instead.
    """
    if RDKIT_IMPORT_ERROR is not None:  # pragma: no cover - environment-dependent
        raise RuntimeError(f"protonation enumeration needs RDKit: {RDKIT_IMPORT_ERROR}")

    mol = _parse(smiles)
    input_smiles = Chem.MolToSmiles(mol)

    candidates: list[_Candidate] | None = None
    method = "rdkit_tautomer"
    if dimorphite_available():
        candidates = _dimorphite_candidates(mol, ph, ph_window)
        if candidates is not None:
            method = "dimorphite"
    if candidates is None:
        candidates = _rule_candidates(mol, ph)

    if tautomers:
        candidates = _expand_tautomers(candidates, max_tautomers)

    states = _rank(candidates, max_states)
    if len(states) == 1 and method == "rdkit_tautomer":
        # No rule fired and no distinct tautomer: the molecule is used as drawn, and saying
        # "rdkit_tautomer" would overstate what happened.
        method = "passthrough"
    return ProtomerSet(input_smiles=input_smiles, ph=ph, states=states, method=method)


def canonical_state(smiles: str | Any, ph: float = DEFAULT_PH, **kwargs: Any) -> Protomer:
    """The single most populated state — the reduced, point-estimate case of S1.4.

    Provided so a run *can* be a point estimate, not because a point estimate is the
    deliverable. Anything reported from this alone carries no uncertainty band; use
    :func:`sensitivity_table` to say how much the band would have moved it.
    """
    return enumerate_states(smiles, ph=ph, **kwargs).canonical


SENSITIVITY_COLUMNS = (
    "molecule",
    "input_smiles",
    "ph",
    "method",
    "n_states",
    "rank",
    "smiles",
    "inchikey",
    "formal_charge",
    "is_canonical",
    "log10_penalty",
    "origin",
    "state_id",
    "charge_min",
    "charge_max",
    "charge_spread",
)


def sensitivity_table(
    smiles: Iterable[str] | Mapping[str, str],
    ph: float = DEFAULT_PH,
    *,
    max_states: int = 8,
    tautomers: bool = True,
) -> pd.DataFrame:
    """One row per ``(molecule, state)`` — the S1.4 evidence table.

    ``smiles`` is an iterable of SMILES, or a mapping ``{molecule_id: smiles}`` when the
    molecules have names worth keeping (a CCD code, a library id); an unnamed iterable is
    labelled by position so rows stay joinable either way.

    The per-molecule columns (``n_states``, ``charge_min``, ``charge_max``,
    ``charge_spread``) are repeated on every row of that molecule so the table needs no
    second frame to be read: ``df.groupby("molecule")["charge_spread"].first()`` is the
    charge band, ``df[df.is_canonical]`` is the point-estimate run, and the difference
    between the two is what S1.4 asks to be carried downstream.

    Molecules are enumerated in the order given, and each molecule's rows are in rank order.
    """
    items: list[tuple[str, str]]
    if isinstance(smiles, Mapping):
        items = [(str(k), v) for k, v in smiles.items()]
    elif isinstance(smiles, str):
        raise TypeError("sensitivity_table takes many molecules; pass [smiles], not smiles")
    else:
        items = [(str(i), s) for i, s in enumerate(smiles)]

    rows: list[dict[str, Any]] = []
    for name, text in items:
        result = enumerate_states(text, ph=ph, max_states=max_states, tautomers=tautomers)
        charges = result.charges
        for state in result.states:
            rows.append(
                {
                    "molecule": name,
                    "input_smiles": result.input_smiles,
                    "ph": result.ph,
                    "method": result.method,
                    "n_states": len(result),
                    "rank": state.rank,
                    "smiles": state.smiles,
                    "inchikey": state.inchikey,
                    "formal_charge": state.formal_charge,
                    "is_canonical": state.is_canonical,
                    "log10_penalty": state.log10_penalty,
                    "origin": state.origin,
                    "state_id": state.state_id,
                    "charge_min": min(charges),
                    "charge_max": max(charges),
                    "charge_spread": result.charge_spread,
                }
            )
    if not rows:
        return pd.DataFrame(columns=list(SENSITIVITY_COLUMNS))
    return pd.DataFrame(rows)[list(SENSITIVITY_COLUMNS)]


def param_cache(root: Path | str, conformer_version: str = "v1") -> ParamCache:
    """A :class:`~atomfrust.chem.cache.ParamCache` stamped with :data:`PROTONATION_VERSION`.

    The one constructor a caller in this pipeline should use. ``ParamCache`` defaults its
    ``protonation_version`` to ``"v1"``, so a hand-built cache keeps hitting after this
    module bumps the version — exactly the silent reuse the key exists to prevent.
    """
    from atomfrust.chem.cache import ParamCache  # noqa: PLC0415 - avoids an import cycle

    return ParamCache(
        root, protonation_version=PROTONATION_VERSION, conformer_version=conformer_version
    )


# --------------------------------------------------------------------------- internals


def _parse(smiles: str | Any) -> Any:
    if isinstance(smiles, str):
        text = smiles.strip()
        if not text:
            raise ValueError("empty SMILES")
        mol = Chem.MolFromSmiles(text)
        if mol is None:
            raise ValueError(f"RDKit could not parse SMILES {text!r}")
        return mol
    if smiles is None:
        raise ValueError("molecule is None")
    mol = Chem.Mol(smiles)
    Chem.SanitizeMol(mol)
    return mol


def _sites(mol: Any) -> list[_Site]:
    """Every ionisable atom under :data:`IONISATION_RULES`, in a stable order.

    Sorted by atom index so the enumeration does not inherit RDKit's substructure-match
    order; deduplicated because two rules could in principle claim one atom.
    """
    found: dict[int, _Site] = {}
    for rule in IONISATION_RULES:
        pattern = Chem.MolFromSmarts(rule.smarts)
        if pattern is None:  # pragma: no cover - a typo in a rule, caught by the tests
            raise ValueError(f"rule {rule.name} has invalid SMARTS {rule.smarts!r}")
        for match in mol.GetSubstructMatches(pattern):
            index = match[rule.site_index]
            found.setdefault(index, _Site(rule.name, index, rule.delta, rule.pka))
    return [found[i] for i in sorted(found)]


def _apply(mol: Any, sites: Sequence[_Site]) -> Any:
    """Move the protons of ``sites``. Atom indices are stable — no atom is added or removed,
    only its explicit hydrogen count and formal charge change."""
    editable = Chem.RWMol(mol)
    for site in sites:
        atom = editable.GetAtomWithIdx(site.atom_index)
        hydrogens = atom.GetTotalNumHs()
        atom.SetFormalCharge(atom.GetFormalCharge() + site.delta)
        atom.SetNumExplicitHs(max(0, hydrogens + site.delta))
        # Without this the implicit-H valence model would put the proton straight back.
        atom.SetNoImplicit(True)
    out = editable.GetMol()
    Chem.SanitizeMol(out)
    return out


def _rule_candidates(mol: Any, ph: float) -> list[_Candidate]:
    """The fallback band: the dominant microspecies, then each single-site deviation.

    Deviations are single-site only. The full 2^n microspecies lattice is exponential and
    its multi-site members are the product of two already-unlikely choices, so they sit far
    below the states that matter; the all-neutral form is included as well because it is the
    state a naive pipeline would have used and the band should show what that costs.
    """
    sites = _sites(mol)
    firing = [s for s in sites if s.fires_at(ph)]

    variants: list[tuple[float, Sequence[_Site], str]] = [
        (0.0, firing, "dominant" if firing else "as_drawn")
    ]
    for site in sites:
        deviated = [s for s in firing if s.atom_index != site.atom_index]
        if not site.fires_at(ph):
            deviated = [*deviated, site]
        variants.append((site.penalty(ph), deviated, f"minor:{site.rule}@{site.atom_index}"))
    if len(sites) > 1:
        variants.append(
            (sum(s.penalty(ph) for s in firing), [], "neutral")
        )

    out: list[_Candidate] = []
    for penalty, chosen, origin in variants:
        state = _apply(mol, chosen)
        out.append(_Candidate(0, penalty, 0, Chem.MolToSmiles(state), state, origin))
    return out


def _dimorphite_candidates(mol: Any, ph: float, ph_window: float) -> list[_Candidate] | None:
    """Dimorphite-DL over ``ph ± ph_window``; ``None`` if it is unusable, so the caller falls
    back rather than crashing.

    Untested in this environment — ``dimorphite_dl`` is not installed here, so every call
    below is guarded and any deviation from the expected API is a fallback, not an error.
    The canonical state is the one Dimorphite-DL reports for the exact pH; the rest are
    ordered by absolute formal charge distance from it, then by SMILES, since Dimorphite-DL
    returns a set with no ranking.
    """
    try:  # pragma: no cover - never executes without dimorphite_dl installed
        from dimorphite_dl import DimorphiteDL

        smiles = Chem.MolToSmiles(mol)
        band = DimorphiteDL(
            min_ph=ph - ph_window, max_ph=ph + ph_window, pka_precision=1.0,
            max_variants=16, label_states=False, silent=True,
        ).protonate(smiles)
        point = DimorphiteDL(
            min_ph=ph, max_ph=ph, pka_precision=0.0,
            max_variants=1, label_states=False, silent=True,
        ).protonate(smiles)
    except Exception:  # pragma: no cover
        return None

    if not point:  # pragma: no cover
        return None
    canonical = Chem.MolFromSmiles(point[0])
    if canonical is None:  # pragma: no cover
        return None
    reference = Chem.GetFormalCharge(canonical)

    out = [_Candidate(0, 0.0, 0, Chem.MolToSmiles(canonical), canonical, "dimorphite:ph")]
    for text in band:  # pragma: no cover
        state = Chem.MolFromSmiles(text)
        if state is None:
            continue
        text = Chem.MolToSmiles(state)
        if text == out[0].smiles:
            continue
        distance = float(abs(Chem.GetFormalCharge(state) - reference))
        out.append(_Candidate(0, max(distance, 1e-6), 0, text, state, "dimorphite:band"))
    return out


def _expand_tautomers(candidates: list[_Candidate], max_tautomers: int) -> list[_Candidate]:
    """Add the tautomers of the *canonical* ionisation state only.

    Deliberately not every ionisation variant: enumeration is combinatorial in both axes and
    the tautomers of an already-minor microspecies are minor twice over. Tautomers are
    ordered by RDKit's ``ScoreTautomer`` (descending), which prefers aromatic rings and
    carbonyls over enols. They land on ``axis=1`` and carry ``nan`` penalty because that
    score is an ordering, not a population — see :class:`_Candidate`.

    ``max_tautomers`` bounds the enumeration: it is exponential in the number of mobile
    hydrogens, and a nucleotide-like ligand will otherwise produce thousands of states that
    ``max_states`` immediately throws away.
    """
    parameters = rdMolStandardize.CleanupParameters()
    parameters.maxTautomers = max_tautomers
    enumerator = rdMolStandardize.TautomerEnumerator(parameters)

    parent = candidates[0]
    try:
        found = list(enumerator.Enumerate(parent.mol))
    except Exception:  # pragma: no cover - RDKit gives up on pathological inputs
        return candidates
    scored = sorted(
        ((enumerator.ScoreTautomer(t), Chem.MolToSmiles(t), t) for t in found),
        key=lambda row: (-row[0], row[1]),
    )
    extra = [
        _Candidate(1, None, rank, text, state, f"{parent.origin}+tautomer")
        for rank, (_, text, state) in enumerate(scored)
        if text != parent.smiles
    ]
    return [*candidates, *extra]


def _rank(candidates: list[_Candidate], max_states: int) -> tuple[Protomer, ...]:
    """Sort, deduplicate by InChIKey, truncate. Rank 0 is the canonical state.

    Truncation is of the tail, so ``max_states`` can never drop the canonical state; the
    ``rank`` field is assigned *after* truncation and is therefore always dense.
    """
    seen: set[str] = set()
    states: list[Protomer] = []
    for candidate in sorted(candidates, key=lambda c: c.sort_key):
        try:
            inchikey = Chem.MolToInchiKey(candidate.mol) or ""
        except Exception:  # pragma: no cover
            inchikey = ""
        if not inchikey or inchikey in seen:
            continue
        seen.add(inchikey)
        states.append(
            Protomer(
                smiles=candidate.smiles,
                inchikey=inchikey,
                formal_charge=Chem.GetFormalCharge(candidate.mol),
                is_canonical=not states,
                rank=len(states),
                # Three decimals is already more precision than a class-representative pKa
                # supports; it exists to keep the table readable, not to be believed.
                log10_penalty=(
                    round(candidate.penalty, 3) if candidate.penalty is not None else None
                ),
                origin=candidate.origin,
            )
        )
        if max_states > 0 and len(states) >= max_states:
            break
    if not states:  # pragma: no cover - only if InChI generation fails for every state
        raise ValueError("no state produced a valid InChIKey")
    return tuple(states)
