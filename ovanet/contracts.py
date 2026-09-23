"""
ovanet/contracts.py

Canonical data shapes for every module boundary in the OvaNet pipeline (C1-C8).
Plain dataclasses, no pydantic, no HTTP layer. Every function in ovanet.vision,
ovanet.radiomics, ovanet.graph, ovanet.fusion, etc. must return objects matching
these shapes exactly.

Frozen: see docs/CONTRACTS.md for the human-readable version and sign-off log.
Any change here must be logged in docs/DECISIONS.md and announced to both members.
"""

from dataclasses import dataclass
from typing import Optional, Literal
import numpy as np

# ---------------------------------------------------------------------------
# Fixed constants (frozen Day 2 -- do not change without both members' sign-off)
# ---------------------------------------------------------------------------

IMAGE_SIZE = 256
E_CNN_DIM = 512
GRAPH_FEATURE_DIM = 5
GCN_EMBED_DIM = 16
RADIOMICS_FEATURE_ORDER = ["mean", "std", "entropy", "glcm_contrast", "glcm_homogeneity"]

GRAPH_STAT_NAMES = [
    "n_follicles",
    "mean_nn_dist_rel",
    "std_nn_dist_rel",
    "mean_degree",
    "clustering_coeff",
    "peripheral_index",
    "size_cv",
    "follicle_area_fraction",
]

RISK_BANDS = ["LOW", "MODERATE", "HIGH"]
FEATURE_FAMILIES = ["radiomics", "follicle", "graph", "cnn", "gnn"]


# ---------------------------------------------------------------------------
# C1 -- preprocess
# ---------------------------------------------------------------------------

@dataclass
class QualityCheck:
    ok: bool
    reasons: list
    blur_var: float


@dataclass
class PreprocessResult:
    raw: np.ndarray
    enh: np.ndarray
    quality: QualityCheck


# ---------------------------------------------------------------------------
# C2 -- vision
# ---------------------------------------------------------------------------

@dataclass
class Follicle:
    id: int
    cx: float
    cy: float
    area_px: float
    eq_radius_px: float
    eccentricity: float
    solidity: float
    mean_intensity: float


@dataclass
class VisionResult:
    ovary_mask: np.ndarray
    follicle_labels: np.ndarray
    follicles: list
    e_cnn: np.ndarray
    seg_method: str  # "classical" | "unet"
    warnings: list


# ---------------------------------------------------------------------------
# C3 -- radiomics
# ---------------------------------------------------------------------------

@dataclass
class RadiomicsResult:
    mean: float
    std: float
    entropy: float
    glcm_contrast: float
    glcm_homogeneity: float

    def as_vector(self) -> np.ndarray:
        """Fixed order per RADIOMICS_FEATURE_ORDER."""
        return np.array(
            [self.mean, self.std, self.entropy, self.glcm_contrast, self.glcm_homogeneity],
            dtype=np.float32,
        )


# ---------------------------------------------------------------------------
# C4 -- graph
# ---------------------------------------------------------------------------

@dataclass
class GraphEdge:
    source: int
    target: int
    weight: float


@dataclass
class GraphNode:
    id: int
    x: float
    y: float


@dataclass
class GraphResult:
    n: int
    x: np.ndarray
    adj: np.ndarray
    edges: list
    nodes: list
    stats: dict


# ---------------------------------------------------------------------------
# C5 -- clinical (rule-based checklist input only; NOT fed to ML in MVP)
# ---------------------------------------------------------------------------

@dataclass
class ClinicalInput:
    age: Optional[float] = None
    bmi: Optional[float] = None
    cycle_irregular: Optional[bool] = None
    hyperandrogenism: Optional[bool] = None
    lh_fsh: Optional[float] = None
    amh: Optional[float] = None
    years_since_menarche: Optional[float] = None
    source: str = "user_entered"  # "user_entered" | "synthetic_demo"


# ---------------------------------------------------------------------------
# C6 -- fusion
# ---------------------------------------------------------------------------

@dataclass
class TopFeature:
    name: str
    family: str
    contribution: float


@dataclass
class BaselineVote:
    votes_pcos: int
    of: int
    label: str


@dataclass
class FusionResult:
    risk_score: float
    band: str  # "LOW" | "MODERATE" | "HIGH"
    family_attribution: dict
    top_features: list
    baseline_vote: BaselineVote
    model_version: str


# ---------------------------------------------------------------------------
# C7 -- longitudinal
# ---------------------------------------------------------------------------

@dataclass
class LongitudinalResult:
    trend: str  # "IMPROVING" | "STABLE" | "WORSENING" | "INSUFFICIENT_DATA"
    deltas: dict
    report_text: str
    is_demo_pair: bool


# ---------------------------------------------------------------------------
# C8 -- full pipeline output
# ---------------------------------------------------------------------------

@dataclass
class AnalysisResult:
    quality: QualityCheck
    vision: VisionResult
    radiomics: RadiomicsResult
    graph: GraphResult
    fusion: FusionResult
    timings: dict
    warnings: list
    is_mock: bool = False


# ---------------------------------------------------------------------------
# Validation helpers -- used by tests/test_contracts.py and to sanity-check
# any real module's output the moment it lands
# ---------------------------------------------------------------------------

def validate_preprocess(p: PreprocessResult) -> None:
    assert p.raw.shape == (IMAGE_SIZE, IMAGE_SIZE), f"raw shape {p.raw.shape}"
    assert p.enh.shape == (IMAGE_SIZE, IMAGE_SIZE), f"enh shape {p.enh.shape}"
    assert p.raw.dtype == np.float32, f"raw dtype {p.raw.dtype}"
    assert p.enh.dtype == np.float32, f"enh dtype {p.enh.dtype}"
    assert 0.0 <= float(p.raw.min()) and float(p.raw.max()) <= 1.0, "raw not in [0,1]"
    assert 0.0 <= float(p.enh.min()) and float(p.enh.max()) <= 1.0, "enh not in [0,1]"


def validate_vision(v: VisionResult) -> None:
    assert v.ovary_mask.shape == (IMAGE_SIZE, IMAGE_SIZE)
    assert v.follicle_labels.shape == (IMAGE_SIZE, IMAGE_SIZE)
    assert v.e_cnn.shape == (E_CNN_DIM,), f"e_cnn shape {v.e_cnn.shape}"
    assert v.seg_method in ("classical", "unet")


def validate_radiomics(r: RadiomicsResult) -> None:
    vec = r.as_vector()
    assert vec.shape == (5,)
    assert not np.isnan(vec).any(), "radiomics vector has NaN"


def validate_graph(g: GraphResult) -> None:
    assert g.x.shape == (g.n, GRAPH_FEATURE_DIM), f"x shape {g.x.shape} vs n={g.n}"
    assert g.adj.shape == (g.n, g.n), f"adj shape {g.adj.shape} vs n={g.n}"
    for name in GRAPH_STAT_NAMES:
        assert name in g.stats, f"missing graph stat: {name}"


def validate_fusion(f: FusionResult) -> None:
    assert 0.0 <= f.risk_score <= 100.0
    assert f.band in RISK_BANDS
    for fam in FEATURE_FAMILIES:
        assert fam in f.family_attribution, f"missing family attribution: {fam}"


def validate_analysis(a: AnalysisResult) -> None:
    validate_vision(a.vision)
    validate_radiomics(a.radiomics)
    validate_graph(a.graph)
    validate_fusion(a.fusion)