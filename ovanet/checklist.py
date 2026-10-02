"""
ovanet/checklist.py

Rule-based Rotterdam/2023-guideline checklist. NOT fed into the ML model (C5 contract).
Purely transparent, clinician-facing logic -- shown side-by-side with the AI risk score.
"""

from ovanet.contracts import ClinicalInput


def evaluate_checklist(clinical: ClinicalInput) -> dict:
    """Returns a transparent, rule-based Rotterdam/2023-aligned summary. Not a diagnosis."""
    criteria_met = []
    criteria_unknown = []

    if clinical.cycle_irregular is True:
        criteria_met.append("Irregular cycles (oligo/anovulation)")
    elif clinical.cycle_irregular is None:
        criteria_unknown.append("Cycle regularity")

    if clinical.hyperandrogenism is True:
        criteria_met.append("Clinical/biochemical hyperandrogenism")
    elif clinical.hyperandrogenism is None:
        criteria_unknown.append("Hyperandrogenism signs")

    if clinical.lh_fsh is not None and clinical.lh_fsh > 2.0:
        criteria_met.append(f"Elevated LH/FSH ratio ({clinical.lh_fsh:.1f})")

    if clinical.amh is not None and clinical.amh > 4.5:
        criteria_met.append(f"Elevated AMH ({clinical.amh:.1f} ng/mL) -- 2023 guideline alternative to ultrasound")

    n_met = len(criteria_met)
    n_unknown = len(criteria_unknown)

    if n_met >= 2:
        summary = "Meets Rotterdam criteria pattern (>=2 of: irregular cycles, hyperandrogenism, PCOM on imaging)"
    elif n_met == 1:
        summary = "Partially meets Rotterdam criteria -- 1 of 2+ required features present"
    else:
        summary = "Insufficient clinical criteria entered to assess Rotterdam pattern"

    note = None
    if clinical.years_since_menarche is not None and clinical.years_since_menarche < 2:
        note = ("2023 guideline caution: ultrasound-based diagnosis is advised against "
                "within 2 years of menarche due to physiologic multi-follicular ovaries.")

    return {
        "criteria_met": criteria_met,
        "criteria_unknown": criteria_unknown,
        "n_met": n_met,
        "summary": summary,
        "adolescent_caution": note,
        "disclaimer": "This checklist is rule-based and transparent. It is NOT used by the AI "
                       "risk model and does NOT constitute a diagnosis.",
    }