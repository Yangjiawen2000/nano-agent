"""
Virtual Patient Population for Brain-Targeted Nanoparticle PBPK
================================================================
Applies age- and disease-state-specific PBPK parameter scaling to simulate
inter-patient variability in NP brain delivery.

Patient label convention:  "<age_group>_<disease_state>"
  e.g. "adult_healthy", "elderly_glioma", "pediatric_neuroinflammation"

Age groups   : pediatric | adult | elderly
Disease states: healthy | glioma | neuroinflammation | alzheimer

Scientific references:
  Age:     Bhattacharya Adv Drug Deliv Rev 2022; Nájera Pharmaceutics 2025
  Glioma:  Wiley PNAS 2015 (TfR overexpression); Bhattacharya 2022 (EPR)
  Neuroinflamm: Khung ACS Nano 2023 (cytokine-driven BBB disruption)
  AD:      Mucke Science 2009; Zlokovic Nature Rev Neurosci 2011 (P-gp, AQP4)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

# ── Age-related scaling factors ────────────────────────────────────────────────

AGE_FACTORS: dict[str, dict] = {
    'pediatric': {
        'k_bind_scale':  1.30,   # TfR overexpression in developing brain
        'k_trans_scale': 1.20,   # Higher BBB permeability (tight junctions still forming)
        'k_endo_scale':  1.10,   # Enhanced endocytic activity
        'CL_scale':      0.85,   # Lower hepatic mass relative to adult (per kg)
        'description':   '0–12 y: TfR overexpression, developing BBB, reduced CL',
        'refs':          'Bhattacharya Adv Drug Deliv Rev 2022',
    },
    'adult': {
        'k_bind_scale':  1.00,
        'k_trans_scale': 1.00,
        'k_endo_scale':  1.00,
        'CL_scale':      1.00,
        'description':   '18–65 y: baseline (intact BBB, normal TfR expression)',
        'refs':          'baseline',
    },
    'elderly': {
        'k_bind_scale':  0.80,   # TfR downregulation with ageing
        'k_trans_scale': 1.10,   # Mildly leaky BBB (tight junction loosening)
        'k_endo_scale':  0.90,   # Reduced vesicular transport efficiency
        'CL_scale':      1.25,   # Reduced hepatic CYP3A4, lower renal GFR
        'description':   '>65 y: ↓TfR, mildly leaky BBB, ↑systemic CL',
        'refs':          'Nájera Pharmaceutics 2025',
    },
}

# ── Disease-related scaling factors ───────────────────────────────────────────

DISEASE_FACTORS: dict[str, dict] = {
    'healthy': {
        'k_bind_scale':  1.00,
        'k_trans_scale': 1.00,
        'k_endo_scale':  1.00,
        'CL_scale':      1.00,
        'description':   'Intact BBB, normal receptor expression',
        'refs':          'baseline',
    },
    'glioma': {
        'k_bind_scale':  1.50,   # TfR overexpression in grade III–IV glioma
        'k_trans_scale': 1.40,   # EPR effect + focal BBB disruption
        'k_endo_scale':  1.20,   # Higher tumour-cell endocytic activity
        'CL_scale':      0.90,   # Altered hepatic metabolism in cancer patients
        'description':   'Glioma: TfR++ (EPR), BBB disruption, ↑transcytosis',
        'refs':          'Wiley PNAS 2015; Bhattacharya 2022',
    },
    'neuroinflammation': {
        'k_bind_scale':  1.20,   # Upregulated inflammation markers on BBB endothelium
        'k_trans_scale': 1.60,   # Significant BBB disruption by cytokines (IL-1β, TNF-α)
        'k_endo_scale':  1.30,   # Activated microglia → enhanced phagocytosis
        'CL_scale':      1.10,   # Systemic inflammation → altered drug metabolism
        'description':   'Neuro-inflammation: ↑BBB permeability, activated microglia',
        'refs':          'Khung ACS Nano 2023',
    },
    'alzheimer': {
        'k_bind_scale':  0.85,   # Reduced TfR expression (neurodegeneration)
        'k_trans_scale': 0.90,   # Impaired AQP4/transcytosis machinery; amyloid plaques
        'k_endo_scale':  0.95,   # Reduced endosomal trafficking
        'CL_scale':      1.15,   # Elevated P-gp efflux; altered BBB transporters
        'description':   'AD: ↓TfR, impaired transcytosis, amyloid burden, ↑P-gp',
        'refs':          'Mucke Science 2009; Zlokovic Nature Rev Neurosci 2011',
    },
}

VALID_AGES     = list(AGE_FACTORS.keys())
VALID_DISEASES = list(DISEASE_FACTORS.keys())


# ── VirtualPatient dataclass ───────────────────────────────────────────────────

@dataclass
class VirtualPatient:
    age_group:     str = 'adult'    # pediatric | adult | elderly
    disease_state: str = 'healthy'  # healthy | glioma | neuroinflammation | alzheimer

    def __post_init__(self):
        if self.age_group not in AGE_FACTORS:
            raise ValueError(f"age_group must be one of {VALID_AGES}, got '{self.age_group}'")
        if self.disease_state not in DISEASE_FACTORS:
            raise ValueError(f"disease_state must be one of {VALID_DISEASES}, got '{self.disease_state}'")

    @property
    def label(self) -> str:
        return f"{self.age_group}_{self.disease_state}"

    def scaling(self) -> dict:
        """Combined age × disease multiplicative scaling factors."""
        af = AGE_FACTORS[self.age_group]
        df = DISEASE_FACTORS[self.disease_state]
        return {
            'k_bind_scale':  af['k_bind_scale']  * df['k_bind_scale'],
            'k_trans_scale': af['k_trans_scale'] * df['k_trans_scale'],
            'k_endo_scale':  af['k_endo_scale']  * df['k_endo_scale'],
            'CL_scale':      af['CL_scale']      * df['CL_scale'],
        }

    def description(self) -> str:
        af = AGE_FACTORS[self.age_group]
        df = DISEASE_FACTORS[self.disease_state]
        return f"{af['description']}  ×  {df['description']}"


# ── Public API ─────────────────────────────────────────────────────────────────

def parse_patient_type(patient_type: str) -> VirtualPatient:
    """
    Parse a patient label string into a VirtualPatient.
    Accepts formats: "adult_healthy", "elderly_glioma", etc.
    Also handles aliases: "healthy_adult" → age=adult, disease=healthy.
    """
    s = str(patient_type).strip().lower()
    # Handle common aliases
    aliases = {
        'adult_healthy': ('adult', 'healthy'),
        'healthy_adult': ('adult', 'healthy'),
        'adult':         ('adult', 'healthy'),
        'healthy':       ('adult', 'healthy'),
    }
    if s in aliases:
        age, dis = aliases[s]
        return VirtualPatient(age, dis)

    parts = s.split('_', 1)
    if len(parts) == 2 and parts[0] in VALID_AGES and parts[1] in VALID_DISEASES:
        return VirtualPatient(parts[0], parts[1])
    if len(parts) == 2 and parts[1] in VALID_AGES and parts[0] in VALID_DISEASES:
        return VirtualPatient(parts[1], parts[0])

    raise ValueError(
        f"Cannot parse patient_type '{patient_type}'. "
        f"Use '<age>_<disease>' where age ∈ {VALID_AGES} and disease ∈ {VALID_DISEASES}."
    )


def apply_patient_scaling(params, patient_type) -> object:
    """
    Apply patient-specific PBPK parameter scaling.

    Parameters
    ----------
    params       : PBPKParams instance from pbpk_simulator.py
    patient_type : str (e.g. "adult_healthy") or VirtualPatient instance

    Returns a new PBPKParams with scaled k_bind, k_endo, k_trans, CL.
    """
    from pbpk_simulator import PBPKParams

    if isinstance(patient_type, str):
        patient = parse_patient_type(patient_type)
    else:
        patient = patient_type

    s = patient.scaling()
    return PBPKParams(
        k_bind    = float(np.clip(params.k_bind    * s['k_bind_scale'],  0.001, 5.0)),
        k_off     = params.k_off,
        k_endo    = float(np.clip(params.k_endo    * s['k_endo_scale'],  0.01,  1.0)),
        k_trans   = float(np.clip(params.k_trans   * s['k_trans_scale'], 0.001, 0.5)),
        k_lyso    = params.k_lyso,
        k_recycle = params.k_recycle,
        CL        = float(np.clip(params.CL        * s['CL_scale'],      0.05,  5.0)),
    )


def list_profiles() -> list[dict]:
    """Return all 12 patient profiles as dicts (for Agent tool)."""
    profiles = []
    for age in VALID_AGES:
        for disease in VALID_DISEASES:
            p = VirtualPatient(age, disease)
            s = p.scaling()
            profiles.append({
                'label':         p.label,
                'age_group':     age,
                'disease_state': disease,
                'description':   p.description(),
                **{k: round(v, 3) for k, v in s.items()},
                'refs': (AGE_FACTORS[age]['refs'] + '; '
                         + DISEASE_FACTORS[disease]['refs']).strip('; '),
            })
    return profiles


def compare_patients_for_design(design: dict,
                                 patient_types: list[str] | None = None) -> list[dict]:
    """
    Run the same NP design across multiple patient profiles and compare AUC.
    Returns list of results sorted by AUC_brain descending.
    """
    from pbpk_simulator import pbpk_simulate

    if patient_types is None:
        patient_types = [p['label'] for p in list_profiles()]

    results = []
    for pt in patient_types:
        try:
            r = pbpk_simulate(patient_type=pt, **design)
            s = parse_patient_type(pt).scaling()
            results.append({
                'patient':     pt,
                'AUC_brain':   round(r['AUC'], 6),
                'AUC_ratio':   round(r['AUC_ratio'], 4),
                'k_bind':      round(r['params'].k_bind, 4),
                'k_trans':     round(r['params'].k_trans, 4),
                'CL':          round(r['params'].CL, 4),
                'k_bind_scale': round(s['k_bind_scale'], 3),
                'description': parse_patient_type(pt).description(),
            })
        except Exception as e:
            results.append({'patient': pt, 'error': str(e)})

    results.sort(key=lambda x: x.get('AUC_brain', -1), reverse=True)
    return results


# ── CLI demo ───────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 70)
    print("Virtual Patient Profiles — PBPK Parameter Scaling")
    print("=" * 70)
    print(f"\n{'Label':<30} {'k_bind':>8} {'k_trans':>8} {'k_endo':>8} {'CL':>8}")
    print("-" * 70)
    for p in list_profiles():
        print(f"  {p['label']:<28} {p['k_bind_scale']:>8.2f} "
              f"{p['k_trans_scale']:>8.2f} {p['k_endo_scale']:>8.2f} {p['CL_scale']:>8.2f}")

    print("\n\nCross-patient comparison (optimal design: 80nm, anti-TfR, -15mV):")
    design = {'size_nm': 80, 'zeta_mv': -15, 'peg': 'yes',
              'ligand_type': 'anti_tfr', 'ligand_density': 30}
    results = compare_patients_for_design(design)
    print(f"\n  {'Patient':<30} {'AUC_brain':>10} {'AUC_ratio':>10} {'CL':>8}")
    print("  " + "-" * 60)
    for r in results:
        if 'error' not in r:
            print(f"  {r['patient']:<30} {r['AUC_brain']:>10.5f} "
                  f"{r['AUC_ratio']:>10.4f} {r['CL']:>8.4f}")
