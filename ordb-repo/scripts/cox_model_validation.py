"""
ORDB Step 4: Cox Proportional Hazards modeling -- CODE VALIDATION ONLY.

IMPORTANT: This script runs on SYNTHETIC data generated specifically to
test whether the modeling code works correctly. It does NOT contain real
hallucination observations. Nothing here should be read as a scientific
finding about actual hallucination risk. It exists to prove the pipeline
is technically sound, ready for real data + a real LLM API key.

Real next step: replace generate_synthetic_validation_data() with actual
labeled output from pipeline.py once steps 1-3 have been run for real.
"""

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index


# ---------------------------------------------------------------------
# Synthetic validation data. We deliberately BUILD IN a volume effect
# (more items synthesized -> higher hallucination hazard) so we can
# check the Cox model correctly RECOVERS a known pattern. This is a
# sanity check on the code, not a claim about real hallucination data.
# ---------------------------------------------------------------------
def generate_synthetic_validation_data(n_groups: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    n_items = rng.integers(3, 25, size=n_groups)          # "time" variable: synthesis volume
    score_variance = rng.uniform(0.1, 2.0, size=n_groups)  # content-ambiguity covariate

    # Built-in ground truth relationship (for validation only):
    # hazard increases with n_items AND with score_variance
    true_hazard_scale = 0.02 * n_items + 0.15 * score_variance
    hallucinated = rng.binomial(1, p=np.clip(true_hazard_scale, 0, 0.95))

    return pd.DataFrame({
        "n_items_synthesized": n_items,
        "score_variance": score_variance,
        "hallucinated": hallucinated,  # 1 = event occurred (hallucination), 0 = censored
    })


def fit_cox_model(df: pd.DataFrame) -> CoxPHFitter:
    cph = CoxPHFitter()
    # duration = n_items_synthesized (our "time" axis)
    # event = hallucinated (1) or not (0, treated as censored)
    cph.fit(df, duration_col="n_items_synthesized", event_col="hallucinated",
            formula="score_variance")
    return cph


def evaluate_model(cph: CoxPHFitter, df: pd.DataFrame) -> float:
    partial_hazard = cph.predict_partial_hazard(df)
    c_index = concordance_index(df["n_items_synthesized"], -partial_hazard, df["hallucinated"])
    return c_index


if __name__ == "__main__":
    print("=" * 70)
    print("CODE VALIDATION ONLY -- synthetic data, not real hallucination data")
    print("=" * 70)

    df = generate_synthetic_validation_data(n_groups=200)
    print(f"\nGenerated {len(df)} synthetic groups.")
    print(f"Hallucination rate in synthetic data: {df['hallucinated'].mean():.1%}")
    print(f"\nSample rows:\n{df.head()}")

    cph = fit_cox_model(df)
    print("\n--- Cox Proportional Hazards model summary ---")
    print(cph.summary[["coef", "exp(coef)", "p"]])

    c_index = evaluate_model(cph, df)
    print(f"\nConcordance index (C-index): {c_index:.3f}")
    print("(0.5 = random, 1.0 = perfect ranking of risk)")

    print("\n--- Sanity check ---")
    print("We built in a real relationship: more items + more score variance")
    print("-> higher hallucination hazard. If the model recovers a positive,")
    print("statistically significant coefficient on score_variance, and a")
    print("C-index meaningfully above 0.5, the CODE is working correctly.")
    print("This does NOT mean real hallucination data will show this pattern.")
