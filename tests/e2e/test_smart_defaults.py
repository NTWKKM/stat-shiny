import os

import pytest
from playwright.sync_api import Page, expect

BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000")


@pytest.mark.e2e
def test_smart_variable_defaults(page: Page):
    """
    ✅ Verify that "Load Example Data" triggers smart variable selection
    across different analysis tabs.
    """
    # 1. Load the App
    page.goto(BASE_URL)

    # 2. Go to Data Tab and Click Load Example Data
    # Standalone tab
    page.get_by_role("tab", name="📁 Data Management").click()

    load_btn = page.locator("#data-btn_load_example")
    expect(load_btn).to_be_visible()
    load_btn.click()

    # Wait for data to load
    expect(page.get_by_text("Loaded 1600 Clinical Records")).to_be_visible(
        timeout=15000
    )

    # 3. Verify Tab: Correlation
    # Navigate: General Statistics -> Correlation Analysis
    # Use role="button" to target the dropdown toggle specifically
    page.get_by_role("button", name="📊 General Statistics").click()
    page.get_by_role("tab", name="Correlation Analysis").click()

    # Check values
    expect(page.locator("#corr-cv1")).to_have_value("Lab_Glucose")
    expect(page.locator("#corr-cv2")).to_have_value("Lab_HbA1c")

    # 4. Verify Tab: Diagnostic (ROC)
    page.get_by_role("button", name="📊 General Statistics").click()
    page.get_by_role("tab", name="Diagnostic Tests").click()

    # Select Subtab "📈 ROC Curve & AUC"
    page.get_by_role("tab", name="📈 ROC Curve & AUC").click()

    expect(page.locator("#diag-sel_roc_truth")).to_have_value("Gold_Standard_Disease")
    expect(page.locator("#diag-sel_roc_score")).to_have_value("Test_Score_Rapid")

    # 5. Verify Tab: Survival
    # Navigate: Advanced Statistics -> Survival Analysis
    page.get_by_role("button", name="🔬 Advanced Statistics").click()
    page.get_by_role("tab", name="Survival Analysis").click()

    expect(page.locator("#survival-surv_time")).to_have_value("Time_Months")
    expect(page.locator("#survival-surv_event")).to_have_value("Status_Death")
    # Group defaults to Treatment_Group
    expect(page.locator("#survival-surv_group")).to_have_value("Treatment_Group")

    # 6. Verify Tab: Causal Inference
    # Navigate: Clinical Research Tools -> Causal Methods
    page.get_by_role("button", name="🏥 Clinical Research Tools").click()
    page.get_by_role("tab", name="Causal Methods").click()

    # Select Subtab "⚖️ PSM & IPW"
    page.get_by_role("tab", name="⚖️ PSM & IPW").click()

    expect(page.locator("#causal-psm_treatment")).to_have_value("Treatment_Group")
    expect(page.locator("#causal-psm_outcome")).to_have_value("Outcome_Cured")

    # 7. Verify Tab: Agreement
    page.get_by_role("button", name="📊 General Statistics").click()
    page.get_by_role("tab", name="Agreement & Reliability").click()

    # Defaults often pick Diagnosis_Dr_A / Dr_B if keywords are right
    expect(page.locator("#agreement-sel_kappa_v1")).to_have_value("Diagnosis_Dr_A")
    expect(page.locator("#agreement-sel_kappa_v2")).to_have_value("Diagnosis_Dr_B")

    # 8. Verify Tab: Regression Analysis (Logit)
    page.get_by_role("button", name="🔬 Advanced Statistics").click()
    page.get_by_role("tab", name="Regression Analysis").click()
    # Subtab usually defaults to Logistic
    expect(page.locator("#core_reg-sel_outcome")).to_have_value("Outcome_Cured")

    # 9. Verify Tab: Advanced Regression (Mediation/SEM)
    page.get_by_role("button", name="🔬 Advanced Statistics").click()
    page.get_by_role("tab", name="Advanced Regression").click()

    # Mediation Analysis default
    expect(page.locator("#adv_inf-med_outcome")).to_have_value("Outcome_Cured")
    expect(page.locator("#adv_inf-med_treatment")).to_have_value("Treatment_Group")

    # 10. Verify Tab: Baseline Matching (Table 1)
    page.get_by_role("tab", name="📋 Table 1 & Matching").click()
    expect(page.locator("#bm-sel_group_col")).to_have_value("Treatment_Group")

    print("✅ Smart Variable Selection Verification Passed")
