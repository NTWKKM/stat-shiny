import os

import pytest
from playwright.sync_api import Page, expect

BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000")


@pytest.mark.e2e
def test_load_example_data(page: Page):
    """
    ✅ Test that "Load Example Data" button works correctly.
    """
    page.goto(BASE_URL)

    # 1. Navigate to Data tab
    page.get_by_role("tab", name="📁 Data Management").click()
    page.wait_for_timeout(1000)

    # 2. Verify button existence and click
    load_btn = page.get_by_role("button", name="📄 Load Example Data")
    expect(load_btn).to_be_visible()
    load_btn.click()

    # 3. Verify the data table and metadata are populated
    grid = page.locator("#data-out_df_preview")
    expect(grid).to_contain_text("Treatment_Group", timeout=25000)
    expect(grid).to_contain_text("Age_Years", timeout=5000)

    # 4. Verify row count info in metadata section
    metadata_div = page.locator("#data-ui_file_metadata")
    expect(metadata_div).to_contain_text("1,600 rows", timeout=10000)

    # 5. Check if specific mapping is working
    page.locator("#data-sel_var_edit").select_option("Treatment_Group")
    expect(page.get_by_text("0=Standard Care")).to_be_visible(timeout=10000)
