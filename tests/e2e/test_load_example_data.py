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
    page.get_by_role("tab", name="📁 Data").click()

    # 2. Verify button existence
    load_btn = page.locator("#data-btn_load_example")
    expect(load_btn).to_be_visible()

    # 3. Click the button
    load_btn.click()

    # 4. Verify notification appears
    # Using expect automatically handles waiting/polling
    try:
        expect(page.get_by_text("Loaded 1600 Clinical Records")).to_be_visible(
            timeout=15000
        )
    except Exception:
        # Fallback check for metadata if notification is missed
        expect(page.locator("#data-ui_file_metadata")).to_contain_text(
            "1,600 rows", timeout=5000
        )

    # 5. Verify the data table is populated
    grid = page.locator("#data-out_df_preview")
    expect(grid).to_contain_text("Treatment_Group", timeout=5000)
    expect(grid).to_contain_text("Age_Years", timeout=5000)

    # 6. Verify row count info in metadata section
    metadata_div = page.locator("#data-ui_file_metadata")
    expect(metadata_div).to_contain_text("1,600 rows")

    # 7. Check if specific mapping is working
    page.locator("#data-sel_var_edit").select_option("Treatment_Group")
    expect(page.get_by_text("0=Standard Care")).to_be_visible()
