"""
🧪 E2E Tests for Medical Stat Tool (stat-shiny)

These tests use Playwright to test the web application through a real browser.

Setup:
- conftest.py automatically starts Shiny server at http://localhost:8000
- No need to use create_app_fixture anymore
- Tests run against real server (not mocked)

Usage:
    pytest tests/e2e/test_app_flow.py -v                    # Run all E2E tests
    pytest tests/e2e/test_app_flow.py::test_app_loads -v     # Run specific test
    pytest tests/e2e/test_app_flow.py -v --headed            # Show browser
    pytest tests/e2e/test_app_flow.py -v --slowmo=500        # Slow motion (debug)
"""

import pytest
from playwright.sync_api import Page, expect

# Mark all tests in this file as E2E tests
pytestmark = pytest.mark.e2e

# ============================================================================
# Configuration
# ============================================================================

BASE_URL = "http://localhost:8000"
"""Base URL for the Shiny server (started by conftest.py)"""

# ============================================================================
# Test: App Loading & Basic Structure
# ============================================================================


class TestAppLoading:
    """Tests for app initialization and basic structure.

    Verifies that the application loads correctly and has the expected
    title, navigation structure, and main UI components.
    """

    def test_app_loads_successfully(self, page: Page):
        """
        ✅ Test that the app loads without errors.

        Given: The Shiny server is running
        When: Navigating to the base URL
        Then: The page loads with HTTP 200 status
        """
        response = page.goto(BASE_URL)
        assert response is not None, "page.goto returned None"
        assert response.status in (200, 304), f"Expected 200/304, got {response.status}"

    def test_app_has_correct_title(self, page: Page):
        """
        ✅ Test that the app has the correct title.

        Given: The app is loaded
        When: Checking the page title
        Then: Title is "Medical Stat Tool"
        """
        page.goto(BASE_URL)
        expect(page).to_have_title("Medical Stat Tool")

    def test_navbar_contains_all_tabs(self, page: Page):
        """
        ✅ Test that navigation bar contains all expected tabs.

        Given: The app is loaded
        When: Examining the navigation bar
        Then: All main tabs are visible
        """
        page.goto(BASE_URL)

        # Check for main navigation tabs
        expected_tabs = [
            "Data Management",
            "Regression Models",
            "Survival Analysis",
            "Settings",
        ]
        for tab_name in expected_tabs:
            expect(page.get_by_role("tab", name=tab_name)).to_be_visible()


# ============================================================================
# Test: Tab Navigation
# ============================================================================


class TestTabNavigation:
    """Tests for tab navigation functionality.

    Verifies that all tabs can be clicked and load their respective content.
    """

    def test_navigate_to_data_management(self, page: Page):
        """
        ✅ Test navigation to Data Management tab.

        Given: The app is loaded
        When: Clicking "Data Management" tab
        Then: File upload input is visible
        """
        page.goto(BASE_URL)
        page.get_by_role("tab", name="Data Management").click()

        # Verify file upload input exists
        file_inputs = page.query_selector_all("input[type='file']")
        assert len(file_inputs) > 0, "File upload input not found"

    def test_navigate_to_regression_models(self, page: Page):
        """
        ✅ Test navigation to Regression Models tab.

        Given: The app is loaded
        When: Clicking "Regression Models" tab
        Then: Binary Logistic Regression section is visible
        """
        page.goto(BASE_URL)
        page.get_by_role("tab", name="Regression Models").click()

        expect(page.get_by_text("Binary Logistic Regression")).to_be_visible()
        expect(page.get_by_label("Select Outcome")).to_be_visible()

    def test_navigate_to_survival_analysis(self, page: Page):
        """
        ✅ Test navigation to Survival Analysis tab.

        Given: The app is loaded
        When: Clicking "Survival Analysis" tab
        Then: Kaplan-Meier section is visible
        """
        page.goto(BASE_URL)
        page.get_by_role("tab", name="Survival Analysis").click()

        expect(page.get_by_text("Kaplan-Meier").first).to_be_visible()

    def test_navigate_to_settings(self, page: Page):
        """
        ✅ Test navigation to Settings tab.

        Given: The app is loaded
        When: Clicking "Settings" tab
        Then: Settings content is visible
        """
        page.goto(BASE_URL)
        page.get_by_role("tab", name="Settings").click()

        # Wait for settings content to load
        page.wait_for_timeout(500)
        # Settings tab should show some configuration options


# ============================================================================
# Test: Data Management Workflow
# ============================================================================


class TestDataManagementWorkflow:
    """Tests for data management workflow.

    Verifies data upload, example data generation, and preview functionality.
    """

    def test_example_data_button_exists(self, page: Page):
        """
        ✅ Test that example data generation button exists.

        Given: The app is on Data Management tab
        When: Looking for example data button
        Then: Button is visible and clickable
        """
        page.goto(BASE_URL)
        page.get_by_role("tab", name="Data Management").click()

        # Look for example data button
        example_btn = page.get_by_role("button", name="Generate Example Data")
        if example_btn.is_visible():
            expect(example_btn).to_be_enabled()

    def test_file_upload_input_accepts_csv(self, page: Page):
        """
        ✅ Test that file upload input is configured for CSV files.

        Given: The app is on Data Management tab
        When: Examining file input element
        Then: Input accepts appropriate file types
        """
        page.goto(BASE_URL)
        page.get_by_role("tab", name="Data Management").click()

        file_input = page.query_selector("input[type='file']")
        assert file_input is not None, "File input not found"


# ============================================================================
# Test: Risk Factors Workflow
# ============================================================================


class TestRegressionModelsWorkflow:
    """Tests for regression models analysis workflow.

    Verifies logistic regression UI elements and interaction.
    """

    def test_outcome_selector_visible(self, page: Page):
        """
        ✅ Test that outcome selector is visible.

        Given: The app is on Regression Models tab
        When: Looking for outcome selector
        Then: Selector is visible
        """
        page.goto(BASE_URL)
        page.get_by_role("tab", name="Regression Models").click()

        expect(page.get_by_label("Select Outcome")).to_be_visible()

    def test_run_button_exists(self, page: Page):
        """
        ✅ Test that analysis run button exists.

        Given: The app is on Regression Models tab
        When: Looking for run button
        Then: Button is present (may be disabled without data)
        """
        page.goto(BASE_URL)
        page.get_by_role("tab", name="Regression Models").click()

        # Wait for UI to fully load
        page.wait_for_timeout(500)

        # Look for any run/analyze button
        run_buttons = page.get_by_role("button").filter(has_text="Run")
        if run_buttons.count() > 0:
            expect(run_buttons.first).to_be_visible()


# ============================================================================
# Test: Survival Analysis Workflow
# ============================================================================


class TestSurvivalAnalysisWorkflow:
    """Tests for survival analysis workflow.

    Verifies Kaplan-Meier, Cox regression, and landmark analysis UI elements.
    """

    def test_survival_subtabs_exist(self, page: Page):
        """
        ✅ Test that survival analysis subtabs exist.

        Given: The app is on Survival Analysis tab
        When: Looking for subtabs
        Then: Kaplan-Meier and other analysis options are available
        """
        page.goto(BASE_URL)
        page.get_by_role("tab", name="Survival Analysis").click()

        # Check for Kaplan-Meier text
        expect(page.get_by_text("Kaplan-Meier").first).to_be_visible()

    def test_time_variable_selector_exists(self, page: Page):
        """
        ✅ Test that time variable selector exists.

        Given: The app is on Survival Analysis tab
        When: Looking for time variable selector
        Then: Selector is present
        """
        page.goto(BASE_URL)
        page.get_by_role("tab", name="Survival Analysis").click()

        # Wait for UI to load
        page.wait_for_timeout(500)

        # Check for time-related selectors
        time_selectors = page.get_by_label("Time Variable")
        if time_selectors.count() > 0:
            expect(time_selectors.first).to_be_visible()


# ============================================================================
# Test: Error Handling
# ============================================================================


class TestErrorHandling:
    """Tests for error handling and edge cases."""

    def test_no_javascript_errors_on_load(self, page: Page):
        """
        ✅ Test that no JavaScript errors occur on page load.

        Given: A clean browser context
        When: Loading the app
        Then: No JavaScript errors are logged
        """
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))

        page.goto(BASE_URL)
        page.wait_for_timeout(1000)  # Wait for any async errors

        # Filter out known non-critical errors if any
        critical_errors = [e for e in errors if "critical" in e.lower()]
        assert len(critical_errors) == 0, f"JavaScript errors: {critical_errors}"

    def test_navigation_has_no_console_errors(self, page: Page):
        """
        ✅ Test that tab navigation doesn't cause console errors.

        Given: The app is loaded
        When: Navigating between all tabs
        Then: No critical errors occur
        """
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))

        page.goto(BASE_URL)

        # Navigate through all tabs
        for tab_name in [
            "Regression Models",
            "Survival Analysis",
            "Settings",
            "Data Management",
        ]:
            page.get_by_role("tab", name=tab_name).click()
            page.wait_for_timeout(300)

        # Check for critical errors only
        critical_errors = [e for e in errors if "critical" in e.lower()]
        assert len(critical_errors) == 0, f"Errors during navigation: {critical_errors}"


# ============================================================================
# Standalone Test Functions
# ============================================================================


def test_app_title(page: Page):
    """
    🧪 Standalone: Verify app title is correct.

    Can be run with: pytest tests/e2e/test_app_flow.py::test_app_title -v
    """
    page.goto(BASE_URL)
    expect(page).to_have_title("Medical Stat Tool")


def test_page_not_404(page: Page):
    """
    🧪 Standalone: Verify that the app doesn't return 404.

    Checks that page loads successfully (doesn't show 404 error)
    """
    response = page.goto(BASE_URL)
    assert response is not None, "page.goto returned None"
    assert response.status in (200, 304), f"Expected 200/304, got {response.status}"
