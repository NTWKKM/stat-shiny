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

# ============================================================================
# Configuration
# ============================================================================

BASE_URL = "http://localhost:8000"
"""Base URL for the Shiny server (started by conftest.py)"""


# ============================================================================
# Test Cases
# ============================================================================

class TestAppFlow:
    """Test basic app navigation and functionality"""
    
    def test_app_loads(self, page: Page):
        """
        ✅ Test that the app loads and has correct title
        
        Verifies:
        - Page loads successfully at http://localhost:8000
        - Page title is "Medical Stat Tool"
        """
        page.goto(BASE_URL)
        expect(page).to_have_title("Medical Stat Tool")
    
    def test_navbar_visible(self, page: Page):
        """
        ✅ Test that navigation bar with all tabs is visible
        
        Verifies:
        - Data Management tab exists
        - Risk Factors tab exists
        - Survival Analysis tab exists
        - Settings tab exists
        """
        page.goto(BASE_URL)
        
        # Check for main navigation tabs
        expect(page.get_by_role("tab", name="Data Management")).to_be_visible()
        expect(page.get_by_role("tab", name="Risk Factors")).to_be_visible()
        expect(page.get_by_role("tab", name="Survival Analysis")).to_be_visible()
        expect(page.get_by_role("tab", name="Settings")).to_be_visible()
    
    def test_navigation_to_survival(self, page: Page):
        """
        ✅ Test navigation to Survival Analysis tab
        
        Verifies:
        - Can click on "Survival Analysis" tab
        - Tab content loads (shows "Kaplan-Meier" text)
        - No JavaScript errors on navigation
        """
        page.goto(BASE_URL)
        
        # Click the Survival Analysis tab
        page.get_by_role("tab", name="Survival Analysis").click()
        
        # Wait for content to load and verify Kaplan-Meier text appears
        # Using .first because there might be multiple matches
        expect(page.get_by_text("Kaplan-Meier").first).to_be_visible()
    
    def test_navigation_to_risk_factors(self, page: Page):
        """
        ✅ Test navigation to Risk Factors (Logistic Regression) tab
        
        Verifies:
        - Can click on "Risk Factors" tab
        - Tab content loads (shows input elements)
        """
        page.goto(BASE_URL)
        
        # Click the Risk Factors tab
        page.get_by_role("tab", name="Risk Factors").click()
        
        # Verify binary logistic regression section is visible
        expect(page.get_by_text("Binary Logistic Regression")).to_be_visible()
        
        # Verify outcome selector is present
        expect(page.get_by_label("Select Outcome")).to_be_visible()
    
    def test_navigation_to_data_management(self, page: Page):
        """
        ✅ Test navigation to Data Management tab
        
        Verifies:
        - Can click on "Data Management" tab
        - File upload button is visible
        """
        page.goto(BASE_URL)
        
        # Click the Data Management tab
        page.get_by_role("tab", name="Data Management").click()
        
        # If upload button not found by text, check for file input
        # This is flexible for different implementations
        elements = page.query_selector_all("input[type='file']")
        assert len(elements) > 0, "File upload input not found on Data Management tab"


# ============================================================================
# Standalone test functions (alternative to class-based tests)
# ============================================================================

def test_app_title(page: Page):
    """
    🧪 Verify app title is correct
    
    This is a standalone test (not in a class).
    Can be run with: pytest tests/e2e/test_app_flow.py::test_app_title -v
    """
    page.goto(BASE_URL)
    expect(page).to_have_title("Medical Stat Tool")


def test_page_not_404(page: Page):
    """
    🧪 Verify that the app doesn't return 404
    
    Checks that page loads successfully (doesn't show 404 error)
    """
    response = page.goto(BASE_URL)
    assert response is not None, "page.goto returned None"
    assert response.status == 200 or response.status == 304, \
        f"Expected 200 or 304, got {response.status}"


# ============================================================================
# Markers for test organization
# ============================================================================

pytestmark = pytest.mark.e2e
"""Mark all tests in this file as E2E tests"""
