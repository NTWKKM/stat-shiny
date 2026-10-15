"""
Unit tests for Shinystat AI Agent Skills promotion in Home tab and Data sidebar.
"""

from tabs.tab_data import data_ui
from tabs.tab_home import home_ui


def test_home_ui_contains_skills_promotion():
    """Verify that home_ui contains the AI agent skills promotion in sidebar and showcase."""
    rendered_ui = str(home_ui("home").tagify())

    # Check sidebar promotion
    assert "Shinystat AI Skills" in rendered_ui
    assert "npx skills add https://github.com/ntwkkm/shinystat-skills" in rendered_ui
    assert "https://github.com/ntwkkm/shinystat-skills" in rendered_ui

    # Check the two skills breakdown
    assert "shinystat" in rendered_ui
    assert "shinystat-cloud" in rendered_ui
    assert "--skill shinystat" in rendered_ui
    assert "--skill shinystat-cloud" in rendered_ui

    # Check supported agent mentions
    assert "Antigravity" in rendered_ui
    assert "Claude Code" in rendered_ui
    assert "Cursor" in rendered_ui


def test_data_ui_sidebar_contains_skills_promotion():
    """Verify that data_ui sidebar contains the AI agent skills card."""
    rendered_ui = str(data_ui("data").tagify())

    assert "Shinystat for AI Agents" in rendered_ui
    assert "npx skills add https://github.com/ntwkkm/shinystat-skills" in rendered_ui
    assert "https://github.com/ntwkkm/shinystat-skills" in rendered_ui
