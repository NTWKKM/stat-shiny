"""
Unit tests for utils/reporting_checklists.py

Tests CONSORT and STROBE checklist creation and validation.
"""

from utils.reporting_checklists import (
    ChecklistItem,
    ChecklistStatus,
    auto_populate_strobe,
    create_consort_checklist,
    create_strobe_checklist,
    format_strobe_html_compact,
    generate_checklist_markdown,
)
import pytest


class TestChecklistItem:
    """Tests for ChecklistItem dataclass."""

    def test_item_creation(self):
        """Test creating a checklist item."""
        item = ChecklistItem(
            number="1a",
            item="Title",
            description="Identification as a randomised trial",
            section="Title and Abstract",
        )

        assert item.number == "1a"
        assert item.item == "Title"
        assert item.status == ChecklistStatus.NOT_DONE
        assert item.page_number == ""

    def test_item_with_status(self):
        """Test item with custom status."""
        item = ChecklistItem(
            number="2",
            item="Background",
            description="Scientific background",
            section="Introduction",
            status=ChecklistStatus.COMPLETE,
            page_number="5",
            notes="See paragraph 2",
        )

        assert item.status == ChecklistStatus.COMPLETE
        assert item.page_number == "5"
        assert item.notes == "See paragraph 2"


class TestCONSORTChecklist:
    """Tests for CONSORT checklist creation."""

    def test_create_consort_checklist(self):
        """Test creating CONSORT checklist."""
        checklist = create_consort_checklist()

        assert checklist.name == "CONSORT 2010"
        assert len(checklist.items) > 20  # CONSORT has 25+ items

    def test_consort_sections(self):
        """Test CONSORT checklist sections."""
        checklist = create_consort_checklist()
        sections = set(item.section for item in checklist.items)

        assert "Title and Abstract" in sections
        assert "Introduction" in sections
        assert "Methods" in sections
        assert "Results" in sections
        assert "Discussion" in sections

    def test_consort_update_item(self):
        """Test updating a CONSORT checklist item."""
        checklist = create_consort_checklist()

        success = checklist.update_item(
            "1a",
            ChecklistStatus.COMPLETE,
            page="1",
            notes="Title includes 'randomized'",
        )

        assert success
        # Find the updated item
        item = next(i for i in checklist.items if i.number == "1a")
        assert item.status == ChecklistStatus.COMPLETE
        assert item.page_number == "1"
        assert item.notes == "Title includes 'randomized'"

    def test_update_item_preserves_values(self):
        """Test that update_item preserves existing values when inputs are None."""
        checklist = create_consort_checklist()
        # First set values
        checklist.update_item(
            "1a", ChecklistStatus.PARTIAL, page="5", notes="Initial note"
        )

        # Update only status
        checklist.update_item("1a", ChecklistStatus.COMPLETE)

        item = next(i for i in checklist.items if i.number == "1a")
        assert item.status == ChecklistStatus.COMPLETE
        assert item.page_number == "5"  # Should be preserved
        assert item.notes == "Initial note"  # Should be preserved


class TestSTROBEChecklist:
    """Tests for STROBE checklist creation."""

    def test_create_strobe_cohort(self):
        """Test creating STROBE cohort checklist."""
        checklist = create_strobe_checklist("cohort")

        assert "STROBE" in checklist.name
        assert "Cohort" in checklist.name
        assert len(checklist.items) > 15

    def test_create_strobe_case_control(self):
        """Test creating STROBE case-control checklist."""
        checklist = create_strobe_checklist("case_control")

        assert "Case Control" in checklist.name


class TestChecklistCompletion:
    """Tests for checklist completion tracking."""

    def test_completion_summary_empty(self):
        """Test completion summary with no items complete."""
        checklist = create_consort_checklist()
        summary = checklist.get_completion_summary()

        assert summary["complete"] == 0
        assert summary["total_applicable"] > 0
        assert summary["completion_rate"] == 0

    def test_completion_summary_partial(self):
        """Test completion summary with some items complete."""
        checklist = create_consort_checklist()

        # Complete first 3 items
        for i, item in enumerate(checklist.items[:3]):
            checklist.update_item(item.number, ChecklistStatus.COMPLETE)

        summary = checklist.get_completion_summary()
        assert summary["complete"] == 3
        assert summary["completion_rate"] > 0

    def test_get_completion_summary_excludes_not_applicable(self):
        """Test completion summary excludes NOT_APPLICABLE items."""
        checklist = create_consort_checklist()

        # Mark items as N/A checks
        # 25 items in CONSORT. Let's make 5 N/A.
        items_to_na = checklist.items[:5]
        for item in items_to_na:
            checklist.update_item(item.number, ChecklistStatus.NOT_APPLICABLE)

        # Complete 5 items
        items_to_complete = checklist.items[5:10]
        for item in items_to_complete:
            checklist.update_item(item.number, ChecklistStatus.COMPLETE)

        summary = checklist.get_completion_summary()

        # Total applicable should be Total - 5
        expected_total = len(checklist.items) - 5
        assert summary["total_applicable"] == expected_total
        assert summary["complete"] == 5
        # 5/20 * 100 = 25.0%
        expected_rate = round(5 / expected_total * 100, 1)
        assert summary["completion_rate"] == expected_rate


class TestSTROBEAutoPopulation:
    """Tests for STROBE auto-population and formatting."""

    def test_auto_populate_strobe_logic(self):
        """Test auto_populate_strobe with various metadata."""
        # Case 1: Minimal metadata (defaults)
        checklist = auto_populate_strobe({}, study_type="cohort")
        # Check basic defaulting
        # Item 5 (Setting) should not be set if n_total is 0
        item_5 = next(i for i in checklist.items if i.number == "5")
        assert item_5.status == ChecklistStatus.NOT_DONE

        # Case 2: Rich metadata
        meta = {
            "n_total": 100,
            "n_analyzed": 90,
            "outcome_name": "Death",
            "predictors": ["Age", "Sex"],
            "has_missing_report": True,
            "has_ci": True,
            "method": "firth",
            "has_sensitivity": True,
            "has_subgroup": True,
        }
        checklist = auto_populate_strobe(meta, study_type="cohort")

        # Verify auto-marks
        # 5. Setting (Partial)
        assert (
            next(i for i in checklist.items if i.number == "5").status
            == ChecklistStatus.PARTIAL
        )
        # 7. Variables (Complete)
        assert (
            next(i for i in checklist.items if i.number == "7").status
            == ChecklistStatus.COMPLETE
        )
        # 12a. Method (Complete, Firth)
        item_12a = next(i for i in checklist.items if i.number == "12a")
        assert item_12a.status == ChecklistStatus.COMPLETE
        assert "Firth" in item_12a.notes
        assert "95% CI" in item_12a.notes
        # 13a. Participants (Partial, n_analyzed < n_total)
        item_13a = next(i for i in checklist.items if i.number == "13a")
        assert item_13a.status == ChecklistStatus.PARTIAL
        assert "Excluded: 10" in item_13a.notes

    def test_strobe_invalid_study_type(self):
        """Test auto_populate_strobe raises ValueError for invalid type."""
        with pytest.raises(ValueError, match="Invalid study_type"):
            auto_populate_strobe({}, study_type="invalid_type")

    def test_auto_populate_no_ci_does_not_mark_16a(self):
        """Verify 16a is not marked complete when has_ci is False."""
        meta = {"n_total": 50, "n_analyzed": 50, "has_ci": False}
        checklist = auto_populate_strobe(meta, study_type="cohort")
        item_16a = next(i for i in checklist.items if i.number == "16a")
        assert item_16a.status == ChecklistStatus.NOT_DONE

    def test_auto_populate_excluded_clamped(self):
        """Verify excluded count is clamped to 0 when n_analyzed > n_total."""
        meta = {"n_total": 50, "n_analyzed": 100}
        checklist = auto_populate_strobe(meta, study_type="cohort")
        item_13a = next(i for i in checklist.items if i.number == "13a")
        assert "Excluded: 0" in item_13a.notes

    def test_format_strobe_html_compact_structure_and_escaping(self):
        """Test HTML formatting and escaping."""
        checklist = create_strobe_checklist()
        # Inject unsafe content
        unsafe_note = "<script>alert('xss')</script>"
        checklist.update_item("1a", ChecklistStatus.COMPLETE, notes=unsafe_note)

        html_out = format_strobe_html_compact(checklist)

        # Check structure
        assert "STROBE Completion:" in html_out
        assert "table" in html_out

        # Check escaping
        assert "<script>" not in html_out
        assert "&lt;script&gt;" in html_out or "&#x3C;script&#x3E;" in html_out


class TestChecklistExport:
    """Tests for checklist export functionality."""

    def test_to_html(self):
        """Test HTML export."""
        checklist = create_consort_checklist()
        html = checklist.to_html()

        assert "<table" in html
        assert "CONSORT 2010" in html
        assert "Title" in html

    def test_to_markdown(self):
        """Test markdown export."""
        checklist = create_consort_checklist()
        md = generate_checklist_markdown(checklist)

        assert "# CONSORT 2010 Checklist" in md
        assert "## Title and Abstract" in md
        assert "❌" in md  # NOT_DONE emoji
