"""
Unit tests for utils/reporting_checklists.py

Tests CONSORT and STROBE checklist creation and validation.
"""

from utils.reporting_checklists import (
    ChecklistItem,
    ChecklistStatus,
    create_consort_checklist,
    create_strobe_checklist,
    generate_checklist_markdown,
)


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
