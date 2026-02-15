"""Unit tests for utils/download_helpers.py."""

from unittest.mock import patch

from utils.download_helpers import (
    _build_error_html,
    safe_download_html,
    safe_report_generation,
)


# ---------------------------------------------------------------------------
# _build_error_html
# ---------------------------------------------------------------------------
class TestBuildErrorHtml:
    def test_returns_valid_html(self):
        html = _build_error_html("Test Title", "Test message")
        assert "<!DOCTYPE html>" in html
        assert "<html>" in html
        assert "</html>" in html

    def test_escapes_title(self):
        html = _build_error_html("<script>alert(1)</script>", "msg")
        assert "<script>" not in html
        assert "&lt;script&gt;" in html

    def test_escapes_message(self):
        html = _build_error_html("title", '<img src="x" onerror="alert(1)">')
        assert 'onerror="alert(1)"' not in html
        assert "&lt;img" in html

    def test_contains_title_and_message(self):
        html = _build_error_html("No Data", "Run analysis first")
        assert "No Data" in html
        assert "Run analysis first" in html


# ---------------------------------------------------------------------------
# safe_download_html
# ---------------------------------------------------------------------------
class TestSafeDownloadHtml:
    def test_valid_content_returned_unchanged(self):
        content = "<!DOCTYPE html><html><body>Report</body></html>"
        result = safe_download_html(content, label="Test")
        assert result == content

    def test_none_content_returns_error_page(self):
        result = safe_download_html(None, label="ROC Report")
        assert "<!DOCTYPE html>" in result
        assert "No Results" in result
        assert "ROC Report" in result

    def test_empty_string_returns_error_page(self):
        result = safe_download_html("", label="Report")
        assert "<!DOCTYPE html>" in result
        assert "No Results" in result

    def test_whitespace_only_returns_error_page(self):
        result = safe_download_html("   \n\t  ", label="Report")
        assert "<!DOCTYPE html>" in result
        assert "No Results" in result

    def test_error_page_contains_doctype(self):
        result = safe_download_html(None, label="X")
        assert result.startswith("<!DOCTYPE html>")

    def test_default_label(self):
        result = safe_download_html(None)
        assert "Report" in result


# ---------------------------------------------------------------------------
# safe_report_generation
# ---------------------------------------------------------------------------
class TestSafeReportGeneration:
    def test_success_returns_content(self):
        content = "<!DOCTYPE html><html><body>OK</body></html>"
        result = safe_report_generation(lambda: content, label="X")
        assert result == content

    def test_exception_returns_error_page(self):
        def _boom():
            raise ValueError("fail")

        result = safe_report_generation(_boom, label="Boom")
        assert "<!DOCTYPE html>" in result
        assert "Generation Failed" in result

    def test_none_returns_error_page(self):
        result = safe_report_generation(lambda: None, label="Empty")
        assert "<!DOCTYPE html>" in result
        assert "No Results" in result

    def test_empty_string_returns_error_page(self):
        result = safe_report_generation(lambda: "", label="Blank")
        assert "<!DOCTYPE html>" in result
        assert "No Results" in result

    def test_default_label(self):
        result = safe_report_generation(lambda: None)
        assert "Report" in result

    def test_exception_notifies_error(self):
        def _boom():
            raise ValueError("fail")

        with patch("utils.download_helpers._notify") as mock_notify:
            result = safe_report_generation(_boom, label="Boom")
            assert "<!DOCTYPE html>" in result
            assert "Generation Failed" in result
            mock_notify.assert_called_once()
            assert mock_notify.call_args[1]["type"] == "error"
