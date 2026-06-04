"""Unit tests for utils/pdf_helpers.py."""

from unittest.mock import MagicMock, patch

from utils.pdf_helpers import (
    safe_download_pdf,
    safe_pdf_report_generation,
)


# ---------------------------------------------------------------------------
# safe_download_pdf
# ---------------------------------------------------------------------------
class TestSafeDownloadPdf:
    @patch("utils.pdf_helpers.html_to_pdf")
    def test_valid_content_returns_pdf_bytes(self, mock_pdf):
        mock_pdf.return_value = b"%PDF-fake"
        content = "<!DOCTYPE html><html><body>Report</body></html>"
        result = safe_download_pdf(content, label="Test")
        assert result == b"%PDF-fake"
        mock_pdf.assert_called_once_with(content)

    @patch("utils.pdf_helpers.html_to_pdf")
    def test_none_content_returns_error_pdf(self, mock_pdf):
        mock_pdf.return_value = b"%PDF-error"
        result = safe_download_pdf(None, label="ROC Report")
        assert result == b"%PDF-error"
        # Should have been called with an error HTML page
        call_args = mock_pdf.call_args[0][0]
        assert "No Results" in call_args or "No results" in call_args.lower()

    @patch("utils.pdf_helpers.html_to_pdf")
    def test_empty_string_returns_error_pdf(self, mock_pdf):
        mock_pdf.return_value = b"%PDF-error"
        result = safe_download_pdf("", label="Report")
        assert isinstance(result, bytes)

    @patch("utils.pdf_helpers.html_to_pdf")
    def test_whitespace_only_returns_error_pdf(self, mock_pdf):
        mock_pdf.return_value = b"%PDF-error"
        result = safe_download_pdf("   \n\t  ", label="Report")
        assert isinstance(result, bytes)

    @patch("utils.pdf_helpers.html_to_pdf", side_effect=RuntimeError("No browser"))
    def test_conversion_failure_returns_fallback(self, mock_pdf):
        content = "<!DOCTYPE html><html><body>OK</body></html>"
        result = safe_download_pdf(content, label="Report")
        # Should return something (error HTML bytes as fallback)
        assert isinstance(result, bytes)


# ---------------------------------------------------------------------------
# safe_pdf_report_generation
# ---------------------------------------------------------------------------
class TestSafePdfReportGeneration:
    @patch("utils.pdf_helpers.html_to_pdf")
    def test_success_returns_pdf(self, mock_pdf):
        mock_pdf.return_value = b"%PDF-ok"
        content = "<!DOCTYPE html><html><body>OK</body></html>"
        result = safe_pdf_report_generation(lambda: content, label="X")
        assert result == b"%PDF-ok"

    @patch("utils.pdf_helpers.html_to_pdf")
    def test_exception_returns_error_pdf(self, mock_pdf):
        mock_pdf.return_value = b"%PDF-error"

        def _boom():
            raise ValueError("fail")

        result = safe_pdf_report_generation(_boom, label="Boom")
        assert isinstance(result, bytes)

    @patch("utils.pdf_helpers.html_to_pdf")
    def test_none_result_returns_error_pdf(self, mock_pdf):
        mock_pdf.return_value = b"%PDF-error"
        result = safe_pdf_report_generation(lambda: None, label="Empty")
        assert isinstance(result, bytes)

    @patch("utils.pdf_helpers.html_to_pdf")
    def test_default_label(self, mock_pdf):
        mock_pdf.return_value = b"%PDF-error"
        result = safe_pdf_report_generation(lambda: None)
        assert isinstance(result, bytes)
