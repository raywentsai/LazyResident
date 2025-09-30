"""
PDF text extraction module using PyMuPDF (fitz)
For processing historical medical records
"""
import fitz  # PyMuPDF
import logging
from pathlib import Path
from typing import Optional

from .config import TEMP_DIR


logger = logging.getLogger(__name__)


class PDFProcessor:
    """Handles PDF text extraction for historical medical records"""
    
    def __init__(self):
        self.temp_files = []  # Track temporary files for cleanup
    
    def extract_text_from_pdf(self, pdf_path: str) -> Optional[str]:
        """
        Extract text from PDF file with improved text extraction

        Args:
            pdf_path: Path to PDF file

        Returns:
            Extracted text or None if error
        """
        if not Path(pdf_path).exists():
            logger.warning("PDF file not found: %s", pdf_path)
            return None

        try:
            logger.debug("Extracting text from %s", Path(pdf_path).name)

            # Open PDF document
            doc = fitz.open(pdf_path)

            # Extract text from all pages with multiple methods
            text_parts = []
            total_chars = 0

            for page_num in range(len(doc)):
                page = doc.load_page(page_num)

                # Method 1: Standard text extraction
                text = page.get_text()

                # Method 2: If standard extraction yields little text, try text blocks
                if len(text.strip()) < 50:
                    text_blocks = page.get_text("blocks")
                    block_texts = []
                    for block in text_blocks:
                        if len(block) >= 4 and isinstance(block[4], str):
                            block_texts.append(block[4])
                    text = "\n".join(block_texts)

                # Method 3: If still little text, try dictionary extraction
                if len(text.strip()) < 50:
                    text_dict = page.get_text("dict")
                    dict_texts = []
                    for block in text_dict.get("blocks", []):
                        if "lines" in block:
                            for line in block["lines"]:
                                for span in line.get("spans", []):
                                    if "text" in span:
                                        dict_texts.append(span["text"])
                    text = " ".join(dict_texts)

                # Clean and format the text
                text = text.strip()
                if text:
                    # Remove excessive whitespace and normalize line breaks
                    import re
                    text = re.sub(r'\n\s*\n', '\n\n', text)  # Normalize paragraph breaks
                    text = re.sub(r'[ \t]+', ' ', text)      # Normalize spaces

                    text_parts.append(f"--- Page {page_num + 1} ---\n{text}")
                    total_chars += len(text)

                logger.debug("Page %d extracted with %d characters", page_num + 1, len(text))

            doc.close()

            # Combine all pages
            full_text = "\n\n".join(text_parts)

            logger.debug("PDF text extraction completed")
            logger.debug("Total pages: %d", len(text_parts))
            logger.debug("Total characters: %d", total_chars)

            return full_text.strip() if full_text.strip() else None

        except Exception as exc:
            logger.exception("PDF extraction error")
            return None
    
    def extract_text_from_uploaded_file(self, uploaded_file) -> Optional[str]:
        """
        Extract text from Streamlit uploaded file object

        Args:
            uploaded_file: Streamlit UploadedFile object

        Returns:
            Extracted text or None if error
        """
        if uploaded_file is None:
            return None

        try:
            # Create temporary file with safe filename
            safe_filename = "".join(c for c in uploaded_file.name if c.isalnum() or c in (' ', '-', '_', '.')).rstrip()
            temp_path = TEMP_DIR / f"temp_pdf_{safe_filename}"

            # Write uploaded file to temporary location
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Track for cleanup
            self.temp_files.append(temp_path)

            # Extract text
            text = self.extract_text_from_pdf(str(temp_path))

            # Clean up temporary file
            self._cleanup_temp_file(temp_path)

            return text

        except Exception as exc:
            logger.exception("Error processing uploaded PDF %s", uploaded_file.name)
            return None

    def extract_text_from_multiple_files(self, uploaded_files) -> Optional[str]:
        """
        Extract text from multiple Streamlit uploaded file objects

        Args:
            uploaded_files: List of Streamlit UploadedFile objects

        Returns:
            Combined extracted text or None if error
        """
        if not uploaded_files:
            return None

        try:
            all_texts = []

            for uploaded_file in uploaded_files:
                logger.debug("Processing PDF: %s", uploaded_file.name)

                text = self.extract_text_from_uploaded_file(uploaded_file)
                if text:
                    # Add file header and content
                    file_header = f"=== {uploaded_file.name} ==="
                    all_texts.append(f"{file_header}\n{text}")

                    logger.debug("Successfully processed %s (%d characters)", uploaded_file.name, len(text))
                else:
                    logger.warning("Failed to extract text from %s", uploaded_file.name)

            if all_texts:
                # Combine all texts with clear separators
                combined_text = "\n\n".join(all_texts)

                logger.debug("Combined text from %d files (%d total characters)", len(all_texts), len(combined_text))

                return combined_text
            else:
                logger.info("No text extracted from uploaded files")
                return None

        except Exception as exc:
            logger.exception("Error processing multiple PDF files")
            return None
    
    def _cleanup_temp_file(self, file_path: Path):
        """Clean up temporary file"""
        try:
            if file_path.exists():
                file_path.unlink()
                if file_path in self.temp_files:
                    self.temp_files.remove(file_path)
                    
        except Exception as exc:
            logger.warning("Could not clean up temp file %s: %s", file_path, exc)

    def cleanup_all_temp_files(self):
        """Clean up all temporary files"""
        for temp_file in self.temp_files.copy():
            self._cleanup_temp_file(temp_file)
    
    def __del__(self):
        """Cleanup on object destruction"""
        self.cleanup_all_temp_files()