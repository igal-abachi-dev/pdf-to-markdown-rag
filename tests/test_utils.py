import unittest

from rag_pdf_ingest.utils import safe_slug, strip_markdown_fence, without_front_matter


class UtilsTests(unittest.TestCase):
    def test_safe_slug_preserves_hebrew(self):
        self.assertEqual(safe_slug("  ספר תזונה / 2026  "), "ספר-תזונה-2026")

    def test_strip_markdown_fence(self):
        self.assertEqual(strip_markdown_fence("```markdown\n# Title\n```"), "# Title")

    def test_without_front_matter(self):
        value = "---\npage: 1\n---\n\n# Heading\n"
        self.assertEqual(without_front_matter(value), "\n# Heading\n")


if __name__ == "__main__":
    unittest.main()

