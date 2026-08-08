import unittest

from rag_pdf_ingest.bibliography import format_pdf_citation, resolve_bibliography


class BibliographyTests(unittest.TestCase):
    def test_sidecar_identity_wins_over_source_filename(self):
        bibliography = resolve_bibliography(
            {
                "bibliography": {
                    "title": "Starting Strength",
                    "authors": ["Mark Rippetoe"],
                    "edition": "3rd",
                    "publication_year": 2011,
                    "language": "en",
                }
            },
            "Mark Rippetoe - Starting Strength, 3rd edition - Copy.pdf",
        )
        self.assertEqual(bibliography["title"], "Starting Strength")
        self.assertEqual(bibliography["title_source"], "sidecar")
        self.assertEqual(bibliography["publication_year"], "2011")
        self.assertEqual(
            format_pdf_citation(bibliography, 142),
            "Mark Rippetoe, Starting Strength, 3rd ed., 2011, PDF p. 142",
        )

    def test_source_filename_stem_is_explicit_fallback(self):
        bibliography = resolve_bibliography({}, "Useful Book - Copy.pdf")
        self.assertEqual(bibliography["title"], "Useful Book - Copy")
        self.assertEqual(bibliography["title_source"], "source_file")
        self.assertEqual(bibliography["authors"], [])
        self.assertEqual(
            format_pdf_citation(bibliography, 4, 6),
            "Useful Book - Copy, PDF pp. 4-6",
        )

    def test_pdf_metadata_precedes_filename_and_printed_label_is_supplementary(self):
        bibliography = resolve_bibliography(
            {},
            "Book - Copy.pdf",
            pdf_metadata={"title": "Canonical PDF Title", "author": "Ada Author"},
        )
        self.assertEqual(bibliography["title"], "Canonical PDF Title")
        self.assertEqual(bibliography["title_source"], "pdf_metadata")
        self.assertEqual(bibliography["authors"], ["Ada Author"])
        self.assertEqual(bibliography["authors_source"], "pdf_metadata")
        self.assertEqual(
            format_pdf_citation(
                bibliography, 18, printed_page_start="3"
            ),
            "Ada Author, Canonical PDF Title, PDF p. 18 (printed p. 3)",
        )


if __name__ == "__main__":
    unittest.main()
