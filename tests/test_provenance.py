import json
import tempfile
import unittest
from pathlib import Path

from rag_pdf_ingest.provenance import load_license_metadata


class ProvenanceTests(unittest.TestCase):
    def test_missing_sidecar_defaults_to_authorized(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            pdf = Path(temp_dir) / "Book.pdf"
            pdf.write_bytes(b"pdf")
            license_info, provenance, sidecar = load_license_metadata(pdf)
            self.assertEqual(license_info, {"status": "authorized"})
            self.assertIsNone(provenance["sidecar_sha256"])
            self.assertIsNone(sidecar)

    def test_license_sidecar_is_loaded_and_hashed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pdf = root / "Book.pdf"
            pdf.write_bytes(b"pdf")
            sidecar = root / "Book.license.json"
            sidecar.write_text(
                json.dumps({"status": "licensed", "scope": "private-rag"}),
                encoding="utf-8",
            )
            license_info, provenance, selected = load_license_metadata(pdf)
            self.assertEqual(license_info["status"], "licensed")
            self.assertEqual(license_info["scope"], "private-rag")
            self.assertEqual(provenance["sidecar_file"], sidecar.name)
            self.assertEqual(len(provenance["sidecar_sha256"]), 64)
            self.assertEqual(selected, sidecar)


if __name__ == "__main__":
    unittest.main()
