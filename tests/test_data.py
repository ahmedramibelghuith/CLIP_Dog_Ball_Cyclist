import csv
import tempfile
import unittest
from pathlib import Path
from clip_project.data import read_rows, split_rows, write_rows


class DataTests(unittest.TestCase):
    def test_partition_reproducible_disjoint_and_balanced(self):
        rows = [{"image_path": f"{label}/{i}.jpg", "label": label, "caption": "example"}
                for label in ("ball", "dog") for i in range(200)]
        parts = split_rows(rows)
        self.assertEqual(parts, split_rows(rows))
        self.assertEqual([len(parts[k]) for k in ("train", "val", "test")], [320, 40, 40])
        paths = [{r["image_path"] for r in part} for part in parts.values()]
        self.assertFalse(paths[0] & paths[1] or paths[0] & paths[2] or paths[1] & paths[2])

    def test_csv_resolves_paths_and_rejects_duplicate_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "image.jpg").touch()
            row = {"image_path": "image.jpg", "label": "dog", "caption": "a dog"}
            path = root / "captions.csv"
            write_rows(path, [row])
            self.assertEqual(read_rows(path, root)[0]["image_path"], str(root / "image.jpg"))
            write_rows(path, [row, row])
            with self.assertRaises(ValueError):
                read_rows(path, root)


if __name__ == "__main__":
    unittest.main()
