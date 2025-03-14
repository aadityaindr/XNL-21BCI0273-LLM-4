import unittest
import os
from utils import extract_text_from_pdf, extract_text_from_image, load_csv_data

class TestCoreFunctions(unittest.TestCase):

    def test_extract_text_from_pdf(self):
        # Assume sample.pdf is in the same directory as this test file.
        pdf_path = os.path.join(os.path.dirname(__file__), "sample.pdf")
        if not os.path.exists(pdf_path):
            self.skipTest("Sample PDF file not found.")
        text = extract_text_from_pdf(pdf_path)
        self.assertTrue(len(text) > 0, "Extracted PDF text should not be empty.")

    def test_extract_text_from_image(self):
        # Assume sample.jpg is in the same directory as this test file.
        img_path = os.path.join(os.path.dirname(__file__), "sample.jpg")
        if not os.path.exists(img_path):
            self.skipTest("Sample image file not found.")
        text = extract_text_from_image(img_path)
        self.assertTrue(len(text) > 0, "Extracted image text should not be empty.")

    def test_csv_loader(self):
        # Assume sample.csv is in the same directory as this test file.
        csv_path = os.path.join(os.path.dirname(__file__), "sample.csv")
        if not os.path.exists(csv_path):
            self.skipTest("Sample CSV file not found.")
        csv_text = load_csv_data(csv_path)
        self.assertTrue(len(csv_text) > 0, "Loaded CSV text should not be empty.")

if __name__ == '__main__':
    unittest.main()
