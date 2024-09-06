import fitz
import os
import easyocr

class ScannedDataExtractor():
    
    def extractImagesWithFitz(self, pdf_files, output_path, path_to_scanned_pdfs):
        print(pdf_files)
        scannedText = []
        for file in pdf_files:
            print(f"File Name : {file}")
            file_path = path_to_scanned_pdfs+file
            if os.path.isfile(file_path):
                doc = fitz.open(file_path)
                zoom = 4
                mat = fitz.Matrix(zoom, zoom)
                count = 0
                # Count variable is to get the number of pages in the pdf
                for p in doc:
                    count += 1
                for i in range(count):
                    val = f"image_{i+1}.png"
                    page = doc.load_page(i)
                    pix = page.get_pixmap(matrix=mat)
                    img_file = os.path.join(output_path,val)
                    pix.save(img_file)
                    scannedText.append(self.readScan(img_file))
                doc.close()
        print(scannedText[7])
        textscans = ' '.join(str(t) for t in scannedText)
        return textscans

    def extractImagesWithPdfImages(self, pdf_file, output_path):
        print(pdf_file)
        print(output_path)
        os.system("pdfimages -j '{0}' '{1}' ".format(pdf_file, output_path))

    def readScan(self, img_file):
        reader = easyocr.Reader(['en'])
        result = reader.readtext(img_file, detail=0)
        return result

    def cleanUpImages(self, output_path):
        os.system(f" rm -r '{output_path}*'")

if __name__ == "__main__":
    pdf_file = 'Scanned PDF OCR/careerguidancebook-_part1.pdf'
    output_path1 = 'Scanned PDF OCR/scans1/'
    output_path2 = 'Scanned PDF OCR/scans2/'
    file_prefix = "image"
    extractor = ScannedDataExtractor()
    extractor.extractImagesWithFitz(pdf_file, output_path1)
    extractor.readScan(output_path1+"image_7.png")
    extractor.cleanUpImages(output_path1)
    # extractImagesWithPdfImages(pdf_file, output_path2+file_prefix)
    # readScan(output_path2+"-006.jpg")
    # cleanUpImages(output_path2)