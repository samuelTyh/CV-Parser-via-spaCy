from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from io import StringIO
import sys


def extract_pdf_content(file):
    rsrcmgr = PDFResourceManager()
    codec = 'utf-8'
    outfp = StringIO()
    laparams = LAParams()
    device = TextConverter(rsrcmgr=rsrcmgr, outfp=outfp, codec=codec, laparams=laparams)
    with open(file, 'rb') as fp:
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        password = None
        maxpages = 0
        caching = True
        pagenos = set()
        for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages, password=password, caching=caching,
                                      check_extractable=True):
            interpreter.process_page(page)
    mystr = outfp.getvalue()
    device.close()
    outfp.close()
    return mystr


with open(sys.argv[1]+'.txt', 'w') as f:
    f.writelines(extract_pdf_content(sys.argv[1]))
