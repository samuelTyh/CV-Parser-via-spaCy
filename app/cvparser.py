import os
import re
import requests

from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from io import StringIO, BytesIO

from .config import ModelConfig
from .ner_trainer import spacy_model_loader


class CVParser:
    def __init__(self, model=None):
        if not model:
            self.ner_model_path = os.path.join(os.getcwd(), ModelConfig.MODEL_PATH)
        else:
            self.ner_model_path = model

    @staticmethod
    def parse_from_file(file):
        """
        Parse file on development env
        :param file: local file path
        :return: string, The file's content in string format
        """
        content_parsed = requests.put(ModelConfig.TIKA_URL, file)
        text = content_parsed['content']
        content = text.encode("ascii", "ignore").decode("utf-8")
        output = re.sub(r"[\n\r\s]+", " ", content)
        return output

    @staticmethod
    def parse_from_url(url):
        pdf = requests.get(url, stream=True)
        content_parsed = requests.put(ModelConfig.TIKA_URL, pdf)
        text = content_parsed.text
        content = text.encode("ascii", "ignore").decode("utf-8")
        output = re.sub(r"[\n\r\s]+", " ", content)
        return output

    def predict_name_entities(self, content):
        """
        Execute NER prediction by pre-trained model

        :param content: string, extracted contents from pdf_extractor
        :return: object, customized contents, phrases, words' classification of CV
        """
        try:
            nlp = spacy_model_loader(os.path.join(self.ner_model_path, "model.pkl"))
        except Exception as e:
            return dict(error_message=f"NER model loading error: {e}")

        content = content.lower()
        doc = nlp(content)
        output = dict()
        for ent in doc.ents:
            output[ent.label_] = []
        for ent in doc.ents:
            if ent.text not in output[ent.label_]:
                output[ent.label_].append(ent.text)
            pass
        return output

    def parse_cv(self, cv_path):
        """
        Execute pdf_extractor and grab the result to extract talents information

        :return: object, customized contents, phrases, words' classification of CV
        """
        if os.environ['FLASK_ENV'] == 'development':
            content = self.parse_from_url(cv_path)
        else:
            content = demo_pdf_parser(cv_path)
        return self.predict_name_entities(content)


def demo_pdf_parser(url):
    rsrcmgr = PDFResourceManager()
    codec = 'utf-8'
    outfp = StringIO()
    laparams = LAParams()
    device = TextConverter(rsrcmgr=rsrcmgr, outfp=outfp, codec=codec, laparams=laparams)
    f = requests.get(url, stream=True).content
    fp = BytesIO(f)
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    password = ""
    maxpages = 0
    caching = True
    pagenos = set()
    for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages, password=password, caching=caching,
                                  check_extractable=True):
        interpreter.process_page(page)
    fp.close()
    text = outfp.getvalue().encode("ascii", "ignore").decode("utf-8")
    output = re.sub(r"[\n\r\s]+", " ", text)
    device.close()
    outfp.close()
    return output
