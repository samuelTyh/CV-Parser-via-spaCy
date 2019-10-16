import os
import sys
import spacy_ner
import pdf_extractor

ne = spacy_ner.NERspacy()
data = ne.convert_dataturks_to_spacy(ne.dataturks_JSON_FilePath)

content = pdf_extractor.extract_pdf_content(sys.argv[1])
with open(sys.argv[1] + '.txt', 'w') as f:
    f.writelines(content)

if __name__ == "__main__":
    if "model" not in os.listdir(os.getcwd()+"/api"):
        ne.train_spacy(data)
        spacy_ner.predict_spacy(content)
    else:
        spacy_ner.predict_spacy(content)
