import spacy

nlp = spacy.load('en_core_web_lg')
doc = nlp(open("pdf_sample/CV_Samuel_Tseng.pdf.txt", "r").read())
output = dict()
for ent in doc.ents:
    output[ent.label_] = []
for ent in doc.ents:
    if ent.text not in output[ent.label_]:
        output[ent.label_].append(ent.text)
    pass
print(output)