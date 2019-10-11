import spacy_ner


ne = spacy_ner.NERspacy()

data = ne.convert_dataturks_to_spacy(ne.dataturks_JSON_FilePath)

if __name__ == "__main__":
    ne.train_spacy(data)
