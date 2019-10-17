import argparse
import spacy_ner
import pdf_extractor

parser = argparse.ArgumentParser()
parser.add_argument("-M", "--model", help="filepath to store model")
parser.add_argument("-f", "--file", required=True, help="Original CV filepath")
argument = parser.parse_args()


def cvparser():

    # Invoke class
    ne = spacy_ner.NERspacy()

    # Get training data and testing data
    train, test = ne.convert_dataturks_to_spacy(ne.dataturks_JSON_FilePath)

    # Parse content from original CV
    cv_filepath = argument.file
    content = pdf_extractor.extract_pdf_content(cv_filepath)

    with open(argument.file + '.txt', 'w') as f:
        f.write(content.replace('\n', ''))

    if argument.model:
        model_filepath = argument.model
    else:
        model_filepath = ne.train_spacy(train, test)

    return spacy_ner.predict_spacy(content, model_filepath)


if __name__ == "__main__":
    cvparser()
