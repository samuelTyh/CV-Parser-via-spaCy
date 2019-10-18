import argparse
import spacy_ner
import pdf_extractor
import config

parser = argparse.ArgumentParser()
parser.add_argument("-M", "--model", help="filepath to store model")
parser.add_argument("-f", "--file", required=True, help="Original CV filepath")
argument = parser.parse_args()


def cvparser(test_size, n_iter, early_stopping):

    # Invoke class
    ne = spacy_ner.NERspacy(test_size, n_iter, early_stopping)

    # Get training data and testing data
    train, test = ne.convert_dataturks_to_spacy(ne.data)

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
    cvparser(test_size=config.test_size, n_iter=config.n_iter, early_stopping=config.early_stopping)
