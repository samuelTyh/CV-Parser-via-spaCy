import json
import argparse
from app import ner_trainer, pdf_extractor, config

parser = argparse.ArgumentParser()
parser.add_argument("-M", "--model", help="filepath to store model")
parser.add_argument("-f", "--file", required=True, help="Original CV filepath")
argument = parser.parse_args()


def cvparser(test_size, n_iter, early_stopping):

    # Invoke class
    ne = ner_trainer.NERspacy(test_size, n_iter, early_stopping)

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

    output = ner_trainer.predict_spacy(content, model_filepath)

    with open("prediction/ner_prediction.json", "w") as f:
        json.dump(output, f)
    return output


if __name__ == "__main__":
    cvparser(
        test_size=config.HyperParameter.test_size,
        n_iter=config.HyperParameter.n_iter,
        early_stopping=config.HyperParameter.early_stopping
    )
