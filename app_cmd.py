import os
import json
import argparse
from app import ner_trainer, config, CVParser


parser = argparse.ArgumentParser()
parser.add_argument("-M", "--model", help="The file's path where the model stored")
parser.add_argument("-f", "--file", required=True, help="Original CV file path")
argument = parser.parse_args()


def parser_trainer(test_size, n_iter, early_stopping, model, dropout):

    # Invoke class
    ne = ner_trainer.NERspacy(test_size, n_iter, early_stopping, model)

    # Get training data and testing data
    train, test = ne.convert_dataturks_to_spacy(ne.data)

    # Invoke model
    if argument.model:
        model_filepath = argument.model
    else:
        model_filepath = ne.train_spacy(train, test, dropout)
    cvparser = CVParser(model=model_filepath)

    # Parse content from original CV
    file_path = argument.file
    content = cvparser.parse_from_file(file_path)

    with open(argument.file + '.txt', 'w') as f:
        f.write(content.replace('\n', '.'))

    output = cvparser.predict_name_entities(content)

    if 'prediction' not in os.listdir('.'):
        os.mkdir('prediction')

    with open("prediction/ner_prediction.json", "w") as f:
        json.dump(output, f)

    return output


parser_trainer(
    test_size=config.SpacyTraining.test_size,
    n_iter=config.SpacyTraining.n_iter,
    early_stopping=config.SpacyTraining.early_stopping,
    model=config.SpacyTraining.model,
    dropout=config.SpacyTraining.dropout,
)
if __name__ == "__main__":
    pass
