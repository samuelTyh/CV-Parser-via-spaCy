import os
import logging
import json
import random
import time
import pickle
import spacy
import numpy as np
import functools
from spacy.gold import GoldParse
from spacy.util import minibatch, compounding
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


def timer(method):
    @functools.wraps(method)
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print(f"Completed in {int(te - ts)} seconds")
        return result
    return timed


def spacy_model_loader(path):
    """
    Helper for loading spacy binary format model
    :param path: string, the path where model stored
    :return: object, spacy model
    """
    with open(path, "rb") as f:
        model = pickle.load(f)
    nlp = spacy.blank(model["lang"])
    for pipe_name in model["pipeline"]:
        pipe = nlp.create_pipe(pipe_name)
        nlp.add_pipe(pipe)
    nlp.from_bytes(model["bytes_data"])

    return nlp


class NERspacy:

    data = os.getcwd() + "/data/NERspacy_project.json"

    def __init__(self, test_size, n_iter, early_stopping, model):
        self.testing_data_proportion = test_size
        self.n_iter = n_iter
        self.not_improve = early_stopping
        self.model = model

    def convert_dataturks_to_spacy(self, filepath):
        random.seed(0)
        try:
            all_data = []
            with open(filepath, 'r') as f:
                lines = f.readlines()

            for line in lines:
                line = line.lower()
                data = json.loads(line)
                text = data['content']
                entities = []
                data_annotations = data['annotation']
                if data_annotations is not None:
                    for annotation in data_annotations:
                        # only a single point in text annotation.
                        point = annotation['points'][0]
                        labels = annotation['label']
                        # handle both list of labels or a single label.
                        if not isinstance(labels, list):
                            labels = [labels]

                        for label in labels:
                            point_start = point['start']
                            point_end = point['end']
                            point_text = point['text']

                            lstrip_diff = len(point_text) - len(point_text.lstrip())
                            rstrip_diff = len(point_text) - len(point_text.rstrip())
                            if lstrip_diff != 0:
                                point_start = point_start + lstrip_diff
                            if rstrip_diff != 0:
                                point_end = point_end - rstrip_diff
                            entities.append((point_start, point_end + 1, label))
                all_data.append((text, {"entities": entities}))
            n_sample = round(self.testing_data_proportion * len(all_data))
            testing_data = random.sample(all_data, n_sample)
            training_data = [data for data in all_data if data not in testing_data]

            return training_data, testing_data

        except Exception as e:
            logging.exception("Unable to process " + filepath + "\n" + "error = " + str(e))
            return None

    @timer
    def train_spacy(self, training_data, testing_data, dropout, display_freq=1, output_dir=None,
                    new_model_name="en_model"):
        # create the built-in pipeline components and add them to the pipeline
        # nlp.create_pipe works for built-ins that are registered with spaCy
        # create blank Language class
        if self.model:
            nlp = spacy.load(self.model)
        else:
            nlp = spacy.blank('en')

        if 'ner' not in nlp.pipe_names:
            ner = nlp.create_pipe('ner')
            nlp.add_pipe(ner, last=True)
        else:
            ner = nlp.get_pipe('ner')

        # add labels
        for _, annotations in training_data:
            for ent in annotations.get('entities'):
                ner.add_label(ent[2])

        # get names of other pipes to disable them during training
        other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
        with nlp.disable_pipes(*other_pipes):  # only train NER
            optimizer = nlp.begin_training()
            print(">>>>>>>>>>  Training the model  <<<<<<<<<<\n")

            losses_best = 100000
            early_stop = 0

            for itn in range(self.n_iter):
                print(f"Starting iteration {itn + 1}")
                random.shuffle(training_data)
                losses = {}
                batches = minibatch(training_data, size=compounding(4., 32., 1.001))
                for batch in batches:
                    text, annotations = zip(*batch)
                    nlp.update(
                        text,  # batch of texts
                        annotations,  # batch of annotations
                        drop=dropout,  # dropout - make it harder to memorise data
                        sgd=optimizer,  # callable to update weights
                        losses=losses)

                if itn % display_freq == 0:
                    print(f"Iteration {itn + 1} Loss: {losses}")

                if losses["ner"] < losses_best:
                    early_stop = 0
                    losses_best = int(losses["ner"])
                else:
                    early_stop += 1

                print(f"Training will stop if the value reached {self.not_improve}, "
                      f"and it's {early_stop} now.\n")

                if early_stop >= self.not_improve:
                    break

            print(">>>>>>>>>>  Finished training  <<<<<<<<<<")

            if output_dir:
                path = output_dir + f"en_model_ner_{round(losses_best, 2)}"
            else:
                path = os.getcwd() + f"/lib/inactive_model/en_model_ner_{round(losses_best, 2)}"
                os.mkdir(path)
            if testing_data:
                self.validate_spacy(model=nlp, data=testing_data)

        with nlp.use_params(optimizer.averages):
            nlp.meta["name"] = new_model_name
            bytes_data = nlp.to_bytes()
            lang = nlp.meta["lang"]
            pipeline = nlp.meta["pipeline"]

        model_data = dict(bytes_data=bytes_data, lang=lang, pipeline=pipeline)

        with open(path + '/model.pkl', 'wb') as f:
            pickle.dump(model_data, f)

        return path

    @staticmethod
    def validate_spacy(model, data, n_resume=0, evaluation_set=None):

        # test the model and evaluate it
        for text, annot in data:
            f = open(f"test_outcome/resume{n_resume}.txt", "w")
            doc_to_test = model(text)
            d = {}
            for ent in doc_to_test.ents:
                d[ent.label_] = []
            for ent in doc_to_test.ents:
                d[ent.label_].append(ent.text)

            for i in set(d.keys()):

                f.write("\n\n")
                f.write(i + ":" + "\n")
                for j in set(d[i]):
                    f.write(j.replace('\n', '') + "\n")
            evaluation_set = {}
            for ent in doc_to_test.ents:
                evaluation_set[ent.label_] = [0, 0, 0, 0, 0, 0]
            for ent in doc_to_test.ents:
                doc_gold_text = model.make_doc(text)
                gold = GoldParse(doc_gold_text, entities=annot.get("entities"))
                y_true = [ent.label_ if ent.label_ in x else 'Not ' + ent.label_ for x in gold.ner]
                y_pred = [x.ent_type_ if x.ent_type_ == ent.label_ else 'Not ' + ent.label_ for x in doc_to_test]
                if evaluation_set[ent.label_][0] == 0:
                    precision, recall, fscore, support = precision_recall_fscore_support(
                        y_true, y_pred, average='weighted', labels=np.unique(y_pred))
                    accuracy = accuracy_score(y_true, y_pred)
                    evaluation_set[ent.label_][0] = 1
                    evaluation_set[ent.label_][1] += precision
                    evaluation_set[ent.label_][2] += recall
                    evaluation_set[ent.label_][3] += fscore
                    evaluation_set[ent.label_][4] += accuracy
                    evaluation_set[ent.label_][5] += 1
            n_resume += 1
        with open("test_outcome/evaluation_report.txt", 'w') as f:
            f.writelines(f"Testing data size: {n_resume}\n\n")
            for name in evaluation_set:
                f.writelines(f"\nFor Entity {name}\n")
                f.writelines(
                    f"Accuracy : {round((evaluation_set[name][4] / evaluation_set[name][5]) * 100, 4)}%\n"
                )
                f.writelines(f"Precision : {round(evaluation_set[name][1] / evaluation_set[name][5], 4)}\n")
                f.writelines(f"Recall : {round(evaluation_set[name][2] / evaluation_set[name][5], 4)}\n")
                f.writelines(f"F-score : {round(evaluation_set[name][3] / evaluation_set[name][5], 4)}\n\n")
