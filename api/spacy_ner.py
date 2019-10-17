import os
import logging
import json
import random
import time
import spacy
import numpy as np
from spacy.gold import GoldParse
from spacy.util import minibatch, compounding
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score


def timer(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print("Completed in {} seconds".format(int(te - ts)))
        return result
    return timed


class NERspacy(object):

    dataturks_JSON_FilePath = os.getcwd() + "/dataturks_JSON_FilePath/NERspacy_project.json"
    test_JSON_FilePath = os.getcwd() + "/dataturks_JSON_FilePath/NERspacy_project_test.json"
    n_iter = 60
    not_improve = 10

    @staticmethod
    def convert_dataturks_to_spacy(filepath):
        try:
            training_data = []
            with open(filepath, 'r') as f:
                lines = f.readlines()

            for line in lines:
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
                training_data.append((text, {"entities": entities}))
            return training_data
        except Exception as e:
            logging.exception("Unable to process " + filepath + "\n" + "error = " + str(e))
            return None

    @timer
    def train_spacy(self, train_data, dropout=0.2, display_freq=1):
        nlp = spacy.blank('en')  # create blank Language class
        # create the built-in pipeline components and add them to the pipeline
        # nlp.create_pipe works for built-ins that are registered with spaCy
        if 'ner' not in nlp.pipe_names:
            ner = nlp.create_pipe('ner')
            nlp.add_pipe(ner, last=True)

        # add labels
        for _, annotations in train_data:
            for ent in annotations.get('entities'):
                ner.add_label(ent[2])

        # get names of other pipes to disable them during training
        other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
        with nlp.disable_pipes(*other_pipes):  # only train NER
            optimizer = nlp.begin_training()
            print(">>>>>>>>>>  Training the model  <<<<<<<<<<")

            losses_best = 100000
            early_stop = 0

            for itn in range(self.n_iter):
                print("Starting iteration {}".format(itn + 1))
                random.shuffle(train_data)
                losses = {}
                batches = minibatch(train_data, size=compounding(4., 16., 1.001))
                for batch in batches:
                    text, annotations = zip(*batch)
                    nlp.update(
                        text,  # batch of texts
                        annotations,  # batch of annotations
                        drop=dropout,  # dropout - make it harder to memorise data
                        sgd=optimizer,  # callable to update weights
                        losses=losses)

                if itn % display_freq == 0:
                    print("Iteration {} Loss: {}".format(itn + 1, losses))

                if losses["ner"] < losses_best:
                    early_stop = 0
                    losses_best = int(losses["ner"])
                    with nlp.use_params(optimizer.averages):
                        nlp.to_disk(os.getcwd()+"/api/model")
                else:
                    early_stop += 1

                print("Training will stop early if value reached {not_improve}, "
                      "it's {early_stop} now.".format(not_improve=self.not_improve, early_stop=early_stop))

                if early_stop >= self.not_improve:
                    print(">>>>>>>>>>  Finished training  <<<<<<<<<<")
                    break
                if itn == self.n_iter:
                    print(">>>>>>>>>>  Finished training  <<<<<<<<<<")

        # test the model and evaluate it
        examples = self.convert_dataturks_to_spacy(self.test_JSON_FilePath)
        c = 0
        for text, annot in examples:

            f = open("resumesample/resume" + str(c) + ".txt", "w")
            doc_to_test = nlp(text)
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
            d = {}
            for ent in doc_to_test.ents:
                d[ent.label_] = [0, 0, 0, 0, 0, 0]
            for ent in doc_to_test.ents:
                doc_gold_text = nlp.make_doc(text)
                gold = GoldParse(doc_gold_text, entities=annot.get("entities"))
                y_true = [ent.label_ if ent.label_ in x else 'Not ' + ent.label_ for x in gold.ner]
                y_pred = [x.ent_type_ if x.ent_type_ == ent.label_ else 'Not ' + ent.label_ for x in doc_to_test]
                if d[ent.label_][0] == 0:
                    # f.write("For Entity "+ent.label_+"\n")
                    # f.write(classificatio n_report(y_true, y_pred)+"\n")
                    (p, r, f, s) = precision_recall_fscore_support(
                        y_true, y_pred, average='weighted', labels=np.unique(y_pred))
                    a = accuracy_score(y_true, y_pred)
                    d[ent.label_][0] = 1
                    d[ent.label_][1] += p
                    d[ent.label_][2] += r
                    d[ent.label_][3] += f
                    d[ent.label_][4] += a
                    d[ent.label_][5] += 1
            c += 1
        for i in d:
            print("\n For Entity " + i + "\n")
            print("Accuracy : " + str((d[i][4] / d[i][5]) * 100) + "%")
            print("Precision : " + str(d[i][1] / d[i][5]))
            print("Recall : " + str(d[i][2] / d[i][5]))
            print("F-score : " + str(d[i][3] / d[i][5]))


def predict_spacy(content):
    nlp = spacy.load(os.getcwd() + "/api/model")
    doc = nlp(content)
    output = dict()
    for ent in doc.ents:
        output[ent.label_] = []
    for ent in doc.ents:
        if ent.text not in output[ent.label_]:
            output[ent.label_].append(ent.text)
        pass
    print(output)

    with open("prediction/ner_prediction.json", "w") as f:
        json.dump(output, f)
