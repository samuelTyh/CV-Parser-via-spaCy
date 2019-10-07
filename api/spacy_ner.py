import logging
import json
import random
import spacy


class NERspacy(object):

    dataturks_JSON_FilePath = "../dataturks_JSON_FilePath/NERspacy_project.json"

    def convert_dataturks_to_spacy(self):
        try:
            training_data = []
            with open(self.dataturks_JSON_FilePath, 'r') as f:
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
            logging.exception("Unable to process " + self.dataturks_JSON_FilePath + "\n" + "error = " + str(e))
            return None

    @staticmethod
    def train_spacy(train_data):
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
            for itn in range(1):
                print("Statring iteration " + str(itn))
                random.shuffle(train_data)
                losses = {}
                for text, annotations in train_data:
                    nlp.update(
                        [text],  # batch of texts
                        [annotations],  # batch of annotations
                        drop=0.2,  # dropout - make it harder to memorise data
                        sgd=optimizer,  # callable to update weights
                        losses=losses)
                print(losses)
