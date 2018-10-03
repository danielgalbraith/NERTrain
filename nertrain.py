#!/usr/bin/env python
# coding: utf8
from __future__ import unicode_literals, print_function

import plac
import random
from pathlib import Path
import spacy


# training data
TRAIN_DATA = [
    ("DxOMark is out with its full review of the iPhone XS camera system. Coming in with a score of 105, the latest flagship iPhone beat out the camera performance of all other smartphones except for Huawei’s P20 Pro with its triple-camera setup.", {
        'entities': [(0, 6, 'ORG'),(43, 51, 'PRODUCT'),(94, 96, 'CARDINAL'),(119, 124, 'PRODUCT'),(194, 199, 'ORG'),(203, 209, 'PRODUCT')]
    }),
    ("The iPhone XR isn't a handset using 'last year's technology' -- the bulk of the key chip and camera specs appear to be identical to the step-up iPhone XS models.", {
        'entities': [(5, 13, 'PRODUCT'),(145, 153, 'PRODUCT')]
    })
]


@plac.annotations(
    model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    output_dir=("Optional output directory", "option", "o", Path),
    n_iter=("Number of training iterations", "option", "n", int))
def main(model=None, output_dir=None, n_iter=100):
    """Load the model, set up the pipeline and train the entity recognizer."""
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank('en')  # create blank Language class
        print("Created blank 'en' model")

    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner, last=True)
    # otherwise, get it so we can add labels
    else:
        ner = nlp.get_pipe('ner')

    # add labels
    for _, annotations in TRAIN_DATA:
        for ent in annotations.get('entities'):
            ner.add_label(ent[2])

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        optimizer = nlp.begin_training()
        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA)
            losses = {}
            for text, annotations in TRAIN_DATA:
                nlp.update(
                    [text],  # batch of texts
                    [annotations],  # batch of annotations
                    drop=0.5,  # dropout - make it harder to memorise data
                    sgd=optimizer,  # callable to update weights
                    losses=losses)
            print(losses)

    # test the trained model
    test_text = "It’s not all good news for the XS’ screen, though. The notch has persisted from the X and like its forbear, the XS fails to make the most of the extra screen real estate. It’ll display the time, whether your location is being used, your signal, whether you’re connected to wifi, plus the battery icon (though not battery percentage). Apple has also decided to bump up the size of these icons compared to older 4 and 4.7-inch phones – so there’s actually less room to display information than older phones. Like we said though, this is nit-picking. The iPhone XS is a great-looking phone, crafted from high-quality materials with an excellent screen. It may be familiar, but it’s still exceptional."
    doc = nlp(test_text)
    print('Entities', [(ent.text, ent.label_) for ent in doc.ents])
    
    # save model to output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        # test the saved model
        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)
        for text, _ in TRAIN_DATA:
            doc = nlp2(text)
            print('Entities', [(ent.text, ent.label_) for ent in doc.ents])
            print('Tokens', [(t.text, t.ent_type_, t.ent_iob) for t in doc])


if __name__ == '__main__':
    plac.call(main)
