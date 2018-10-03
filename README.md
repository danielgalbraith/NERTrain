# NERTrain

Training Named Entity Recogniser with spaCy in Python 3.

Task to train model to recognise entities 'iPhone XS' as PRODUCT and 'Apple' as ORG. Includes training sentences and test paragraph. Uses standard spaCy English model, i.e. CNN trained on OntoNotes, with GloVe vectors trained on Common Crawl. Command to run training:

`python train.py -m en [-n] [-o]`

Optional arguments for number of iterations e.g. [-n 30] and output directory [-o /path/to/dir].

Requires spaCy 2.0+ and Python 3.0+.
