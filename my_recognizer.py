import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    
    for i in test_set.get_all_Xlengths().items():
        x, lens = test_set.get_item_Xlengths(i[0])
        # Create a dict where key = word, value = log liklihood
        word_liklihoods = {}
        for word, model in models.items():
            try:
                word_liklihoods[word] = model.score(x, lens)
            except:
                word_liklihoods[word] = float('-inf')
        probabilities.append(word_liklihoods)
        guesses.append(max(word_liklihoods, key=word_liklihoods.get))
    # TODO implement the recognizer
    return probabilities, guesses
    
