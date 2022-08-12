""" Contains the part of speech tagger class. """
import pandas as pd


def load_data(sentence_file, tag_file=None):
    """Loads data from two files: one containing sentences and one containing tags.

    tag_file is optional, so this function can be used to load the test data.

    Suggested to split the data by the document-start symbol.

    """

    sentence_data = pd.read_csv(sentence_file, header=0, usecols=['word'])['word'].values.tolist()

    if tag_file:

        tag_data = pd.read_csv(tag_file, header=0, usecols=['tag'])['tag'].values.tolist()
        return sentence_data, tag_data

    return sentence_data


def evaluate(data, model):
    """Evaluates the POS model on some sentences and gold tags.

    This model can compute a few different accuracies:
        - whole-sentence accuracy
        - per-token accuracy
        - compare the probabilities computed by different styles of decoding

    You might want to refactor this into several different evaluation functions.
    
    """

    pass


class POSTagger:

    trigram_probabilities = {}
    prior_two_count = {}

    def __init__(self):
        """Initializes the tagger model parameters and anything else necessary. """
        pass

    @staticmethod
    def get_lists_with_start_tags(sentences, pos_list):

        x = [1, 2, 3, 1, 2, 3, 1]
        new_sentences_list = []
        new_pos_list = []

        [(new_sentences_list.extend([word]), new_pos_list.extend([tag])) if word != '-DOCSTART-'
         else (new_sentences_list.extend(['<START>', '<START>', '-DOCSTART-']),
               new_pos_list.extend(['N/A', 'N/A', tag]))
         for word, tag in zip(sentences, pos_list)]

        return new_sentences_list, new_pos_list

    def train(self, data):
        """Trains the model by computing transition and emission probabilities.

        You should also experiment:
            - smoothing.
            - N-gram models with varying N.
        
        """

        sentences, pos_list = POSTagger.get_lists_with_start_tags(data[0], data[1])

        for index in range(len(sentences)-3):

            prior_two_words = tuple(sentences[index:index+2])
            trigram = tuple(sentences[index:index+3])

            # increment total number of occurences of prior two words
            if prior_two_words in self.prior_two_count:
                self.prior_two_count[prior_two_words] += 1.0
            else:
                self.prior_two_count[prior_two_words] = 1.0

            # increment number of occurences of word after prior two
            if trigram in self.trigram_probabilities:
                word_pos = pos_list[index+3]
                if word_pos in self.trigram_probabilities:
                    self.trigram_probabilities[word_pos] += 1.0
                else:
                    self.trigram_probabilities[word_pos] = 1.0

    def sequence_probability(self, sequence, tags):
        """Computes the probability of a tagged sequence given the emission/transition
        probabilities.
        """
        return 0

    def inference(self, sequence):
        """Tags a sequence with part of speech tags.

        You should implement different kinds of inference (suggested as separate
        methods):

            - greedy decoding
            - decoding with beam search
            - viterbi
        """
        return []


if __name__ == "__main__":

    pos_tagger = POSTagger()

    train_data = load_data("data/train_x.csv", "data/train_y.csv")

    dev_data = load_data("data/dev_x.csv", "data/dev_y.csv")
    test_data = load_data("data/test_x.csv")

    pos_tagger.train(train_data)

    # # Experiment with your decoder using greedy decoding, beam search, viterbi...
    #
    # # Here you can also implement experiments that compare different styles of decoding,
    # # smoothing, n-grams, etc.
    # evaluate(dev_data, pos_tagger)

    # # Predict tags for the test set
    # test_predictions = []
    # for sentence in test_data:
    #     test_predictions.extend(pos_tagger.inference(sentence)
    #
    # # Write them to a file to update the leaderboard
    # # TODO
