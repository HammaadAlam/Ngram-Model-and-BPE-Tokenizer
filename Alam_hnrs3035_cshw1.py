import re
import argparse
import pickle
import random
from collections import defaultdict, Counter

class NgramModel:
    def  __init__ (self, n: int) -> None:
         """
         Initializes an Ngram model with the order
         :param n: order of the Ngram model (1 for unigram, 2 for bigram)
         """
         if n == 1 or n == 2:
             self.n = n
         else:
             raise ValueError("Must be 1 or 2")

         self.context_counts = Counter() #Stores context occurrences
         self.ngrams = defaultdict(Counter) #Stores ngram occurrence
         self.vocab = set() #Stores Unique words
         self.probabilities = defaultdict(Counter) #Stores probability of words following other words

    def train (self, corpus: str):
        """
        Trains the Ngram model with the given corpus based on if it is a unigram or bigram
        :param corpus: corpus to train Ngram model on
        """
        tokens = re.findall(r'\b\w+\b|[.,!?;]', corpus.lower())
        self.vocab.update(tokens) #update

        # Unigram model
        if self.n == 1:
            for i in range(len(tokens) - 1):
                context = tokens[i]  # Previous token
                word = tokens[i + 1]  # Current token
                self.context_counts[context] += 1
                self.ngrams[context][word] += 1

        # Bigram model
        elif self.n == 2:
            for i in range(len(tokens) - 2):
                context = (tokens[i], tokens[i + 1])  # Previous two tokens
                word = tokens[i + 2]  # Current token
                self.context_counts[context] += 1
                self.ngrams[context][word] += 1

        #Calculating probability for words based on previous token(s)
        for context in self.ngrams:
            total_count = sum(self.ngrams[context].values())
            for word in self.ngrams[context]:
                self.probabilities[context][word] = self.ngrams[context][word] / total_count

    def predict_next_word(self, input: tuple, deterministic=False):
        """
        Predicts the next word in the Ngram model based on the given input and trained model
        :param input: input to use for prediction to the Ngram model
        :param deterministic: Decides whether to use a deterministic model
        :return: next word in the Ngram model based on the given input and trained model
        """

        # Unigram predicting next word
        if self.n == 1:
            context = input[0]
        # Bigram predicting next word
        elif self.n == 2:
            context = tuple(input[-2:])  # Previous two tokens

        if context not in self.ngrams:
            raise Exception(f"'{context}' not found in vocabulary.")

        next_word_probabilities = self.ngrams[context]

        #Selecting the next word with the highest probability
        if deterministic:
            return max(next_word_probabilities, key=next_word_probabilities.get)
        #Randomly sample words based on probability distribution
        else:
            words, probabilities = zip(*next_word_probabilities.items())
            return random.choices(words, weights=probabilities, k=1)[0]

class BPEAlgorithm:
    """Class for Byte Pair Encoding Tokenizer"""
    def __init__(self):
        self.vocabulary = set()  # Stores tokens and subwords
        self.token_to_id = {}  # Maps token to ids
        self.id_to_token = {}  # Maps ids back to tokens

    def train(self, corpus: str, k: int = 500):
        """
        Trains the BPE Algorithm with the given corpus to provide a vocabulary to tokenize
        :param corpus: corpus to train BPE Algorithm on
        :param k: number of iterations that train the Algorithm
        """
        corpus_split = list(corpus)  # Tokenize corpus into characters
        self.vocabulary = set(corpus_split)

        # Training algorithm k number of times, deafult k=500
        for i in range(k):
            pairs = Counter()
            for j in range(len(corpus_split) - 1):  # Iterate through corpus to find most frequent subwords
                pairs[(corpus_split[j], corpus_split[j + 1])] += 1  # Create pairs of subwords
            if not pairs:
                break
            print(i)

            most_frequent_pair = max(pairs, key=pairs.get)   # Find most frequent pairs
            new_token = ''.join(most_frequent_pair)  # Join together the most frequent pairs
            self.vocabulary.add(new_token)  # Add the joined pairs to the vocabulary
            i = 0

            # Apply BPE merging based on learned vocabulary
            while i < len(corpus_split) - 1:
                if (corpus_split[i], corpus_split[i + 1]) == most_frequent_pair:
                    corpus_split[i] = new_token
                    del corpus_split[i + 1]
                else:
                    i += 1

        # Map tokens to ids and ids back to tokens
        self.token_to_id = {token: id_num for id_num, token in enumerate(sorted(self.vocabulary))}
        self.id_to_token = {id_num: token for token, id_num in self.token_to_id.items()}

    def tokenize(self, input):
        """
        Tokenize the input string.
        :param input:
        :return: tuple of tokens and token ids
        """
        char_string = list(input)  # Convert the input string to a list of characters
        tokens = []
        token_ids = []

        # Apply the BPE tokenization to the input string based on the vocabulary
        while len(char_string) > 1:
            most_frequent_pair = None

            # Find the most frequent pair of adjacent tokens
            for i in range(len(char_string) - 1):
                pair = (char_string[i], char_string[i + 1])
                if ''.join(pair) in self.vocabulary:  # Check if the pair exists in the vocabulary
                    most_frequent_pair = pair
                    break

            # If no pair is found, stop the _process
            if most_frequent_pair is None:
                break

            # Merge most frequent pair
            i = 0
            while i < len(char_string) - 1:
                if (char_string[i], char_string[i + 1]) == most_frequent_pair:
                    char_string[i] = ''.join(most_frequent_pair)  # Merge pair
                    del char_string[i + 1]  # Remove second part of the pair
                else:
                    i += 1

        # Generate tokens and token IDs
        for token in char_string:
            tokens.append(token)
            token_ids.append(self.token_to_id.get(token, -1))  # Use self.token_to_id to get IDs

        return tokens, token_ids

def main():
    """
    Main function to set up all necessary command line arguments
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('activity', choices=['train_ngram', 'predict_ngram', 'train_bpe', 'tokenize'])
    parser.add_argument('--data', type=str)
    parser.add_argument('--save', type=str)
    parser.add_argument('--load', type=str)
    parser.add_argument('--word', type=str)
    parser.add_argument('--nwords', type=int)
    parser.add_argument('--text', type=str)
    parser.add_argument('--n', type=int)
    parser.add_argument('--k', type=int, default=500)
    parser.add_argument('--d', action='store_true')

    args = parser.parse_args()

    if args.activity == 'train_ngram':

        #Opening and reading data file
        with open(args.data, "r") as f:
            corpus = f.read()

        #Intializing Ngram model and training it using given corpus
        model = NgramModel(args.n)
        model.train(corpus)

        #Saving trained model to file
        with open(args.save, "wb") as f:
            pickle.dump(model, f)

    elif args.activity == 'predict_ngram':

        with open(args.load, "rb") as f:
            model = pickle.load(f)

        #Unigram model
        if model.n == 1:

            ##Getting input words to predict based off of
            input_word = args.word.strip()
            if not input_word:
                raise Exception("Error: Unigram requires an input word")

            generated_words = [input_word]

            for _ in range(args.nwords):
                next_word = model.predict_next_word([input_word], deterministic=args.d)
                if not next_word:
                    break

                generated_words.append(next_word)
                input_word = next_word

        #Bigram model
        else:

            #Getting input words to predict based off of
            input_words = tuple(args.word.strip().split())
            if len(input_words) != 2:
                raise Exception("Error: Bigram requires two input words")

            generated_words = list(input_words)
            for _ in range(args.nwords):

                next_word = model.predict_next_word(tuple(generated_words[-2:]), deterministic=args.d)

                if not next_word:
                    break
                generated_words.append(next_word)
        print("Generated text:", " ".join(generated_words))

    elif args.activity == 'train_bpe':

        #Opening and reading data file
        with open(args.data, "r") as f:
            corpus = f.read()

        #Initializing BPEAlgorithm and training it using given corpus
        model = BPEAlgorithm()
        model.train(corpus, k=args.k)

        #Saving trained model to file
        with open(args.save, "wb") as f:
            pickle.dump(model, f)

    elif args.activity == 'tokenize':

        #Loading trained BPE model
        with open(args.load, 'rb') as f:
            bpe_model = pickle.load(f)

        #Tokenizing text and printing tokens and ids
        tokens, token_ids = bpe_model.tokenize(args.text)
        print('Tokens:', tokens)
        print('Token IDs:', token_ids)

if __name__ == '__main__':
    main()