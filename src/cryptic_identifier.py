import json
import re
import nltk
import wordninja
from typing import List, Tuple

# Ensure necessary NLTK components are downloaded
nltk.download('wordnet')
from nltk.stem.wordnet import WordNetLemmatizer


class CrypticIdentifier:
    """
    Module to identify cryptic column headers. It evaluates whether a given text
    is cryptic or logically understandable.
    Example usage:
        identifier = CrypticIdentifier(vocab_file="wordnet.json")
        identifier.iscryptic("newyorkcitytotalpopulation")  # False
        identifier.iscryptic("tot_revq4")  # True
    """

    def __init__(self, vocab_file=None, word_rank_file=None, k_whole=4, k_split=2):
        """
        Initialize CrypticIdentifier with vocab, word ranks, and thresholds.
        Args:
            vocab_file (str, optional): Path to a JSON vocabulary file. Defaults to None.
            word_rank_file (str, optional): Path to word ranking file for wordninja. Defaults to None.
            k_whole (int, optional): Length threshold for whole strings. Defaults to 4.
            k_split (int, optional): Length threshold for split words. Defaults to 2.
        """
        self.k_whole = k_whole
        self.k_split = k_split
        self.lem = WordNetLemmatizer()

        # Load vocabulary if provided
        self.vocab = None
        if vocab_file:
            with open(vocab_file, "r") as file:
                self.vocab = json.load(file)
                print(f"Loaded vocabulary: {len(self.vocab)} entries.")

        # Configure word splitting model
        self.splitter = wordninja.LanguageModel(word_rank_file) if word_rank_file else wordninja

    def split_rm_punc(self, text: str) -> List[str]:
        """
        Split text into words while removing punctuation.
        """
        return re.sub(r'[^\w\s]', ' ', text).split()

    def separate_camel_case(self, text: str) -> List[str]:
        """
        Split camel case into separate words.
        """
        return re.findall(r'[A-Z][a-z]*|[a-z]+|\d+', text)

    def convert_to_base(self, text: str) -> str:
        """
        Convert a word to its lemmatized (base) form.
        """
        return self.lem.lemmatize(text.lower())

    def _split(self, text: str) -> List[str]:
        """
        Split text into logical components using camel case and punctuation.
        """
        text = text.replace('_', ' ')
        return self.split_rm_punc(" ".join(self.separate_camel_case(text)))

    def _iscryptic(self, text: str) -> bool:
        """
        Determine if a single word is cryptic based on vocabulary and structure.
        """
        words = self._split(text)
        if all(word.isnumeric() for word in words):
            return True
        if self.vocab is None:
            self.vocab = set(nltk.corpus.wordnet.words('english'))
        return any(self.convert_to_base(word) not in self.vocab for word in words)

    def doublecheck_cryptic(self, text: str) -> Tuple[bool, List[str]]:
        """
        Perform additional checks to confirm if text contains cryptic elements.
        Args:
            text (str): Input column header text.
        Returns:
            Tuple[bool, List[str]]: Whether the text is cryptic and its split components.
        """
        def split_check(words: List[str]) -> Tuple[bool, List[str]]:
            cryptic_flags = [
                len(word) < self.k_split or self._iscryptic(word) for word in words
            ]
            return any(cryptic_flags), words

        if len(text) >= self.k_whole and self._iscryptic(text):
            split_words = self.splitter.split(text)
            return split_check(split_words)
        return False, self._split(text)

    def iscryptic(self, text: str) -> bool:
        """
        Determine if a text is cryptic.
        """
        return self.doublecheck_cryptic(text)[0]

    def split_results(self, text: str) -> List[str]:
        """
        Retrieve split tokens from the input text.
        """
        return self.doublecheck_cryptic(text)[1]


def read_gt_names(file_name: str) -> dict:
    """
    Read ground truth names from a file.
    Args:
        file_name (str): Path to the file containing ground truth names.
    Returns:
        dict: Dictionary of names.
    """
    names = {}
    with open(file_name, "r") as file:
        for line in file:
            names[line.strip().lower()] = True
    return names


if __name__ == "__main__":
    import argparse

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Cryptic column header identifier")
    parser.add_argument('--text', type=str, required=True, help="Input text to check")
    parser.add_argument('--vocab_path', type=str, default="./lookups/wordnet.json",
                        help="Path to vocabulary JSON file")
    parser.add_argument('--word_rank_path', type=str, default="./lookups/wordninja_words_alpha.txt.gz",
                        help="Path to wordninja rank file")
    args = parser.parse_args()

    # Initialize CrypticIdentifier and process input
    identifier = CrypticIdentifier(vocab_file=args.vocab_path, word_rank_file=args.word_rank_path)
    is_cryptic, split_res = identifier.doublecheck_cryptic(args.text)

    print(f"Is Cryptic: {is_cryptic}")
    print(f"Split Result: {split_res}")