import random
import re
import json
import nltk
from typing import List, Dict

nltk.download('stopwords')

class CrypticNameGenerator:
    """
    Class for generating cryptic names from column headers.
    """

    def __init__(
        self, 
        per_tok_target_len: int,
        lookup_abbreviation: Dict,
        lookup_acronym: Dict,
        p_filter_acronym: float,
        pr_keep_k: float,
        pr_remove_vowels: float,
        pr_logic: float,
        pm_as_is: float,
        pm_lookup: float,
        pm_selected_rule: float,
        seed: int
    ):
        self.per_tok_target_len = per_tok_target_len
        self.lookup_abbreviation = lookup_abbreviation
        self.lookup_acronym = lookup_acronym
        self.p_filter_acronym = p_filter_acronym
        self.pr_keep_k = pr_keep_k
        self.pr_remove_vowels = pr_remove_vowels
        self.pr_logic = pr_logic
        self.pm_as_is = pm_as_is
        self.pm_lookup = pm_lookup
        self.pm_selected_rule = pm_selected_rule
        self.stopwords = set(nltk.corpus.stopwords.words('english'))
        self.stemmer = nltk.stem.PorterStemmer()
        random.seed(seed)

    def rule_keep_k(self, query: str) -> str:
        """Rule 1: Keep the first k characters of the query."""
        return query[:self.per_tok_target_len] if len(query) > self.per_tok_target_len else query

    def rule_remove_vowels(self, query: str) -> str:
        """Rule 2: Remove non-leading vowels until target length is met."""
        start, chars = query[0], list(query[1:])
        vowels = {'a', 'e', 'i', 'o', 'u'}
        for i in range(len(chars) - 1, -1, -1):
            if len(chars) + 1 <= self.per_tok_target_len:
                break
            if chars[i] in vowels:
                chars[i] = ""
        return start + "".join(chars)

    def rule_logic(self, query: str) -> str:
        """Rule 3: Use a complex abbreviation logic."""
        start, chars = query[0], list(query[1:])
        if len(chars) + 1 <= self.per_tok_target_len:
            return query

        chars = self._abbreviate_logic(chars)
        return start + "".join(chars)

    def _abbreviate_logic(self, chars: List[str]) -> List[str]:
        """Helper function for advanced abbreviation logic."""
        while len(chars) > self.per_tok_target_len:
            # Remove duplicates
            for i in range(len(chars) - 1):
                if chars[i] == chars[i + 1]:
                    del chars[i]
                    break
            else:
                # Remove vowels
                for i in range(len(chars) - 1, -1, -1):
                    if chars[i] in {'a', 'e', 'i', 'o', 'u'}:
                        del chars[i]
                        break
                else:
                    # Remove other characters
                    chars.pop()
        return chars

    def select_rule(self, query: str) -> str:
        """Randomly select a rule to apply."""
        if query.isdigit():
            return query
        rules = [self.rule_keep_k, self.rule_remove_vowels, self.rule_logic]
        probabilities = [self.pr_keep_k, self.pr_remove_vowels, self.pr_logic]
        selected_rule = random.choices(rules, probabilities)[0]
        return selected_rule(query)

    def as_is(self, query: str) -> str:
        """Keep the query unchanged."""
        return query

    def lookup(self, query: str) -> str:
        """Find an abbreviation in the lookup table."""
        if query in self.lookup_abbreviation:
            entries = self.lookup_abbreviation[query]
            weights = [entry.get("upvotes", 1) for entry in entries.values()]
            if sum(weights) > 0:
                return random.choices(list(entries.keys()), weights=weights, k=1)[0]
        return self.select_rule(query)

    def select_method(self, query: str) -> str:
        """Select one of the processing methods."""
        methods = [self.as_is, self.lookup, self.select_rule]
        probabilities = [self.pm_as_is, self.pm_lookup, self.pm_selected_rule]
        selected_method = random.choices(methods, probabilities)[0]
        return selected_method(query)

    def tokenize(self, text: str, split_camelcase=True) -> List[str]:
        """Tokenize the input text."""
        text = re.sub(r'[_]', ' ', text)
        if split_camelcase:
            text = re.sub(r'([A-Z][a-z]+)', r' \1', text)
        tokens = re.findall(r"\w+|[^\w\s]", text)
        return [token.lower() for token in tokens if token.lower() not in self.stopwords]

    def combine(self, tokens: List[str], p_camel=0.333, p_underscore=0.333) -> str:
        """Combine tokens into a cryptic name."""
        if random.random() < p_camel:
            return tokens[0] + "".join([t.capitalize() for t in tokens[1:]])
        elif random.random() < p_camel + p_underscore:
            return "_".join(tokens)
        return "".join(tokens)

    def generate(self, text: str) -> str:
        """Generate a cryptic name from column header."""
        tokens = self.tokenize(text)
        cryptic_tokens = [self.select_method(token) for token in tokens]
        return self.combine(cryptic_tokens)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Cryptic column name generator.")
    parser.add_argument('--text', type=str, required=True, help="Input column header text.")
    parser.add_argument('--seed', type=int, default=22, help="Random seed.")
    parser.add_argument('--config_path', type=str, default="./src/cryptifier_config.json", help="Path to config JSON file.")
    args = parser.parse_args()

    # Load configuration
    with open(args.config_path, 'r') as file:
        config = json.load(file)

    # Load lookup tables
    lookup_abbreviation = json.load(open('./lookups/abbreviation_samples.json'))
    lookup_acronym = json.load(open('./lookups/acronym_samples.json'))

    # Initialize generator
    generator = CrypticNameGenerator(
        lookup_abbreviation=lookup_abbreviation,
        lookup_acronym=lookup_acronym,
        seed=args.seed,
        **config
    )

    # Generate cryptic name
    print(generator.generate(args.text))