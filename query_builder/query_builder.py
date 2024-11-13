import re
from collections import defaultdict

import faiss
import numpy as np
import torch
from optimum.onnxruntime import (ORTModelForSequenceClassification,
                                 ORTModelForTokenClassification)
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer


# folder contains all assets for this project
ASSET_FOLDER = "./assets/"

# Load the embedding model 
sentence_embber = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


# cross encoder for ranking  
cross_encoder_path = "CelDom/ce-ms-marco-MiniLM-L-6-v2"
cross_encoder = ORTModelForSequenceClassification.from_pretrained(
    cross_encoder_path
)
ce_tokenizer = AutoTokenizer.from_pretrained(
    cross_encoder_path
)

# keyword model encoder 
origin_model_path = f"CelDom/sambit_bird"
file_path = f"{ASSET_FOLDER}/words_alpha.txt"
keyword_tokenizer = AutoTokenizer.from_pretrained(origin_model_path)
keywords_model = ORTModelForTokenClassification.from_pretrained(
    origin_model_path
)

# faiss index
index = faiss.read_index_binary(f"{ASSET_FOLDER}/word_binary.faiss")

# model threshold
threshold = 0.7


def load_file_to_lines(filepath):
    """Load content of a file to list"""
    with open(filepath, "r") as file:
        lines = file.readlines()
    return [line.strip() for line in lines]  # Remove any trailing newlines or spaces


vocab_base = load_file_to_lines(file_path)
stop_words_base = set(load_file_to_lines(f"{ASSET_FOLDER}/stopwords-en.v1.txt"))


def warmup():
    """ Warmup cross encoder model"""
    texts = [
        ["How many people live in Berlin?", "How many people live in Berlin?"],
        [
            "Berlin has a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.",
            "New York City is famous for the Metropolitan Museum of Art.",
        ],
    ] * 40

    warmup_features = ce_tokenizer(texts[:10], padding=True, return_tensors="pt")
    with torch.no_grad():
        scores = cross_encoder(**warmup_features).logits


warmup()


def find_closest_words_faiss(sentences, index, noun_list, top_n=10):
    """
    Finds the closest nouns using the Faiss index.
    Get the start and end indices of each word in the string

    Parameters:
        sentences (list of str): list of sub-phrases
        index: Faiss index
        noun_list: list of words matches with Faiss index
        top_n: number of returned words

    Returns:
        list of tuples: Each tuple contains (word, embedding distance) for a word.
    """
    sentence_embedding = sentence_embber.encode(sentences).astype(
        "float32"
    )  # Reshape for Faiss
    sentence_embedding = np.packbits(sentence_embedding > 0).reshape(len(sentences), -1)

    # Search the Faiss index
    D, I = index.search(sentence_embedding, top_n)  # D: Distances, I: Indices
    closest_words = [
        (noun_list[i], float(d))
        for j in range(len(sentences))
        for i, d in zip(I[j], D[j])
    ]  # Get nouns and similarity scores from faiss
    return closest_words


def keep_word(text):
    """
    Given a phrase, extract the keywords from this phrase

    Parameters:
        text (str): Input string.

    Returns:
        2 lists of str: first is keywords, second is what left from text
    """
    inputs = keyword_tokenizer(text, return_tensors="pt").to(keywords_model.device)
    with torch.no_grad():
        logits = keywords_model(**inputs).logits.numpy()
    probs = torch.softmax(torch.from_numpy(logits), dim=2)
    set_keep_text = defaultdict(lambda: 0)
    for i, t in enumerate(probs[0]):
        word_id = inputs.word_ids(0)[i]
        if word_id is None:
            continue
        sp = inputs.word_to_chars(0, word_id)
        set_keep_text[(sp.start, sp.end)] = max(
            set_keep_text[(sp.start, sp.end)], t[1].item()
        )

    ## correct for the over tokenization
    ## recontruct the probs of keyword from tokens to sentence's words
    word_indicies = [(s, t) for s, t in get_word_indices(text) if s < t]
    char_token_intervals, probs = list(set_keep_text.keys()), list(
        set_keep_text.values()
    )
    word_probs = remap_scores(word_indicies, char_token_intervals, probs)
    keep_words = [
        word_indicies[i] for i, prob in enumerate(word_probs) if prob > threshold
    ]
    return split_string_by_intervals(text, keep_words)


def split_string_by_intervals(s, intervals):
    """
    Given string s and list of intervals indicate where in string s we would like to extract, 
    return 2 list of string, one is corresponds with intervals provided, other is a list of 
    interval of s which is created after remove the intervals provided from s. If we concat 2 lists, 
    we recontruct the original s. 

    Parameters:
        s (str): Input string.

    Returns:
        2 lists of string: substring segments. 
    """
    # Sort intervals by start to ensure they are processed in order
    intervals = sorted(intervals)

    # Initialize result lists
    in_interval = []
    out_interval = []

    # Temporary variables
    temp_in_interval = []
    temp_out_interval = []
    interval_index = 0
    start, end = intervals[interval_index] if intervals else (None, None)

    # Iterate over each character in the string with its index
    for i, char in enumerate(s):
        # Check if we're past the end of the current interval
        if start is not None and i >= end:
            # Add the collected interval substring to in_interval
            in_interval.append("".join(temp_in_interval))
            temp_in_interval.clear()
            interval_index += 1

            # Update to the next interval, if available
            if interval_index < len(intervals):
                start, end = intervals[interval_index]
            else:
                start, end = None, None  # No more intervals to process

        # Collect characters within and outside intervals
        if start is not None and start <= i < end:
            if temp_out_interval:
                # Append out_interval segment and clear temp
                out_interval.append("".join(temp_out_interval))
                temp_out_interval.clear()
            temp_in_interval.append(char)
        else:
            if temp_in_interval:
                # Append in_interval segment and clear temp
                in_interval.append("".join(temp_in_interval))
                temp_in_interval.clear()
            temp_out_interval.append(char)

    # Append any remaining characters
    if temp_in_interval:
        in_interval.append("".join(temp_in_interval))
    if temp_out_interval:
        out_interval.append("".join(temp_out_interval))

    return in_interval, out_interval


def get_word_indices(s):
    """
    Get the start and end indices of each word in the string

    Parameters:
        s (str): Input string.

    Returns:
        list of tuples: Each tuple contains (start_index, end_index) for a word.
    """
    word_indices = []
    n = len(s)
    start_index = None

    for i in range(n):
        # Skip spaces
        if s[i].isalnum():
            if start_index is None:
                start_index = i  # Found the start of a word
        else:
            if start_index is not None:
                word_indices.append((start_index, i))  # End of the word
                start_index = None  # Reset for the next word

    # If the last word is not followed by a space
    if start_index is not None:
        word_indices.append((start_index, n))

    return word_indices


def remap_scores(intervals_a, intervals_b, scores_b):
    """
    Remap scores from intervals in B to intervals in A

    Parameters:
        intervals_a (list of tuples): List of intervals in A [(start, end), ...].
        intervals_b (list of tuples): List of intervals in B [(start, end), ...].
        scores_b (list of floats): Scores associated with each interval in B.

    Returns:
        list of floats: Remapped scores for each interval in A.
    """

    # Initialize pointers and result list
    result = [0] * len(intervals_a)
    b_pointer = 0

    # Iterate through each interval in A
    for i, (start_a, end_a) in enumerate(intervals_a):
        total_score = 0

        # Process all intervals in B that end before or within the current interval in A
        while b_pointer < len(intervals_b):
            start_b, end_b = intervals_b[b_pointer]

            # If B's interval is completely inside A's interval
            if start_a <= start_b and end_b <= end_a:
                total_score = max(total_score, scores_b[b_pointer])
                b_pointer += 1
            else:
                break

        # Store the score for the current interval in A
        result[i] = total_score

    return result


def clean_string(input_string):
    """
    remove non-char from string except space and filter a, an, the 

    Parameters:
        s (str): Input string.

    Returns:
        str: cleaned string
    """
    # Remove non-alphanumeric characters, keeping spaces
    cleaned_string = re.sub(r"[^A-Za-z0-9\s]", "", input_string)

    # Remove "a", "an", and "the" as standalone words
    words = cleaned_string.split()
    filtered_words = [word for word in words if word.lower() not in {"a", "an", "the"}]

    # Join the words back into a single string
    result_string = " ".join(filtered_words)

    return result_string


def remove_stop_words(phrase):
    """
    Remove stop words from phrase

    Parameters:
        phrase (str): Input string.

    Returns:
        str: cleaned phrase.
    """
    words = [
        w.strip() for w in phrase.split() if w.strip().lower() not in stop_words_base
    ]
    return " ".join(words)


def query_suggestion(text, top_n=10):
    """
    Given a question or instruction, suggests queries that might provide information that
    answer the question or instruction. 

    Parameters:
        text (str): question, instruction.

    Returns:
        list of str: suggested queries.
    """
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    suggestion_seed, suggestion_refinement_seed = keep_word(text)
    suggestion_seed = clean_string(" ".join(suggestion_seed))
    suggestion_refinement_seed = [
        x.strip() for x in suggestion_refinement_seed if x.strip()
    ]
    suggestion_refinement_seed = [
        remove_stop_words(p) for p in suggestion_refinement_seed
    ]
    suggestion_refinement_seed = [p for p in suggestion_refinement_seed if p.strip()]
    closest_words = find_closest_words_faiss(
        suggestion_refinement_seed, index, vocab_base, top_n=6
    )
    closest_words = [
        w[0].strip() for w in closest_words if w[0].strip() not in stop_words_base
    ]

    if closest_words:
        suggestions = [suggestion_seed + " " + word for word in closest_words]
        text_pairs = [[text, suggest] for suggest in suggestions]
        features = ce_tokenizer(text_pairs, padding=True, return_tensors="pt")
        ranks = cross_encoder(**features).logits
        top_n = min(top_n, ranks.shape[0])
        _, indices = torch.topk(ranks, top_n, dim=0, largest=True)
        return [suggestion_seed] + [suggestions[i] for i in indices]
    return [suggestion_seed]
