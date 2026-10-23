from typing import Dict, List, Tuple


MAX_SEQ_LEN = 256


def truncate_long_sentences(
    sents: List[str], placeholder_entity_map_sents: List[Dict]
) -> Tuple[List[str], List[Dict]]:
    """Split tokenized sentences into model-length chunks, preserving entity maps."""
    new_sents = []
    placeholders = []

    for j, sent in enumerate(sents):
        words = sent.split()
        if len(words) > MAX_SEQ_LEN:
            chunks = [
                " ".join(words[i : i + MAX_SEQ_LEN])
                for i in range(0, len(words), MAX_SEQ_LEN)
            ]
            new_sents.extend(chunks)
            placeholders.extend([placeholder_entity_map_sents[j]] * len(chunks))
        else:
            new_sents.append(sent)
            placeholders.append(placeholder_entity_map_sents[j])

    return new_sents, placeholders
