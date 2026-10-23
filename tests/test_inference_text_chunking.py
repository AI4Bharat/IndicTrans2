import unittest

from inference.text_chunking import MAX_SEQ_LEN, truncate_long_sentences


class TruncateLongSentencesTests(unittest.TestCase):
    def test_exact_multiple_does_not_add_an_empty_chunk_or_shift_entity_maps(self):
        long_tokens = [f"token{i}" for i in range(MAX_SEQ_LEN * 2)]
        first_map = {"entity": "first"}
        long_map = {"entity": "long"}
        last_map = {"entity": "last"}

        chunks, maps = truncate_long_sentences(
            ["first", " ".join(long_tokens), "last"],
            [first_map, long_map, last_map],
        )

        long_chunks = [
            " ".join(long_tokens[i : i + MAX_SEQ_LEN])
            for i in range(0, len(long_tokens), MAX_SEQ_LEN)
        ]
        self.assertEqual(["first", *long_chunks, "last"], chunks)
        self.assertEqual([first_map, long_map, long_map, last_map], maps)
        self.assertEqual(long_tokens, " ".join(chunks[1:3]).split())

    def test_partial_final_chunk_is_preserved(self):
        tokens = [f"token{i}" for i in range(MAX_SEQ_LEN * 2 + 1)]
        chunks, maps = truncate_long_sentences([" ".join(tokens)], [{}])

        chunk_lengths = [len(chunk.split()) for chunk in chunks]
        self.assertEqual([MAX_SEQ_LEN, MAX_SEQ_LEN, 1], chunk_lengths)
        self.assertEqual(tokens, " ".join(chunks).split())
        self.assertEqual(3, len(maps))

    def test_sentence_at_limit_remains_single_chunk(self):
        tokens = [f"token{i}" for i in range(MAX_SEQ_LEN)]
        sentence = " ".join(tokens)

        chunks, maps = truncate_long_sentences([sentence], [{"entity": "at-limit"}])

        self.assertEqual([sentence], chunks)
        self.assertEqual([{"entity": "at-limit"}], maps)


if __name__ == "__main__":
    unittest.main()
