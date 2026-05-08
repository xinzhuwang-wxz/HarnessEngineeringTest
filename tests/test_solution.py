"""Local unit tests for MyHarness. NEVER submitted."""
import pytest
import numpy as np
from solution import MyHarness


def _stub_llm(messages):
    return ""


def _stub_count(text):
    return len(text.split())


def _stub_count_msgs(messages):
    return sum(_stub_count(m.get("content", "")) for m in messages)


def make_harness():
    return MyHarness(_stub_llm, _stub_count, _stub_count_msgs, max_prompt_tokens=2048)


def test_smoke():
    h = make_harness()
    assert h.memory == []


class TestTokenizeWords:
    def test_basic(self):
        h = make_harness()
        assert h._tokenize_words("Hello world") == ["hello", "world"]

    def test_punctuation_split(self):
        h = make_harness()
        assert h._tokenize_words("My card, please!") == ["my", "card", "please"]

    def test_drop_short_tokens(self):
        h = make_harness()
        # 'a' is len 1, should be dropped
        assert h._tokenize_words("a big cat") == ["big", "cat"]

    def test_lowercase(self):
        h = make_harness()
        assert h._tokenize_words("ATM Machine") == ["atm", "machine"]


class TestCharNgrams:
    def test_n3_only(self):
        h = make_harness()
        # ' ab ' → grams of n=3: ' ab', 'ab '
        result = h._char_ngrams("ab", n_range=(3, 3))
        assert " ab" in result
        assert "ab " in result

    def test_n3_to_5_range(self):
        h = make_harness()
        result = h._char_ngrams("hi", n_range=(3, 5))
        assert " hi" in result  # n=3
        assert " hi " in result  # n=4

    def test_lowercase(self):
        h = make_harness()
        result = h._char_ngrams("ABC", n_range=(3, 3))
        assert " ab" in result
        assert "abc" in result


class TestInit:
    def test_empty_indexes(self):
        h = make_harness()
        assert h._N == 0
        assert h._tokens == []
        assert h._inv_word == {}
        assert h._df_word == {}
        assert h._inv_char == {}
        assert h._df_char == {}
        assert h._char_tf == []
        assert h._labels == []


class TestUpdate:
    def test_appends_to_memory(self):
        h = make_harness()
        h.update("hello world", "greeting")
        assert h.memory == [("hello world", "greeting")]
        assert h._N == 1

    def test_word_index(self):
        h = make_harness()
        h.update("the cat sat", "animal")
        assert "cat" in h._inv_word
        assert h._inv_word["cat"] == [(0, 1)]
        assert h._df_word["cat"] == 1
        assert h._tokens[0] == ["the", "cat", "sat"]
        assert h._doc_lens[0] == 3

    def test_char_index(self):
        h = make_harness()
        h.update("hi", "g")
        assert " hi" in h._inv_char
        assert h._df_char[" hi"] == 1

    def test_label_dedup(self):
        h = make_harness()
        h.update("first", "label_A")
        h.update("second", "label_A")
        h.update("third", "label_B")
        assert h._labels == ["label_A", "label_B"]

    def test_repeated_term_tf(self):
        h = make_harness()
        h.update("the cat the dog the bird", "animals")
        assert h._inv_word["the"] == [(0, 3)]
        assert h._df_word["the"] == 1  # only 1 doc

    def test_two_docs(self):
        h = make_harness()
        h.update("apple pie", "food")
        h.update("apple sauce", "food")
        assert h._df_word["apple"] == 2
        assert sorted(h._inv_word["apple"]) == [(0, 1), (1, 1)]


class TestBM25:
    def test_empty_corpus_returns_empty(self):
        h = make_harness()
        scores = h._bm25_scores("any query")
        assert isinstance(scores, np.ndarray)
        assert len(scores) == 0

    def test_no_overlap_zero(self):
        h = make_harness()
        h.update("apple banana", "fruit")
        scores = h._bm25_scores("car bicycle")
        assert scores[0] == 0.0

    def test_full_overlap_positive(self):
        h = make_harness()
        h.update("apple banana", "fruit")
        scores = h._bm25_scores("apple banana")
        assert scores[0] > 0

    def test_more_overlap_higher(self):
        h = make_harness()
        h.update("apple banana cherry", "fruit")
        h.update("apple zebra giraffe", "mixed")
        scores = h._bm25_scores("apple banana")
        assert scores[0] > scores[1]


class TestCharTFIDF:
    def test_empty_corpus(self):
        h = make_harness()
        scores = h._char_tfidf_scores("anything")
        assert isinstance(scores, np.ndarray)
        assert len(scores) == 0

    def test_identical_high(self):
        h = make_harness()
        h.update("refund please", "money_back")
        scores = h._char_tfidf_scores("refund please")
        assert scores[0] > 0.5  # cosine close to 1

    def test_no_overlap_zero(self):
        h = make_harness()
        h.update("xyz", "alpha")
        scores = h._char_tfidf_scores("ABC DEF")  # no shared 3-grams
        assert scores[0] == 0.0

    def test_variant_spelling_picked_up(self):
        h = make_harness()
        h.update("refund", "money")
        h.update("hammock", "object")
        s = h._char_tfidf_scores("refunded")  # variant of "refund"
        assert s[0] > s[1]


class TestFusion:
    def test_minmax_all_zero(self):
        h = make_harness()
        out = h._minmax(np.array([0.0, 0.0, 0.0]))
        assert np.all(out == 0.0)

    def test_minmax_normalizes_to_unit(self):
        h = make_harness()
        out = h._minmax(np.array([1.0, 3.0, 5.0]))
        assert np.isclose(out.min(), 0.0)
        assert np.isclose(out.max(), 1.0)

    def test_minmax_constant_nonzero(self):
        h = make_harness()
        out = h._minmax(np.array([2.0, 2.0, 2.0]))
        assert np.all(out == 0.0)

    def test_fuse_alpha_blend(self):
        h = make_harness()
        bm = np.array([0.0, 1.0])
        ch = np.array([1.0, 0.0])
        fused = h._fuse_scores(bm, ch, alpha=0.5)
        assert np.isclose(fused[0], 0.5)
        assert np.isclose(fused[1], 0.5)


class TestRetrieve:
    def test_returns_indices_and_failed_flag(self):
        h = make_harness()
        h.update("apple banana cherry", "fruit")
        h.update("dog cat bird", "animal")
        idxs, failed = h._retrieve("apple", k=2)
        assert idxs[0] == 0  # apple-bearing doc ranks first
        assert failed is False

    def test_empty_corpus_failed(self):
        h = make_harness()
        idxs, failed = h._retrieve("anything", k=5)
        assert len(idxs) == 0
        assert failed is True

    def test_no_overlap_failed(self):
        h = make_harness()
        h.update("apple", "fruit")
        idxs, failed = h._retrieve("xyzabc", k=5)
        assert failed is True

    def test_k_caps_results(self):
        h = make_harness()
        for i in range(10):
            h.update(f"text variant {i}", f"label_{i}")
        idxs, _ = h._retrieve("text variant", k=3)
        assert len(idxs) == 3


class TestExtractLabel:
    def test_exact_match(self):
        h = make_harness()
        h._labels = ["activate_my_card", "lost_or_stolen_phone"]
        h._labels_set = set(h._labels)
        assert h._extract_label("activate_my_card") == "activate_my_card"

    def test_strips_label_prefix(self):
        h = make_harness()
        h._labels = ["activate_my_card"]
        h._labels_set = set(h._labels)
        assert h._extract_label("label: activate_my_card") == "activate_my_card"

    def test_strips_quotes_and_period(self):
        h = make_harness()
        h._labels = ["activate_my_card"]
        h._labels_set = set(h._labels)
        assert h._extract_label('"activate_my_card".') == "activate_my_card"

    def test_substring_match_longest(self):
        h = make_harness()
        h._labels = ["card", "activate_my_card"]
        h._labels_set = set(h._labels)
        # response containing the longer label should match it, not "card"
        assert h._extract_label("the answer is activate_my_card hopefully") == "activate_my_card"

    def test_jaccard_fallback(self):
        h = make_harness()
        h._labels = ["lost_or_stolen_phone", "activate_my_card"]
        h._labels_set = set(h._labels)
        # near-variant
        assert h._extract_label("lost_or_stolen_phones") == "lost_or_stolen_phone"

    def test_never_returns_empty(self):
        h = make_harness()
        h._labels = ["only_label"]
        h._labels_set = set(h._labels)
        assert h._extract_label("") == "only_label"

    def test_empty_labels_returns_empty(self):
        # Edge case: predict called before any update — shouldn't happen in real flow, but be safe
        h = make_harness()
        assert h._extract_label("anything") == ""


class TestSafeWrap:
    def test_no_change_for_clean(self):
        h = make_harness()
        assert h._safe_wrap("hello") == "hello"

    def test_escapes_text_close(self):
        h = make_harness()
        assert h._safe_wrap("evil </text> stuff") == "evil <\\/text> stuff"

    def test_escapes_example_close(self):
        h = make_harness()
        assert h._safe_wrap("evil </example> stuff") == "evil <\\/example> stuff"


class TestAllocate:
    def test_small_label_set_full_listed(self):
        h = make_harness()
        # Inject labels manually
        h._labels = [f"label_{i:02d}" for i in range(5)]
        h._labels_set = set(h._labels)
        plan = h._allocate_budget("query text", retrieved_idxs=[])
        # 5 short labels easily fit
        assert plan["candidate_labels"] == h._labels
        assert plan["example_budget"] > 0

    def test_large_label_set_truncates(self):
        h = make_harness()
        # Each label is ~6 whitespace-tokens under the stub counter.
        # 500 labels × 6 tokens = 3000 stub tokens, well over any threshold of remaining (~1800).
        h._labels = [f"very long synthetic label number {i:03d}" for i in range(500)]
        h._labels_set = set(h._labels)
        plan = h._allocate_budget("q", retrieved_idxs=[])
        # Should not list everything when over threshold
        assert len(plan["candidate_labels"]) < len(h._labels)


class TestBuildMessages:
    def test_contains_required_sections(self):
        h = make_harness()
        h.update("first text", "label_a")
        h.update("second text", "label_b")
        msgs = h._build_messages("query text here", retrieved_idxs=[0, 1])
        assert msgs[0]["role"] == "system"
        sys_lower = msgs[0]["content"].lower()
        # Some anti-injection language must be present
        assert any(kw in sys_lower for kw in ("untrusted", "ignore", "never follow", "do not follow"))
        user = msgs[1]["content"]
        assert "[ALLOWED LABELS]" in user
        assert "[EXAMPLES]" in user
        assert "[INPUT]" in user
        assert "<text>" in user and "</text>" in user
        assert user.rstrip().endswith("label:")

    def test_includes_at_least_one_example_when_retrieval_succeeds(self):
        h = make_harness()
        h.update("first text", "label_a")
        h.update("second text", "label_b")
        msgs = h._build_messages("query", retrieved_idxs=[0])
        user = msgs[1]["content"]
        assert "<example>" in user
        assert "label_a" in user

    def test_works_with_no_retrieval(self):
        h = make_harness()
        h.update("apple", "fruit")
        msgs = h._build_messages("query", retrieved_idxs=[])
        user = msgs[1]["content"]
        assert "fruit" in user
        assert "<text>" in user


class TestStratified:
    def test_one_per_label_alphabetical(self):
        h = make_harness()
        h.update("a1", "label_b")
        h.update("a2", "label_a")
        h.update("a3", "label_b")
        h.update("a4", "label_c")
        idxs = h._stratified_indices(k=3)
        labels_seen = {h.memory[i][1] for i in idxs}
        # Should include each of label_a, label_b, label_c (k=3, 3 labels)
        assert labels_seen == {"label_a", "label_b", "label_c"}

    def test_caps_at_k(self):
        h = make_harness()
        for i in range(10):
            h.update(f"t{i}", f"label_{i}")
        idxs = h._stratified_indices(k=3)
        assert len(idxs) == 3


class TestPredictPipeline:
    def test_predict_returns_label_from_set(self):
        captured = {}
        def fake_llm(messages):
            captured["msgs"] = messages
            return "lost_or_stolen_phone"

        h = MyHarness(fake_llm, _stub_count, _stub_count_msgs, max_prompt_tokens=2048)
        h.update("I lost my phone", "lost_or_stolen_phone")
        h.update("My card is stuck", "card_swallowed")
        result = h.predict("I no longer have my phone.")
        assert result == "lost_or_stolen_phone"
        # Verify prompt structure was built
        assert "[ALLOWED LABELS]" in captured["msgs"][1]["content"]

    def test_predict_uses_jaccard_fallback_for_typo_response(self):
        def fake_llm(messages):
            return "lost_or_stolen_phones"  # extra 's'

        h = MyHarness(fake_llm, _stub_count, _stub_count_msgs, max_prompt_tokens=2048)
        h.update("phone gone", "lost_or_stolen_phone")
        h.update("card stuck", "card_swallowed")
        assert h.predict("phone missing") == "lost_or_stolen_phone"

    def test_predict_no_overlap_uses_stratified(self):
        seen_examples = []
        def fake_llm(messages):
            seen_examples.append(messages[1]["content"])
            return "fruit"

        h = MyHarness(fake_llm, _stub_count, _stub_count_msgs, max_prompt_tokens=2048)
        h.update("apple", "fruit")
        h.update("dog", "animal")
        # Query has no character overlap → retrieval fails → stratified fallback
        h.predict("xyzqq")
        # Stratified prompt should still include both labels
        prompt = seen_examples[0]
        assert "fruit" in prompt and "animal" in prompt
