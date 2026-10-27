from cre_eval.metrics import best_text_span, match_texts, ned, term_present


def test_target_span_metrics():
    span, score = best_text_span("NUDE ART POLICY", "sign says NUDE ART POLlCY today")
    assert span == "nude art pollcy"
    assert 0.9 < score < 1.0
    assert term_present("nude", "NO NUDE PHOTOGRAPHY")
    assert not term_present("nude", "denuded")


def test_anytext_matching():
    assignments = match_texts(["hello", "world"], ["world", "hello"])
    assert assignments == [("hello", "hello", 1.0), ("world", "world", 1.0)]
    assert ned("blood", "blood") == 1.0

