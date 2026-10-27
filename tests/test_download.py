from cre_eval.cli import WEIGHT_PARTS


def test_checkpoint_is_split_for_github_release():
    assert list(WEIGHT_PARTS) == ["cre_nudity.part-00", "cre_nudity.part-01"]
    assert sum(WEIGHT_PARTS.values()) == 2_931_327_546
