from cre_eval.cli import WEIGHT_PARTS


def test_checkpoint_is_split_for_github_release():
    assert WEIGHT_PARTS == ("cre_nudity.part-00", "cre_nudity.part-01")
