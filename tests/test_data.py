from cre_eval.data import expected_count, load_items


def test_frozen_dataset_sizes():
    for name in ("nudity-gr", "nudity-tr", "coco", "anytext"):
        assert len(load_items(name)) == expected_count(name)


def test_tr_release_contains_only_required_fields():
    item = load_items("nudity-tr", limit=1)[0]
    assert item.target_text
    assert item.target_term
    assert not item.item_id.startswith("/")
