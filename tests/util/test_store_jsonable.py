from inspect_ai.util._store import Store, store_changes, store_jsonable


def test_scalar_store_snapshot_tracks_set_and_delete() -> None:
    store = Store({"a": 1, "b": "x"})
    first = store_jsonable(store)
    assert first == {"a": 1, "b": "x"}
    assert store_jsonable(store) == first

    store.set("a", 2)
    assert store_jsonable(store) == {"a": 2, "b": "x"}
    store.delete("b")
    assert store_jsonable(store) == {"a": 2}


def test_scalar_store_snapshot_tracks_direct_data_writes() -> None:
    store = Store({"a": 1})
    store_jsonable(store)
    store._data["a"] = True
    assert store_changes({"a": 1}, store) is not None
    assert store_jsonable(store) == {"a": True}
    store._data["b"] = None
    store_jsonable(store)
    del store._data["b"]
    store._data["c"] = None
    assert list(store_jsonable(store)) == ["a", "c"]


def test_container_values_mutated_in_place_are_detected() -> None:
    items: list[int] = [1]
    store = Store()
    store.set("items", items)
    before = store_jsonable(store)
    items.append(2)
    after = store_jsonable(store)
    assert before == {"items": [1]}
    assert after == {"items": [1, 2]}
    assert store_changes(before, after) is not None


def test_returned_snapshot_is_not_shared() -> None:
    store = Store({"a": 1})
    snapshot = store_jsonable(store)
    snapshot["a"] = 99
    assert store_jsonable(store) == {"a": 1}
