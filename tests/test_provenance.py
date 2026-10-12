"""Checkpoint provenance: every save records its lineage, and it can be audited."""

import pytest
import torch

from conftest import tiny_config
from model import provenance
from model.framer import FramerModel
from model.utils.helpers import load_checkpoint, save_checkpoint


@pytest.fixture(autouse=True)
def fresh_run():
    provenance.reset()
    yield
    provenance.reset()


def _model(seed=0):
    torch.manual_seed(seed)
    return FramerModel(tiny_config())


def _save(model, path):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    save_checkpoint(model, optimizer, step=1, loss=1.0, path=str(path))
    return torch.load(path, weights_only=False)


def test_fingerprints_follow_the_weights():
    a, b = _model(0), _model(1)
    assert provenance.fingerprint(a.state_dict()) == provenance.fingerprint(_model(0).state_dict())
    assert provenance.fingerprint(a.state_dict()) != provenance.fingerprint(b.state_dict())
    half = {k: v.to(torch.bfloat16) for k, v in a.state_dict().items()}
    assert len(provenance.fingerprint(half)) == 64


def test_a_run_that_loaded_nothing_is_a_root(tmp_path):
    payload = _save(_model(), tmp_path / "a.pt")
    assert payload["provenance"] == {"origin": "framerai", "version": 1, "parents": []}
    assert provenance.verify(payload["provenance"])["status"] == "scratch"


def test_a_resumed_run_carries_its_parent(tmp_path):
    first = _save(_model(), tmp_path / "a.pt")
    provenance.reset()

    model = _model(5)
    load_checkpoint(str(tmp_path / "a.pt"), model=model, role="resume")
    second = _save(model, tmp_path / "b.pt")

    parent = second["provenance"]["parents"][0]
    assert parent["role"] == "resume"
    assert parent["fingerprint"] == provenance.fingerprint(first["model_state_dict"])
    report = provenance.verify(second["provenance"])
    assert report["status"] == "scratch" and report["depth"] == 1 and report["roots"] == 1


def test_a_teacher_is_counted(tmp_path):
    _save(_model(), tmp_path / "teacher.pt")
    provenance.reset()
    load_checkpoint(str(tmp_path / "teacher.pt"), model=_model(), role="teacher")
    student = _save(_model(2), tmp_path / "student.pt")
    report = provenance.verify(student["provenance"])
    assert report["teachers"] == 1 and report["status"] == "scratch"


def test_weights_from_another_origin_are_refused(tmp_path):
    model = _model()
    torch.save({
        "model_state_dict": model.state_dict(),
        "provenance": {"origin": "someone-else", "parents": []},
    }, tmp_path / "foreign.pt")
    with pytest.raises(provenance.ProvenanceError, match="another model"):
        load_checkpoint(str(tmp_path / "foreign.pt"), model=_model(), role="teacher")
    assert provenance.record()["parents"] == []


def test_a_foreign_ancestor_marks_the_whole_lineage():
    record = {"origin": "framerai", "parents": [
        {"role": "init", "fingerprint": "x", "provenance": {"origin": "other", "parents": []}},
    ]}
    assert provenance.verify(record)["status"] == "foreign"


def test_a_legacy_ancestor_is_unrecorded(tmp_path):
    model = _model()
    torch.save({"model_state_dict": model.state_dict()}, tmp_path / "old.pt")
    load_checkpoint(str(tmp_path / "old.pt"), model=_model(), role="resume")
    payload = _save(model, tmp_path / "new.pt")
    assert provenance.verify(payload["provenance"])["status"] == "unrecorded"
    assert provenance.verify(None)["status"] == "unrecorded"


def test_loading_the_same_weights_twice_records_them_once(tmp_path):
    _save(_model(), tmp_path / "a.pt")
    provenance.reset()
    for _ in range(2):
        load_checkpoint(str(tmp_path / "a.pt"), model=_model(), role="resume")
    assert len(provenance.record()["parents"]) == 1
