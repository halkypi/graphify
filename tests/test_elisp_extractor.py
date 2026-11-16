"""Synthetic Emacs Lisp source-extraction acceptance tests (test-first)."""
from pathlib import Path
from graphify.detect import FileType, classify_file
from graphify.extract import _get_extractor
from graphify.extractors.elisp import extract_elisp
from graphify.validate import validate_extraction


def _scan(tmp_path, source, name="fixture.el"):
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return extract_elisp(path)


def _nodes(data, kind):
    return [n for n in data["nodes"] if n.get("node_kind") == kind]


def _edges(data, relation):
    return [e for e in data["edges"] if e["relation"] == relation]


def test_elisp_detection_dispatch_and_registry():
    from graphify.extractors import LANGUAGE_EXTRACTORS
    assert classify_file(Path("x.el")) == FileType.CODE
    assert _get_extractor(Path("x.el")) is extract_elisp
    assert LANGUAGE_EXTRACTORS["elisp"] is extract_elisp


def test_elisp_definitions_and_ert(tmp_path):
    data = _scan(tmp_path, """;;; note
(defun alpha () (beta))
(cl-defun beta (&optional x) x)
(defsubst speedy () (beta))
(defmacro compile-it (x) `(progn ,x))
(defvar count 0)
(defconst value 1)
(defcustom other nil "Doc" :type 'boolean)
(define-minor-mode toggle-mode "Toggle")
(define-derived-mode child-mode fundamental-mode "Child")
(ert-deftest alpha-test () (alpha))
""")
    expected = {
        "alpha": "function", "beta": "function", "speedy": "function",
        "compile-it": "macro", "count": "variable", "value": "variable",
        "other": "variable", "toggle-mode": "mode",
        "child-mode": "mode", "alpha-test": "test",
    }
    by_name = {n["symbol"]: n for n in data["nodes"] if n.get("symbol") in expected}
    assert {name: by_name[name]["node_kind"] for name in expected} == expected
    assert by_name["alpha"]["source_location"] == "L2"
    assert by_name["alpha-test"]["source_location"] == "L11"
    assert any(e["source"] == by_name["alpha"]["id"] and e["target"] == by_name["beta"]["id"] for e in _edges(data, "calls"))
    assert any(e["source"] == by_name["alpha-test"]["id"] and e["target"] == by_name["alpha"]["id"] for e in _edges(data, "tests"))


def test_elisp_no_false_calls_from_data(tmp_path):
    data = _scan(tmp_path, """(defun foo (target)
  ;; (ghost)
  "(ghost)"
  '(ghost)
  `(ghost ,(ghost))
  #'target
  (target)
  (funcall dynamic-fn))
(defun target () t)
""")
    calls = _edges(data, "calls")
    assert len(calls) == 1
    assert calls[0]["source_location"] == "L7"
    assert not any(n.get("symbol") == "ghost" for n in data["nodes"])
    assert any(d["reason"] == "dynamic-call" for d in data["diagnostics"])


def test_elisp_hooks_advice_features(tmp_path):
    data = _scan(tmp_path, """(require 'subr-x)
(provide 'pkg)
(defun on-save () t)
(defun wrapper (f &rest args) (apply f args))
(add-hook 'after-save-hook #'on-save)
(advice-add 'save-buffer :around #'wrapper)
""")
    assert len(_edges(data, "requires")) == 1
    assert len(_edges(data, "provides")) == 1
    assert len(_edges(data, "registers_hook")) == 1
    assert len(_edges(data, "advises")) == 1
    assert any(n.get("symbol") == "subr-x" for n in _nodes(data, "feature"))


def test_elisp_unsupported_reader_and_malformed_fail_closed(tmp_path):
    data = _scan(tmp_path, """(defun live () (funcall maybe))
'(add-hook 'h #'live)
#.(malicious)
(defun good () (live))
""")
    assert not _edges(data, "registers_hook")
    assert any(d["reason"] == "unsupported-reader" for d in data["diagnostics"])
    assert len(_edges(data, "calls")) == 1
    broken = _scan(tmp_path, "(defun x () (foo)", "broken.el")
    assert not _nodes(broken, "function")
    assert not _edges(broken, "calls")
    assert any(d["reason"] == "parse-error" for d in broken["diagnostics"])


def test_elisp_identity_alias_macro_and_schema(tmp_path):
    left = _scan(tmp_path, "(defun duplicate () t)", "a/pkg.el")
    right = _scan(tmp_path, "(defun duplicate () t)", "b/pkg.el")
    assert _nodes(left, "function")[0]["id"] != _nodes(right, "function")[0]["id"]
    data = _scan(tmp_path, """(defmacro thing () nil)
(defalias 'alias-thing #'thing)
(defun use () (thing) (unknown))
""")
    assert len(_edges(data, "aliases")) == 1
    assert len(_edges(data, "invokes_macro")) == 1
    assert any(e["confidence"] == "AMBIGUOUS" for e in data["edges"])
    assert validate_extraction(data) == []
