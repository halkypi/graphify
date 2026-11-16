"""Conservative, dependency-free Emacs Lisp source extractor.

A syntax-aware reader, not an evaluator or a complete Elisp interpreter.
Unknown reader constructs and indirect invocation are explicit diagnostics.
Only direct, executable list forms can contribute call relationships.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path

from graphify.extractors.base import (
    _file_stem, _make_id, _read_source_bytes, active_scan_root,
)

_VERSION = "elisp-reader-1"
_DEFINERS = {
    "defun": "function", "cl-defun": "function", "defsubst": "function",
    "defmacro": "macro", "cl-defmacro": "macro",
    "defvar": "variable", "defconst": "variable", "defcustom": "variable",
    "define-minor-mode": "mode", "define-derived-mode": "mode",
    "ert-deftest": "test", "defalias": "alias",
}
# Executable-control constructs have argument lists but are not function calls.
_CONTROL = {
    "progn", "prog1", "prog2", "if", "when", "unless", "and", "or",
    "save-excursion", "save-current-buffer", "save-restriction", "save-match-data",
    "unwind-protect", "condition-case", "ignore-errors", "ignore-error",
    "while", "catch", "throw", "not", "function", "quote", "interactive",
    "eval-when-compile", "eval-and-compile", "with-temp-buffer",
    "with-current-buffer", "with-temp-file", "with-output-to-string",
    "with-silent-modifications", "atomic-change-group", "with-no-warnings",
    "dolist", "dotimes", "let", "let*", "lambda", "setq", "setq-default",
    "setf", "pcase", "pcase-let", "pcase-let*", "cond", "defadvice",
    "declare", "declare-function", "autoload", "require", "provide",
    "add-hook", "remove-hook", "advice-add", "advice-remove",
    "defgroup", "defface", "defclass", "defmethod", "defgeneric",
}
_OPAQUE = {"pcase", "pcase-let", "pcase-let*", "cl-loop", "cl-defstruct", "defadvice", "condition-case"}
_EAGER = {"list", "cons", "car", "cdr", "append", "nconc", "mapcar", "mapc",
          "concat", "format", "message", "error", "propertize", "equal", "eq",
          "file-name-concat", "expand-file-name", "file-name-directory"}


@dataclass(frozen=True)
class Token:
    kind: str
    value: str
    line: int


@dataclass(frozen=True)
class Form:
    kind: str
    value: str
    line: int
    children: tuple["Form", ...] = ()


class ReadFailure(Exception):
    def __init__(self, line: int, description: str):
        self.line = line
        super().__init__(description)


def _tokens(text: str) -> list[Token]:
    out: list[Token] = []
    pos, line, size = 0, 1, len(text)
    special = "()[]'`\",;"
    while pos < size:
        c = text[pos]
        if c.isspace():
            line += c == "\n"
            pos += 1
            continue
        if c == ";":
            end = text.find("\n", pos)
            pos = size if end < 0 else end
            continue
        if text.startswith("#|", pos):
            start, depth = line, 1
            pos += 2
            while pos < size and depth:
                if text.startswith("#|", pos):
                    depth += 1
                    pos += 2
                elif text.startswith("|#", pos):
                    depth -= 1
                    pos += 2
                else:
                    line += text[pos] == "\n"
                    pos += 1
            if depth:
                raise ReadFailure(start, "unclosed block comment")
            continue
        if c == '"':
            start, begin = line, pos
            pos += 1
            closed = False
            while pos < size:
                if text[pos] == "\\":
                    # Escaped newline is a reader continuation; still count it.
                    if pos + 1 < size:
                        line += text[pos + 1] == "\n"
                        pos += 2
                    else:
                        pos += 1
                elif text[pos] == '"':
                    pos += 1
                    closed = True
                    break
                else:
                    line += text[pos] == "\n"
                    pos += 1
            if not closed:
                raise ReadFailure(start, "unclosed string")
            out.append(Token("string", text[begin:pos], start))
            continue
        if text.startswith("#'", pos):
            out.append(Token("function_quote", "#'", line))
            pos += 2
            continue
        if text.startswith("#.", pos) or text.startswith("#s(", pos) or text.startswith("#(", pos) or text.startswith("#&", pos):
            width = 2 if text.startswith(("#.", "#s(", "#&"), pos) else 1
            out.append(Token("unsupported_reader", text[pos:pos + width], line))
            pos += width
            continue
        if c == "?" and pos + 1 < size and text[pos + 1] in special:
            out.append(Token("atom", text[pos:pos + 2], line))
            pos += 2
            continue
        if c in "()[]'`":
            kinds = {"(": "open", ")": "close", "[": "vopen", "]": "vclose",
                     "'": "quote", "`": "quasiquote"}
            out.append(Token(kinds[c], c, line))
            pos += 1
            continue
        if c == ",":
            size_tok = 2 if text.startswith(",@", pos) else 1
            out.append(Token("unquote", text[pos:pos + size_tok], line))
            pos += size_tok
            continue
        begin, start = pos, line
        # Elisp symbols may escape delimiters; keep the escaped character in atom.
        while pos < size and not text[pos].isspace() and text[pos] not in special:
            if text[pos] == "\\" and pos + 1 < size:
                pos += 2
            else:
                pos += 1
        if pos == begin:
            raise ReadFailure(line, f"unexpected character: {c!r}")
        out.append(Token("atom", text[begin:pos], start))
    return out


def _read(text: str) -> list[Form]:
    toks, at = _tokens(text), 0

    def parse(depth: int = 0) -> Form:
        nonlocal at
        if depth > 256:
            raise ReadFailure(toks[at].line, "reader nesting exceeds 256")
        if at >= len(toks):
            raise ReadFailure(toks[-1].line if toks else 1, "missing expression")
        tok = toks[at]
        at += 1
        if tok.kind in ("open", "vopen"):
            end = "close" if tok.kind == "open" else "vclose"
            children = []
            while at < len(toks) and toks[at].kind != end:
                if toks[at].kind in ("close", "vclose"):
                    raise ReadFailure(toks[at].line, "mismatched delimiter")
                children.append(parse(depth + 1))
            if at >= len(toks):
                raise ReadFailure(tok.line, "unclosed list or vector")
            at += 1
            return Form("list" if tok.kind == "open" else "vector", "", tok.line, tuple(children))
        if tok.kind in ("close", "vclose"):
            raise ReadFailure(tok.line, "unexpected closing delimiter")
        if tok.kind in ("quote", "quasiquote", "unquote", "function_quote", "unsupported_reader"):
            return Form(tok.kind, tok.value, tok.line, (parse(depth + 1),))
        return Form(tok.kind, tok.value, tok.line)

    forms = []
    while at < len(toks):
        forms.append(parse())
    return forms


def _head(form: Form) -> str | None:
    if form.kind == "list" and form.children and form.children[0].kind == "atom":
        return form.children[0].value
    return None


def _symbol(form: Form | None) -> str | None:
    return form.value if form is not None and form.kind == "atom" else None


def _literal_symbol(form: Form | None) -> str | None:
    if form is None:
        return None
    if form.kind in ("quote", "function_quote") and len(form.children) == 1:
        return _symbol(form.children[0])
    return None


def extract_elisp(path: Path) -> dict:
    """Return Graphify-shaped nodes, edges, and non-silent parse diagnostics."""
    path = Path(path)
    str_path = str(path)
    root = active_scan_root()
    try:
        relative = path.resolve().relative_to(root.resolve()) if root else path
    except ValueError:
        relative = path
    rel_path = relative.as_posix()
    # A short path digest prevents same-stem normalization collisions; never
    # include checkout path or source commit in corpus IDs.
    stem = _make_id(_file_stem(relative))
    file_id = _make_id(stem)
    path_salt = sha1(rel_path.encode("utf-8"), usedforsecurity=False).hexdigest()[:10]
    nodes: list[dict] = [{
        "id": file_id, "label": path.name, "file_type": "code",
        "source_file": str_path, "source_location": "L1",
        "node_kind": "file", "language": "elisp",
        "metadata": {"extractor": _VERSION, "relative_path": rel_path},
    }]
    edges: list[dict] = []
    diagnostics: list[dict] = []
    by_id = {file_id: nodes[0]}
    names: dict[str, list[str]] = {}

    def diagnostic(reason: str, line: int, detail: str) -> None:
        diagnostics.append({"reason": reason, "line": line, "source_location": f"L{line}", "detail": detail})

    def node(kind: str, name: str, line: int, *, external: bool = False) -> str:
        name_salt = sha1(name.encode("utf-8"), usedforsecurity=False).hexdigest()[:10]
        nid = _make_id(stem, path_salt, "elisp", kind, name, name_salt)
        if nid not in by_id:
            data = {
                "id": nid, "label": name, "symbol": name,
                "file_type": "code", "node_kind": kind,
                "language": "elisp", "source_file": "" if external else str_path,
                "source_location": "" if external else f"L{line}",
                "metadata": {"extractor": _VERSION, "origin_file": str_path} if external else {"extractor": _VERSION},
            }
            nodes.append(data)
            by_id[nid] = data
        elif not external and not by_id[nid]["source_file"]:
            by_id[nid].update(source_file=str_path, source_location=f"L{line}")
        return nid

    def edge(src: str, tgt: str, relation: str, line: int, confidence: str = "EXTRACTED") -> None:
        edges.append({"source": src, "target": tgt, "relation": relation,
                      "confidence": confidence, "source_file": str_path,
                      "source_location": f"L{line}", "weight": 1.0})

    try:
        forms = _read(_read_source_bytes(path).decode("utf-8"))
    except (ReadFailure, UnicodeDecodeError, OSError) as exc:
        line = exc.line if isinstance(exc, ReadFailure) else 1
        diagnostic("parse-error", line, str(exc))
        nodes[0]["metadata"]["diagnostics"] = diagnostics
        return {"nodes": nodes, "edges": edges, "diagnostics": diagnostics}

    # First pass: collect all declarations so forward references are resolved.
    for form in forms:
        head = _head(form)
        if head not in _DEFINERS or len(form.children) < 2:
            continue
        raw = form.children[1]
        name = _literal_symbol(raw) if head == "defalias" else _symbol(raw)
        if not name:
            diagnostic("unsupported-definition", form.line, f"{head}: dynamic or missing name")
            continue
        kind = _DEFINERS[head]
        nid = node(kind, name, form.line)
        names.setdefault(name, []).append(nid)
        edge(file_id, nid, "defines", form.line)

    def reference(name: str, kind: str, line: int) -> tuple[str, bool]:
        candidates = list(dict.fromkeys(names.get(name, [])))
        if len(candidates) == 1:
            return candidates[0], True
        if len(candidates) > 1:
            diagnostic("ambiguous-symbol", line, f"multiple definitions for {name}")
        return node(kind, name, line, external=True), False

    def executable(form: Form, owner: str) -> None:
        if form.kind in ("atom", "string", "vector", "quote", "quasiquote", "function_quote"):
            return
        if form.kind in ("unsupported_reader", "unquote"):
            diagnostic("unsupported-reader", form.line, form.value)
            return
        if form.kind != "list" or not form.children:
            return
        head = _head(form)
        if not head:
            diagnostic("dynamic-call", form.line, "non-symbol call operator")
            return
        args = form.children[1:]
        if head in _DEFINERS:
            return  # top-level definition is processed separately.
        if head in _OPAQUE:
            diagnostic("opaque-form", form.line, head)
            return
        if head in ("quote", "function", "interactive", "declare", "declare-function", "autoload"):
            return
        if head in ("require", "provide"):
            value = _literal_symbol(args[0]) if args else None
            if value:
                target = node("feature", value, form.line, external=head == "require")
                edge(file_id, target, "requires" if head == "require" else "provides", form.line)
            else:
                diagnostic("dynamic-feature", form.line, head)
            return
        if head in ("add-hook", "remove-hook"):
            hook = _literal_symbol(args[0]) if args else None
            function = _literal_symbol(args[1]) if len(args) > 1 else None
            if hook and function:
                target = node("hook", hook, form.line)
                function_id, resolved = reference(function, "external_function", form.line)
                edge(owner, target, "registers_hook" if head == "add-hook" else "removes_hook", form.line)
                edge(target, function_id, "handler", form.line, "EXTRACTED" if resolved else "AMBIGUOUS")
            else:
                diagnostic("dynamic-hook", form.line, head)
            return
        if head == "advice-add":
            advised = _literal_symbol(args[0]) if args else None
            advisor = _literal_symbol(args[2]) if len(args) > 2 else None
            if advised and advisor:
                advised_id, _ = reference(advised, "external_function", form.line)
                advisor_id, _ = reference(advisor, "external_function", form.line)
                edge(advisor_id, advised_id, "advises", form.line)
            else:
                diagnostic("dynamic-advice", form.line, head)
            return
        if head in ("funcall", "apply"):
            target = _literal_symbol(args[0]) if args else None
            if target:
                target_id, resolved = reference(target, "external_function", form.line)
                edge(owner, target_id, "calls", form.line, "EXTRACTED" if resolved else "AMBIGUOUS")
            else:
                diagnostic("dynamic-call", form.line, head)
            for arg in args[1:]:
                executable(arg, owner)
            return
        if head in ("let", "let*"):
            if args and args[0].kind == "list":
                for binding in args[0].children:
                    if binding.kind == "list":
                        for initial in binding.children[1:]:
                            executable(initial, owner)
            for expression in args[1:]:
                executable(expression, owner)
            return
        if head in ("lambda", "dolist", "dotimes"):
            for expression in args[1:]:
                executable(expression, owner)
            return
        if head in ("setq", "setq-default", "setf"):
            for value in args[1::2]:
                executable(value, owner)
            return
        if head == "cond":
            for clause in args:
                if clause.kind == "list":
                    for expression in clause.children:
                        executable(expression, owner)
            return
        if head not in _CONTROL and not head.startswith(":"):
            target_id, resolved = reference(head, "external_function", form.line)
            kind = by_id[target_id].get("node_kind")
            relation = "invokes_macro" if kind == "macro" else "calls"
            edge(owner, target_id, relation, form.line, "EXTRACTED" if resolved else "AMBIGUOUS")
            if resolved and by_id[owner].get("node_kind") == "test" and kind in ("function", "mode"):
                edge(owner, target_id, "tests", form.line)
            if kind == "macro" or (not resolved and head not in _EAGER):
                # Unknown macros choose which arguments execute. A direct form
                # witness is retained, but nested forms are not asserted.
                return
        for arg in args:
            executable(arg, owner)

    for form in forms:
        head = _head(form)
        if head in _DEFINERS and len(form.children) >= 2:
            raw = form.children[1]
            name = _literal_symbol(raw) if head == "defalias" else _symbol(raw)
            if not name or name not in names:
                continue
            owner = names[name][0]
            if head == "defalias":
                target = _literal_symbol(form.children[2]) if len(form.children) > 2 else None
                if target:
                    dest, resolved = reference(target, "external_function", form.line)
                    edge(owner, dest, "aliases", form.line, "EXTRACTED" if resolved else "AMBIGUOUS")
                else:
                    diagnostic("dynamic-alias", form.line, name)
            elif head in ("defvar", "defconst", "defcustom"):
                if len(form.children) > 2:
                    executable(form.children[2], owner)
            elif head in ("define-minor-mode", "define-derived-mode"):
                # Mode header arguments are declarative; no body-call claim.
                diagnostic("opaque-form", form.line, head + " header/body")
            else:
                # Skip declared argument lists, docstrings, and ERT empty params.
                body = form.children[3:]
                for part in body:
                    executable(part, owner)
        else:
            executable(form, file_id)

    nodes[0]["metadata"]["diagnostics"] = diagnostics
    return {"nodes": nodes, "edges": edges, "diagnostics": diagnostics}
