import os
import pathlib
import difflib
import re
import sys
import platform
import pytest

is_unsupported = platform.python_implementation() == 'PyPy' or sys.version_info < (3, 10)
skip_on_unsupported = pytest.mark.skipif(
    is_unsupported, reason="Stub generation is only tested on CPython >= 3.10.0")

def remove_platform_dependent(s):
    '''Remove platform-dependent functions from the stubs'''
    s2 = []
    i = 0
    while i < len(s):
        v = s[i]
        if v.strip().startswith('float16'):
            i += 1
        elif v == 'import mlx.core':
            s2.append('import mlx')
            i += 1
        elif v.startswith('def ret_numpy_half()') or \
           v.startswith('def test_slots()') or \
           v.startswith('TypeAlias'):
            i += 2
        else:
            s2.append(v)
            i += 1
    return s2


ref_paths = list(pathlib.Path(__file__).parent.rglob('*.pyi.ref'))
ref_path_ids = [p.name[:-len('.pyi.ref')] for p in ref_paths]
assert len(ref_paths) > 0, "Stub reference files not found!"

@skip_on_unsupported
@pytest.mark.parametrize('p_ref', ref_paths, ids=ref_path_ids)
def test01_check_stub_refs(p_ref, request):
    """
    Check that generated stub files match reference input
    """
    if not request.config.getoption('enable-slow-tests') and any(
        (x in p_ref.name for x in ['jax', 'tensorflow'])):
        pytest.skip("skipping because slow tests are not enabled")

    p_in = p_ref.with_suffix('')
    with open(p_ref, 'r') as f:
        s_ref = f.read().split('\n')
    with open(p_in, 'r') as f:
        s_in = f.read().split('\n')

    if "test_functions_ext" in p_in.name and sys.version_info < (3, 13):
        s_ref = [line.replace("types.CapsuleType", "CapsuleType") for line in s_ref]
        s_ref.insert(5, "")
        s_ref.insert(6, "from typing_extensions import CapsuleType")

    if "test_enum_ext" in p_in.name and sys.version_info < (3, 11):
        # fallback to Python 3.10's `(str, Enum)` MRO.
        s_ref = [line.replace("(enum.StrEnum)", "(str, enum.Enum)") for line in s_ref]

    if "test_typing_ext" in p_in.name and sys.version_info < (3, 13):
        # The 'T4'/'T5' bindings from test_typing.cpp carry PEP 696 'default='
        # values (and pull in the Unpack/TypeVarTuple imports) that only exist
        # on Python 3.13+, which the reference file captures and the 3.13+ CI
        # job checks exactly. On older interpreters, drop those lines and the
        # 'from typing import' block from both the generated and reference text
        # -- collapsing the blank gaps -- so the rest still diffs cleanly.
        def strip(lines):
            out, i = [], 0
            while i < len(lines):
                l = lines[i]
                if l.startswith("from typing import"):
                    if l.endswith("("):  # skip the rest of a wrapped block
                        while lines[i].strip() != ")":
                            i += 1
                elif l.startswith(("T4 = TypeVar", "T5 = TypeVarTuple")):
                    pass
                elif not (l == "" and (not out or out[-1] == "")):
                    out.append(l)
                i += 1
            return out

        s_in, s_ref = strip(s_in), strip(s_ref)

    s_in = remove_platform_dependent(s_in)
    s_ref = remove_platform_dependent(s_ref)

    diff = list(difflib.unified_diff(
        s_ref,
        s_in,
        fromfile=str(p_ref),
        tofile=str(p_in)
    ))
    if len(diff):
        for p in diff:
            print(p.rstrip(), file=sys.stderr)
        print(
            '\nWarning: generated stubs do not match their references. If you\n'
            'intentionally changed a test suite extension, it may be necessary\n'
            'to replace the .pyi.ref file with the generated .pyi file. But\n'
            'please double-check that the change makes sense.',
            file=sys.stderr
        )
        assert False


def parse_imports(text):
    '''Map module name -> set of names imported from it in a stub'''
    imports = {}
    lines = text.split('\n')
    i = 0
    while i < len(lines):
        m = re.match(r'from ([\w.]+) import (.*)', lines[i])
        if m:
            names = imports.setdefault(m.group(1), set())
            rest = m.group(2).strip()
            if rest == '(':
                i += 1
                while lines[i].strip() != ')':
                    names.add(lines[i].strip().rstrip(',').split(' as ')[0])
                    i += 1
            else:
                for n in rest.split(','):
                    names.add(n.strip().split(' as ')[0])
        i += 1
    return imports


def typing_post_39():
    """
    typing.* members that do not exist on Python 3.9 and hence must not be
    imported from 'typing' in a stub generated with --min-python-version 3.9:
    the post-3.9 part of stubgen's TYPING_INTRODUCED version table (plus
    'CapsuleType', which lives in 'types'/'typing_extensions')
    """
    import importlib.util
    here = pathlib.Path(__file__).parent
    for candidate in (here / 'stubgen.py', here.parents[1] / 'src' / 'stubgen.py'):
        if candidate.exists():
            spec = importlib.util.spec_from_file_location('nb_stubgen', candidate)
            stubgen = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(stubgen)
            introduced = stubgen.TYPING_INTRODUCED
            return {n for n, v in introduced.items() if v > (3, 9)} | {'CapsuleType'}
    raise RuntimeError('Could not locate stubgen.py!')

# 'typing.__all__' as of Python 3.9 (a fixed historical fact; members
# removed by later versions, e.g. 'ByteString', are unproblematic here)
TYPING_BASELINE_39 = {
    'AbstractSet', 'Annotated', 'Any', 'AnyStr', 'AsyncContextManager',
    'AsyncGenerator', 'AsyncIterable', 'AsyncIterator', 'Awaitable',
    'BinaryIO', 'ByteString', 'Callable', 'ChainMap', 'ClassVar',
    'Collection', 'Container', 'ContextManager', 'Coroutine', 'Counter',
    'DefaultDict', 'Deque', 'Dict', 'Final', 'ForwardRef', 'FrozenSet',
    'Generator', 'Generic', 'Hashable', 'IO', 'ItemsView', 'Iterable',
    'Iterator', 'KeysView', 'List', 'Literal', 'Mapping', 'MappingView',
    'Match', 'MutableMapping', 'MutableSequence', 'MutableSet', 'NamedTuple',
    'NewType', 'NoReturn', 'Optional', 'OrderedDict', 'Pattern', 'Protocol',
    'Reversible', 'Sequence', 'Set', 'Sized', 'SupportsAbs', 'SupportsBytes',
    'SupportsComplex', 'SupportsFloat', 'SupportsIndex', 'SupportsInt',
    'SupportsRound', 'Text', 'TextIO', 'Tuple', 'Type', 'TypeVar',
    'TypedDict', 'Union', 'ValuesView', 'TYPE_CHECKING', 'cast', 'final',
    'get_args', 'get_origin', 'get_type_hints', 'no_type_check',
    'no_type_check_decorator', 'overload', 'runtime_checkable',
}


@skip_on_unsupported
def test02_min_python_typing():
    """
    Stubs generated with MIN_PYTHON_VERSION 3.9 must not import 'typing' members
    that only exist on newer Python versions
    """
    p = pathlib.Path(__file__).parent / 'test_typing_ext_min_39.pyi'
    imports = parse_imports(p.read_text())

    typing_names = imports.get('typing', set())
    te_names = imports.get('typing_extensions', set())

    assert not (typing_names & typing_post_39())
    # The typing test extension exercises at least these newer constructs
    assert 'TypeAlias' in te_names
    assert 'Self' in te_names


@skip_on_unsupported
def test03_min_python_capsule():
    """
    With MIN_PYTHON_VERSION 3.9, 'CapsuleType' must come from 'typing_extensions'
    regardless of the Python version the extension was compiled for
    """
    p = pathlib.Path(__file__).parent / 'test_functions_ext_min_39.pyi'
    text = p.read_text()

    assert 'types.CapsuleType' not in text
    assert 'CapsuleType' in parse_imports(text).get('typing_extensions', set())


@skip_on_unsupported
def test04_min_python_str_enum():
    """
    With MIN_PYTHON_VERSION 3.9, string-valued enumerations must derive from
    '(str, enum.Enum)' rather than 'enum.StrEnum' (which requires 3.11+)
    """
    p = pathlib.Path(__file__).parent / 'test_enum_ext_min_39.pyi'
    text = p.read_text()

    assert 'StrEnum' not in text
    assert 'class Color(str, enum.Enum):' in text


@skip_on_unsupported
def test05_typing_introduced_is_complete():
    """
    Guard against new Python versions introducing 'typing' members that
    stubgen's version table does not know about, which would make stubs
    generated with --min-python-version silently claim such members exist
    in 'typing' on older versions.
    """
    import typing
    unknown = set(typing.__all__) - TYPING_BASELINE_39 - typing_post_39()
    assert not unknown, (
        f'This Python version provides typing members that stubgen does not '
        f'know the introduction version of: {sorted(unknown)}. Add them to '
        f'TYPING_INTRODUCED in src/stubgen.py.'
    )
