from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent.parent.parent
TESTVEC_ROOT = REPO_ROOT / 'data_testvecs'


def get_testvecs() -> dict[str, bytes]:
    """Loads all of the testvecs defined in the testvec folder.
    Returns them as a lookup of {filename.stem: bytes}.
    """
    retval: dict[str, bytes] = {}
    for path in TESTVEC_ROOT.iterdir():
        if path.suffix == '.clc':
            retval[path.stem] = path.read_bytes()

    return retval
