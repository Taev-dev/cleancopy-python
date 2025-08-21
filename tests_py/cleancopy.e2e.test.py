import pytest

from cleancopy._cst import parse
from cleancopy._cst.converter import convert

from cleancopy_testutils import get_testvecs

testdata = get_testvecs()


@pytest.mark.parametrize('filestem,tvec_bytes', testdata.items())
def test_all_testvecs(filestem: str, tvec_bytes: bytes):
    """Parsing test vectors must succeed without error.
    Note that, as of right now, this isn't actually testing any of the
    resulting parse; it just makes sure that nothing errors. As such,
    it isn't a particularly useful test, but it's better than nothing.
    """
    cst_doc = parse(tvec_bytes)
    convert(cst_doc)
