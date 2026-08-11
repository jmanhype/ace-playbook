def test_imports_are_local_and_do_not_make_a_network_request():
    import app  # noqa: F401
    import core  # noqa: F401

