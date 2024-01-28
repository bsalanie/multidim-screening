from multidim_screening.main import main


def test_main():
    assert main("Bernard") == "Hello from main, Bernard"
