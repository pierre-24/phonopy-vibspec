import pathlib
import pytest


@pytest.fixture
def context_SiO2(monkeypatch):
    dir_ = pathlib.Path(__file__).parent / 'tests_files/SiO2'
    monkeypatch.chdir(dir_)
