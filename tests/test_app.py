"""Headless smoke test of the Streamlit app using Streamlit's AppTest harness."""

import pytest
from conftest import requires_model

pytestmark = requires_model


@pytest.fixture(scope="module")
def app():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("../app/streamlit_app.py", default_timeout=120).run()
    assert not at.exception
    return at


def test_renders_all_tabs(app):
    assert len(app.tabs) == 3
    labels = [m.label for m in app.metric]
    assert "Risk tier" in labels and "ROC-AUC" in labels


def test_presets_and_dependent_fields(app):
    app.sidebar.button[1].click().run()
    assert not app.exception
    app.selectbox(key="InternetService").set_value("No").run()
    assert not app.exception
    assert app.selectbox(key="OnlineSecurity").value == "No internet service"
    app.selectbox(key="PhoneService").set_value("No").run()
    assert not app.exception
    assert app.selectbox(key="MultipleLines").value == "No phone service"
    assert not app.error
