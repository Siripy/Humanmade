"""Tests for main.py's real-weather fetch helper: a thin, failure-tolerant
wrapper around the Open-Meteo API."""

from __future__ import annotations

import json
import unittest
from io import BytesIO
from unittest.mock import patch

import main


class _FakeResponse(BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


class TestFetchRealWeather(unittest.TestCase):
    def test_parses_a_successful_reading(self):
        body = json.dumps({"current_weather": {"weathercode": 95}}).encode()
        with patch("main.urllib.request.urlopen",
                  return_value=_FakeResponse(body)) as mock_open:
            result = main.fetch_real_weather(51.5, -0.1)
        self.assertEqual(result, "stormy")
        url = mock_open.call_args[0][0]
        self.assertIn("latitude=51.5", url)
        self.assertIn("longitude=-0.1", url)
        self.assertIn("current_weather=true", url)

    def test_maps_sunny_code(self):
        body = json.dumps({"current_weather": {"weathercode": 0}}).encode()
        with patch("main.urllib.request.urlopen",
                  return_value=_FakeResponse(body)):
            self.assertEqual(main.fetch_real_weather(0.0, 0.0), "sunny")

    def test_network_failure_returns_none(self):
        with patch("main.urllib.request.urlopen", side_effect=OSError("no route")):
            self.assertIsNone(main.fetch_real_weather(0.0, 0.0))

    def test_malformed_response_returns_none(self):
        body = json.dumps({"unexpected": "shape"}).encode()
        with patch("main.urllib.request.urlopen",
                  return_value=_FakeResponse(body)):
            self.assertIsNone(main.fetch_real_weather(0.0, 0.0))

    def test_non_json_response_returns_none(self):
        with patch("main.urllib.request.urlopen",
                  return_value=_FakeResponse(b"not json")):
            self.assertIsNone(main.fetch_real_weather(0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
