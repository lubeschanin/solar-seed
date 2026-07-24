"""Tests for the central endpoint registry (src/solar_seed/endpoints.py)."""

from solar_seed.endpoints import ENDPOINTS, _DEFAULTS, endpoint, load_endpoints


class TestDefaults:
    def test_all_defaults_present_and_http(self):
        for key in (
            "goes_xray", "rtsw_wind", "rtsw_mag", "noaa_alerts",
            "donki_flr", "sdo_latest", "synoptic_base",
        ):
            assert ENDPOINTS[key].startswith("http")

    def test_endpoint_lookup(self):
        assert endpoint("goes_xray") == ENDPOINTS["goes_xray"]

    def test_unknown_key_raises(self):
        try:
            endpoint("no_such_service")
            assert False, "expected KeyError"
        except KeyError:
            pass


class TestOverrides:
    def test_override_single_key(self, tmp_path):
        cfg = tmp_path / "endpoints.toml"
        cfg.write_text('[endpoints]\ngoes_xray = "https://example.test/goes.json"\n')
        urls = load_endpoints(paths=[cfg])
        assert urls["goes_xray"] == "https://example.test/goes.json"
        # untouched keys keep their defaults
        assert urls["rtsw_wind"] == _DEFAULTS["rtsw_wind"]

    def test_later_file_wins(self, tmp_path):
        a = tmp_path / "a.toml"
        b = tmp_path / "b.toml"
        a.write_text('[endpoints]\ngoes_xray = "https://a.test/x"\n')
        b.write_text('[endpoints]\ngoes_xray = "https://b.test/x"\n')
        urls = load_endpoints(paths=[a, b])
        assert urls["goes_xray"] == "https://b.test/x"

    def test_missing_and_invalid_files_fall_back(self, tmp_path):
        bad = tmp_path / "bad.toml"
        bad.write_text("not toml [ ")
        urls = load_endpoints(paths=[tmp_path / "missing.toml", bad])
        assert urls == _DEFAULTS
