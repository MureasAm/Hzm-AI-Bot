"""context_probe 时间/农历/天气感知的单元测试（天气/geo 全 mock）。"""
from datetime import datetime

from src.plugins.chatbot import context_probe as cp


class TestFormatLunar:
    def test_lunar_date(self):
        # 2026-08-10 = 农历六月廿八（实测值）
        s = cp._format_lunar(datetime(2026, 8, 10))
        assert "农历" in s and "六月廿八" in s

    def test_jieqi_and_festival_detected(self):
        # 2026-08-07 = 农历六月廿五 五谷母节 立秋（实测值）
        s = cp._format_lunar(datetime(2026, 8, 7))
        assert "立秋" in s
        assert "五谷母节" in s


class _FakeResp:
    def __init__(self, payload: str):
        self.text = payload


class TestResolveLocation:
    def test_digit_returns_as_is(self, monkeypatch):
        monkeypatch.setattr(cp, "_location_cache", {})
        assert cp._resolve_location("101280101") == "101280101"

    def test_name_uses_geo_cache(self, monkeypatch):
        monkeypatch.setattr(cp, "_location_cache",
                            {"广州": {"ts": 9999999999, "id": "101280101"}})

        def _should_not_call(*a, **k):
            raise AssertionError("命中缓存不应发请求")

        monkeypatch.setattr(cp.httpx, "get", _should_not_call)
        assert cp._resolve_location("广州") == "101280101"

    def test_name_geo_lookup(self, monkeypatch):
        monkeypatch.setattr(cp, "_location_cache", {})
        monkeypatch.setattr(cp, "get_weather_key", lambda: "key")
        monkeypatch.setattr(
            cp.httpx, "get",
            lambda *a, **k: _FakeResp('{"code":"200","location":[{"id":"101280101"}]}'),
        )
        assert cp._resolve_location("广州") == "101280101"
        assert cp._location_cache["广州"]["id"] == "101280101"

    def test_name_geo_failure_empty(self, monkeypatch):
        monkeypatch.setattr(cp, "_location_cache", {})
        monkeypatch.setattr(cp, "get_weather_key", lambda: "key")

        def _fail(*a, **k):
            raise RuntimeError("网络错误")

        monkeypatch.setattr(cp.httpx, "get", _fail)
        assert cp._resolve_location("悉尼") == ""


class TestWeatherLine:
    def test_no_config_returns_empty(self, monkeypatch):
        monkeypatch.setattr(cp, "_weather_cache", {})
        monkeypatch.setattr(cp, "get_weather_city", lambda: "")
        monkeypatch.setattr(cp, "get_weather_key", lambda: "")

        def _should_not_call(loc):
            raise AssertionError("未配置时不应发请求")

        monkeypatch.setattr(cp, "_fetch_weather", _should_not_call)
        assert cp._weather_line("") == ""

    def test_cache_hit_does_not_fetch(self, monkeypatch):
        monkeypatch.setattr(cp, "_weather_cache",
                            {"101010100": {"ts": 9999999999, "text": "天气：多云 32℃"}})
        monkeypatch.setattr(cp, "get_weather_key", lambda: "key")

        def _should_not_call(loc):
            raise AssertionError("缓存命中不应重新请求")

        monkeypatch.setattr(cp, "_fetch_weather", _should_not_call)
        assert cp._weather_line("101010100") == "天气：多云 32℃"

    def test_fetch_then_cache(self, monkeypatch):
        monkeypatch.setattr(cp, "_weather_cache", {})
        monkeypatch.setattr(cp, "get_weather_key", lambda: "fake-key")
        calls = []

        def _fake_fetch(loc):
            calls.append(loc)
            return "天气：晴 25℃"

        monkeypatch.setattr(cp, "_fetch_weather", _fake_fetch)
        assert cp._weather_line("101010100") == "天气：晴 25℃"
        assert len(calls) == 1
        assert cp._weather_line("101010100") == "天气：晴 25℃"  # 第二次命中缓存
        assert len(calls) == 1


class TestNowContext:
    def test_contains_label_date_and_lunar(self):
        s = cp.get_now_context()
        assert s.startswith("【当前时间】")
        assert "月" in s and "周" in s and "农历" in s

    def test_per_user_city_weather_injected(self, monkeypatch):
        monkeypatch.setattr(cp, "get_weather_key", lambda: "key")
        monkeypatch.setattr(cp, "_resolve_location", lambda name: "101280101")
        monkeypatch.setattr(cp, "_weather_cache", {})
        monkeypatch.setattr(cp, "_fetch_weather", lambda loc: "天气：多云 32℃")
        s = cp.get_now_context("广州")
        assert "天气：多云 32℃" in s
