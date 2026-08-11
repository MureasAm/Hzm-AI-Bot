"""vision 视觉理解模块的单元测试（mock 掉网络与视觉模型）。"""
import base64
import io

from PIL import Image

from src.plugins.chatbot import vision


def _make_valid_png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (4, 4), (255, 0, 0)).save(buf, format="PNG")
    return buf.getvalue()


class _FakeCompletions:
    def __init__(self, result):
        self._result = result
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        msg = type("M", (), {"content": self._result})()
        choice = type("C", (), {"message": msg})()
        return type("R", (), {"choices": [choice]})()


class _FakeClient:
    """模拟 zhipu_client：.chat.completions.create 返回固定描述并记录调用。"""

    def __init__(self, result="动漫风表情包，灰发萌妹举牌子"):
        self.chat = type("Chat", (), {"completions": _FakeCompletions(result)})()

    @property
    def last_call(self):
        return self.chat.completions.calls[-1]


class TestDetectImageMime:
    def test_png(self):
        assert vision._detect_image_mime(b"\x89PNG\r\n\x1a\nxxxx") == "image/png"

    def test_jpeg(self):
        assert vision._detect_image_mime(b"\xff\xd8\xff\xe0xxxx") == "image/jpeg"

    def test_webp(self):
        assert vision._detect_image_mime(b"RIFF\x00\x00\x00\x00WEBPVP8") == "image/webp"

    def test_gif(self):
        assert vision._detect_image_mime(b"GIF89a...") == "image/gif"

    def test_unknown(self):
        assert vision._detect_image_mime(b"not an image") == ""


class TestRefererFor:
    """Referer 按图链域名选择（QQ 用 QQ 域、B站用 B站域、其他不强加）。"""

    def test_qq_multimedia(self):
        assert vision._referer_for(
            "https://multimedia.nt.qq.com.cn/download?appid=1406&fileid=x&rkey=y"
        ) == "https://qun.qq.com/"

    def test_qq_gchat(self):
        assert vision._referer_for("https://gchat.qpic.cn/abc.jpg") == "https://qun.qq.com/"

    def test_bili_hdslb(self):
        assert vision._referer_for("https://i0.hdslb.com/bfs/xxx.jpg") == "https://www.bilibili.com/"

    def test_bili_biliimg(self):
        assert vision._referer_for("https://archive.biliimg.com/xxx.jpg") == "https://www.bilibili.com/"

    def test_other_domain_no_referer(self):
        assert vision._referer_for("https://example.com/a.jpg") == ""


class TestReadImageHeaders:
    """下载请求确实带上对应域的 Referer（且 trust_env=False 保持）。"""

    async def _capture(self, monkeypatch):
        captured = {}

        class _FakeResponse:
            def raise_for_status(self):
                pass

            @property
            def content(self):
                return b"fake-image"

        class _FakeClient:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def get(self, url):
                captured["url"] = url
                return _FakeResponse()

        monkeypatch.setattr(vision.httpx, "AsyncClient", _FakeClient)
        return captured

    async def test_qq_download_uses_qq_referer(self, monkeypatch):
        captured = await self._capture(monkeypatch)
        data = await vision._read_image_bytes(
            "https://multimedia.nt.qq.com.cn/download?appid=1406&fileid=x&rkey=y"
        )
        assert data == b"fake-image"
        assert captured["headers"]["Referer"] == "https://qun.qq.com/"
        assert captured["trust_env"] is False

    async def test_bili_download_uses_bili_referer(self, monkeypatch):
        captured = await self._capture(monkeypatch)
        await vision._read_image_bytes("https://i0.hdslb.com/bfs/xxx.jpg")
        assert captured["headers"]["Referer"] == "https://www.bilibili.com/"

    async def test_other_download_no_referer(self, monkeypatch):
        captured = await self._capture(monkeypatch)
        await vision._read_image_bytes("https://example.com/a.jpg")
        assert "Referer" not in captured["headers"]


class TestDescribeImage:
    async def test_data_uri_passthrough(self):
        client = _FakeClient("一只猫猫")
        desc = await vision.describe_image(client, "data:image/jpeg;base64,AAAA")
        assert desc == "一只猫猫"
        content = client.last_call["messages"][0]["content"]
        image_part = [c for c in content if c["type"] == "image_url"][0]
        assert image_part["image_url"]["url"] == "data:image/jpeg;base64,AAAA"

    async def test_download_then_base64(self, monkeypatch):
        fake_png = _make_valid_png()

        async def _fake_download(src):
            return fake_png

        monkeypatch.setattr(vision, "_read_image_bytes", _fake_download)
        client = _FakeClient("灰发萌妹")
        desc = await vision.describe_image(client, "https://example.com/a.png")
        assert desc == "灰发萌妹"
        content = client.last_call["messages"][0]["content"]
        image_part = [c for c in content if c["type"] == "image_url"][0]
        url = image_part["image_url"]["url"]
        assert url.startswith("data:image/jpeg;base64,")  # 转码统一为 JPEG
        raw = base64.b64decode(url.split(",", 1)[1])
        assert raw[:3] == b"\xff\xd8\xff"  # 解码后是有效 JPEG

    async def test_download_failure_returns_empty(self, monkeypatch):
        def _raise(src):
            raise ValueError("下载失败")

        monkeypatch.setattr(vision, "_read_image_bytes", _raise)
        client = _FakeClient()
        desc = await vision.describe_image(client, "https://example.com/a.jpg")
        assert desc == ""

    async def test_model_failure_returns_empty(self, monkeypatch):
        async def _fake_download(src):
            return b"xx"

        monkeypatch.setattr(vision, "_read_image_bytes", _fake_download)

        class _BrokenCompletions:
            async def create(self, **kwargs):
                raise RuntimeError("视觉模型 4xx")

        client = type("C", (), {"chat": type("Chat", (), {"completions": _BrokenCompletions()})})()
        desc = await vision.describe_image(client, "https://example.com/a.jpg")
        assert desc == ""
