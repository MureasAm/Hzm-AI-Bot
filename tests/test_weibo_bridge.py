"""weibo_bridge.py 单元测试。

纯函数（cookie 解析 / 配图提取 / mid→bid）+ **推送侧逻辑**（镜像 test_bili_bridge.py）。

之前这里只有 3 个纯函数、37 行——推送到好友、去重、基线、白名单这些全没覆盖，
而 bili 那边有 347 行把这些钉死了。两个模块结构几乎一样（还共用 _bridge_common），
等于同一个功能一边有安全网、一边裸奔。这里补齐。

所有外部调用（微博接口、好友列表、私聊发送、状态写入、配图下载）全部打桩。
"""
import pytest

from src.plugins.chatbot import weibo_bridge as wb


class TestParseCookie:
    def test_basic(self):
        assert wb._parse_cookie_str("SUB=abc; ULV=1; WBPSESS=x=y") == {
            "SUB": "abc", "ULV": "1", "WBPSESS": "x=y"}

    def test_empty(self):
        assert wb._parse_cookie_str("") == {}


class TestExtractImages:
    def test_pic_infos_original_preferred(self):
        item = {"pic_ids": ["a", "b"],
                "pic_infos": {
                    "a": {"large": {"url": "https://x/orj960/a.jpg"},
                          "original": {"url": "https://x/large/a_org.jpg"}},
                    "b": {"large": {"url": "https://x/orj960/b.jpg"}}}}
        assert wb._extract_post_images(item) == [
            "https://x/large/a_org.jpg", "https://x/orj960/b.jpg"]

    def test_old_pics_fallback(self):
        item = {"pics": [{"large": {"url": "https://y/1.jpg"}}, {"url": "https://y/2.jpg"}]}
        assert wb._extract_post_images(item) == ["https://y/1.jpg", "https://y/2.jpg"]

    def test_no_images(self):
        assert wb._extract_post_images({}) == []


class TestMidToBid:
    def test_known_value_stable(self):
        # 微博 mid→bid 是 base36，给个确定值验证不抛且稳定
        assert wb._mid_to_bid("1617118739000") == wb._mid_to_bid("1617118739000")
        assert wb._mid_to_bid("0") == "0"
        assert wb._mid_to_bid("abc") == ""  # 非数字


class TestExtractPostText:
    def test_raw_text_preferred(self):
        assert wb._extract_post_text({"raw_text": "原文", "text": "<b>别的</b>"}) == "原文"

    def test_html_stripped(self):
        assert wb._extract_post_text({"text": "<a href='x'>正文</a> 尾巴"}) == "正文 尾巴"

    def test_empty(self):
        assert wb._extract_post_text({}) == ""


# ==================== 推送侧 ====================

class FakeBot:
    """模拟 OneBot Bot：记录私聊发送，返回固定好友列表。"""

    def __init__(self, friends=("111", "222", "333")):
        self.friends = [{"user_id": f} for f in friends]
        self.sent = []

    async def get_friend_list(self):
        return self.friends

    async def send_private_msg(self, user_id=None, message=None):
        self.sent.append((user_id, message))


def _make_monitor(**attrs):
    m = wb.WeiboMonitor()
    for k, v in attrs.items():
        setattr(m, k, v)
    return m


def _patch(monkeypatch, m, post_id="p1", images=None, cookie="SUB=x", whitelist=None):
    """通用打桩：配好 cookie、拉取返回指定微博、状态不落盘、配图下载直接返回假路径。"""
    monkeypatch.setattr(wb, "get_weibo_cookie", lambda: cookie)
    monkeypatch.setattr(wb, "get_notify_whitelist", lambda: whitelist)
    monkeypatch.setattr(wb, "_save_state", lambda s: None)

    async def fake_fetch():
        return {"id": post_id, "text": "今天好累", "image_urls": images or [],
                "url": "https://weibo.com/u/1/abc"}

    monkeypatch.setattr(m, "_fetch_latest_post", fake_fetch)
    downloaded = []

    async def fake_download(url):
        downloaded.append(url)
        return f"/tmp/{url.rsplit('/', 1)[-1]}"

    monkeypatch.setattr(wb, "_download_image", fake_download)
    return downloaded


class TestNewPostPush:
    async def test_new_post_pushes_once_and_records(self, monkeypatch):
        m = _make_monitor(uid="1", state={}, _primed=True)
        _patch(monkeypatch, m, post_id="p1")
        pushed = []

        async def fake_push(bot, content, image_paths=None):
            pushed.append(content)

        monkeypatch.setattr(m, "_push", fake_push)
        bot = FakeBot()
        await m._check_posts(bot)
        assert len(pushed) == 1
        assert m.state["last_post_id"] == "p1"
        assert "灰泽满刚刚发了微博" in pushed[0]

        # 同一条再来 → 不重复推
        await m._check_posts(bot)
        assert len(pushed) == 1

    async def test_same_id_in_state_not_pushed(self, monkeypatch):
        m = _make_monitor(uid="1", state={"last_post_id": "p1"}, _primed=True)
        _patch(monkeypatch, m, post_id="p1")
        pushed = []

        async def fake_push(bot, content, image_paths=None):
            pushed.append(content)

        monkeypatch.setattr(m, "_push", fake_push)
        await m._check_posts(FakeBot())
        assert pushed == []

    async def test_empty_id_skipped(self, monkeypatch):
        m = _make_monitor(uid="1", state={}, _primed=True)
        _patch(monkeypatch, m, post_id="")
        pushed = []

        async def fake_push(bot, content, image_paths=None):
            pushed.append(content)

        monkeypatch.setattr(m, "_push", fake_push)
        await m._check_posts(FakeBot())
        assert pushed == []


class TestPostDeletionDedup:
    """删博后接口"最新"退回旧博，不该重推（水位线判据，镜像 bili 的撤回场景）。

    踩坑：状态只记一条 id、判据是"和上一条不同"。删掉最新那条后，接口的"最新"
    退回上一条（更旧、id 更小），与记的不同 → 被判成新微博 → 把旧的又推一遍。
    """

    def _monitor(self, monkeypatch, post_id, state):
        m = _make_monitor(uid="1", state=state, _primed=True)
        _patch(monkeypatch, m, post_id=post_id)
        monkeypatch.setattr(wb, "_save_state", lambda s: None)
        monkeypatch.setattr(wb, "get_notify_whitelist", lambda: None)
        return m

    async def test_newer_post_pushes(self, monkeypatch):
        m = self._monitor(monkeypatch, "5342326297725556", {"last_post_id": "5342134488270833"})
        bot = FakeBot(friends=("111",))
        await m._check_posts(bot)
        assert len(bot.sent) == 1
        assert m.state["last_post_id"] == "5342326297725556"

    async def test_deleted_newest_does_not_repush(self, monkeypatch):
        m = self._monitor(monkeypatch, "5342134488270833", {"last_post_id": "5342326297725556"})
        bot = FakeBot(friends=("111",))
        await m._check_posts(bot)
        assert bot.sent == [], "删博后回退到的旧微博不该重推"
        assert m.state["last_post_id"] == "5342326297725556"   # 水位线不降

    async def test_prime_does_not_lower_watermark(self, monkeypatch):
        m = self._monitor(monkeypatch, "5342134488270833", {"last_post_id": "5342326297725556"})
        await m._prime()
        assert m.state["last_post_id"] == "5342326297725556"

    async def test_prime_raises_watermark_to_newest(self, monkeypatch):
        # 停机期间发了新博 → 基线对齐到它，不推送（原本的"不补推停机期间事件"语义）
        m = self._monitor(monkeypatch, "5342326297725556", {"last_post_id": "5342134488270833"})
        await m._prime()
        assert m.state["last_post_id"] == "5342326297725556"


class TestFetchErrorMessages:
    """拉取失败时的报错要能指向真正的原因（踩过两次坑）。"""

    async def _fetch_with(self, monkeypatch, payload, status=200, resp_headers=None):
        m = _make_monitor(uid="1", state={})
        monkeypatch.setattr(wb, "_load_jar", lambda: {})
        monkeypatch.setattr(wb, "_save_jar", lambda c: None)
        hdrs = resp_headers or {"content-type": "application/json"}

        class _Resp:
            status_code = status
            headers = hdrs   # 类体里不能引用同名的外层变量，换个名字
            cookies = {}     # _save_jar(resp.cookies) 会读它

            def json(self):
                return payload

        class _Client:
            def __init__(self, **kw):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def get(self, url, params=None):
                return _Resp()

        monkeypatch.setattr(wb.httpx, "AsyncClient", _Client)
        with pytest.raises(RuntimeError) as e:
            await m._fetch_post_once()
        return str(e.value)

    async def test_cookie_expired_detected_from_body_url(self, monkeypatch):
        # 实测形态：HTTP 200 + {"ok": -100, "url": "https://weibo.com/login.php?..."}
        # 登录态在 body 的 url 里，不在 HTTP location 头里——只查 location 会漏判。
        msg = await self._fetch_with(monkeypatch, {
            "ok": -100, "url": "https://weibo.com/login.php?url=https%3A%2F%2Fweibo.com%2Fu%2F1"})
        assert "Cookie 已失效" in msg
        assert "登录" in msg, "这句要能被 _check_posts 的 cookie 分支认出（它查 '登录'/'login'）"

    async def test_other_error_code_kept_verbatim(self, monkeypatch):
        msg = await self._fetch_with(monkeypatch, {"ok": -200, "msg": "别的错误"})
        assert "-200" in msg and "别的错误" in msg

    async def test_empty_list_not_blamed_on_the_uid(self, monkeypatch):
        # 空列表多半是风控限流；说成"该 UID 暂无微博"会把人带偏
        msg = await self._fetch_with(monkeypatch, {"ok": 1, "data": {"list": []}})
        assert "空列表" in msg
        assert "暂无微博" not in msg


class TestNoCookie:
    async def test_no_cookie_skips_fetch_and_push(self, monkeypatch):
        m = _make_monitor(uid="1", state={}, _primed=True)
        _patch(monkeypatch, m, cookie="")

        async def _boom():
            raise AssertionError("没 cookie 不该去拉微博")

        monkeypatch.setattr(m, "_fetch_latest_post", _boom)
        bot = FakeBot()
        await m._check_posts(bot)
        assert bot.sent == []
        assert m._warned_no_cookie is True   # 只提示一次

    async def test_no_uid_skips_poll(self, monkeypatch):
        m = _make_monitor(uid="", state={}, _primed=True)

        async def _boom(bot):
            raise AssertionError("没 uid 不该进 _check_posts")

        monkeypatch.setattr(m, "_check_posts", _boom)
        await m.poll_once(FakeBot())   # 不抛即通过


class TestPrime:
    """基线：首次启动只对齐、不推送；停机期间的新微博不该在重启后被当新事件推。"""

    async def test_first_poll_primes_and_does_not_push(self, monkeypatch):
        m = _make_monitor(uid="1", state={}, _primed=False)
        _patch(monkeypatch, m, post_id="p1")
        pushed = []

        async def fake_push(bot, content, image_paths=None):
            pushed.append(content)

        monkeypatch.setattr(m, "_push", fake_push)
        await m.poll_once(FakeBot())
        assert pushed == [], "首次轮询只建基线，不推送"
        assert m._primed is True
        assert m.state["last_post_id"] == "p1"

    async def test_restart_does_not_push_existing_post(self, monkeypatch):
        # 模拟"停机期间她发了微博，重启后不该补推"
        m = _make_monitor(uid="1", state={"last_post_id": "旧"}, _primed=False)
        _patch(monkeypatch, m, post_id="停机期间发的")
        pushed = []

        async def fake_push(bot, content, image_paths=None):
            pushed.append(content)

        monkeypatch.setattr(m, "_push", fake_push)
        await m.poll_once(FakeBot())
        assert pushed == []

    async def test_new_post_after_prime_pushes(self, monkeypatch):
        m = _make_monitor(uid="1", state={}, _primed=True)
        _patch(monkeypatch, m, post_id="p2")
        pushed = []

        async def fake_push(bot, content, image_paths=None):
            pushed.append(content)

        monkeypatch.setattr(m, "_push", fake_push)
        await m.poll_once(FakeBot())
        assert len(pushed) == 1

    async def test_prime_failure_still_marks_primed(self, monkeypatch):
        """基线拉取失败也不能卡死——否则永远不 _primed、每轮都重试。"""
        m = _make_monitor(uid="1", state={}, _primed=False)
        monkeypatch.setattr(wb, "_save_state", lambda s: None)

        async def boom():
            raise RuntimeError("网络瞬断")

        monkeypatch.setattr(m, "_fetch_latest_post", boom)
        await m._prime()
        assert m._primed is True


class TestPushTargets:
    async def test_broadcast_when_whitelist_empty(self, monkeypatch):
        m = _make_monitor(uid="1", state={})
        monkeypatch.setattr(wb, "get_notify_whitelist", lambda: None)
        monkeypatch.setattr(wb, "_save_state", lambda s: None)
        bot = FakeBot(friends=("111", "222", "333"))
        await m._push(bot, "内容")
        assert [u for u, _ in bot.sent] == ["111", "222", "333"]

    async def test_whitelist_filters(self, monkeypatch):
        m = _make_monitor(uid="1", state={})
        monkeypatch.setattr(wb, "get_notify_whitelist", lambda: ["222"])
        monkeypatch.setattr(wb, "_save_state", lambda s: None)
        bot = FakeBot(friends=("111", "222", "333"))
        await m._push(bot, "内容")
        assert [u for u, _ in bot.sent] == ["222"]

    async def test_whitelist_matching_nobody_sends_nothing(self, monkeypatch):
        m = _make_monitor(uid="1", state={})
        monkeypatch.setattr(wb, "get_notify_whitelist", lambda: ["999"])
        monkeypatch.setattr(wb, "_save_state", lambda s: None)
        bot = FakeBot()
        await m._push(bot, "内容")
        assert bot.sent == []

    async def test_fresh_friend_list_each_push(self, monkeypatch):
        """状态里有旧好友缓存，但每次推送都要用最新列表（新加的好友要能收到）。"""
        m = _make_monitor(uid="1", state={"friends": [{"user_id": "旧"}]})
        monkeypatch.setattr(wb, "get_notify_whitelist", lambda: None)
        monkeypatch.setattr(wb, "_save_state", lambda s: None)
        bot = FakeBot(friends=("新1", "新2"))
        await m._push(bot, "内容")
        assert [u for u, _ in bot.sent] == ["新1", "新2"]

    async def test_one_friend_failure_does_not_stop_others(self, monkeypatch):
        m = _make_monitor(uid="1", state={})
        monkeypatch.setattr(wb, "get_notify_whitelist", lambda: None)
        monkeypatch.setattr(wb, "_save_state", lambda s: None)

        class FlakyBot(FakeBot):
            async def send_private_msg(self, user_id=None, message=None):
                if user_id == "111":
                    raise RuntimeError("发送失败")
                self.sent.append((user_id, message))

        bot = FlakyBot(friends=("111", "222"))
        await m._push(bot, "内容")
        assert [u for u, _ in bot.sent] == ["222"]


class TestImages:
    async def test_images_attached_to_push(self, monkeypatch):
        m = _make_monitor(uid="1", state={}, _primed=True)
        downloaded = _patch(monkeypatch, m, post_id="p1",
                            images=["https://x/1.jpg", "https://x/2.jpg"])
        monkeypatch.setattr(wb, "_save_state", lambda s: None)
        monkeypatch.setattr(wb, "get_notify_whitelist", lambda: None)
        bot = FakeBot(friends=("111",))
        await m._check_posts(bot)
        assert len(downloaded) == 2
        assert len(bot.sent) == 1
        # 文字 + 两张图 = 3 段
        assert len(bot.sent[0][1]) == 3

    async def test_more_than_six_images_capped(self, monkeypatch):
        m = _make_monitor(uid="1", state={}, _primed=True)
        urls = [f"https://x/{i}.jpg" for i in range(8)]
        downloaded = _patch(monkeypatch, m, post_id="p1", images=urls)
        monkeypatch.setattr(wb, "_save_state", lambda s: None)
        monkeypatch.setattr(wb, "get_notify_whitelist", lambda: None)
        await m._check_posts(FakeBot(friends=("111",)))
        assert len(downloaded) == 6

    async def test_image_download_failure_still_sends_text(self, monkeypatch):
        m = _make_monitor(uid="1", state={}, _primed=True)
        _patch(monkeypatch, m, post_id="p1", images=["https://x/1.jpg"])
        monkeypatch.setattr(wb, "_save_state", lambda s: None)
        monkeypatch.setattr(wb, "get_notify_whitelist", lambda: None)

        async def boom(url):
            raise RuntimeError("下载失败")

        monkeypatch.setattr(wb, "_download_image", boom)
        bot = FakeBot(friends=("111",))
        await m._check_posts(bot)
        assert len(bot.sent) == 1
        assert len(bot.sent[0][1]) == 1   # 只剩文字段，配图失败不该吞掉整条推送


class TestFormatPostPush:
    def test_template_exact(self):
        m = _make_monitor(uid="1", state={})
        assert m._format_post_push("今天直播好累", "https://weibo.com/u/1/abc") == \
            "灰泽满刚刚发了微博哦！\n\n微博内容：今天直播好累\n\nhttps://weibo.com/u/1/abc"

    def test_empty_text_placeholder(self):
        m = _make_monitor(uid="1", state={})
        assert "（图片微博）" in m._format_post_push("", "https://weibo.com/u/1/abc")


class TestErrorThrottle:
    async def test_repeated_error_warn_throttled(self, monkeypatch):
        m = _make_monitor(uid="1", state={}, _primed=True)
        monkeypatch.setattr(wb, "get_weibo_cookie", lambda: "SUB=x")

        async def boom():
            raise RuntimeError("微博接口 HTTP 403")

        monkeypatch.setattr(m, "_fetch_latest_post", boom)
        await m._check_posts(FakeBot())
        first = m._last_err_warn
        assert first > 0                      # 首次报错留下时间戳
        await m._check_posts(FakeBot())
        assert m._last_err_warn == first      # 窗口内不重复喊

    async def test_error_does_not_crash_and_no_push(self, monkeypatch):
        m = _make_monitor(uid="1", state={}, _primed=True)
        monkeypatch.setattr(wb, "get_weibo_cookie", lambda: "SUB=x")

        async def boom():
            raise RuntimeError("网络瞬断")

        monkeypatch.setattr(m, "_fetch_latest_post", boom)
        bot = FakeBot()
        await m._check_posts(bot)   # 不抛
        assert bot.sent == []
