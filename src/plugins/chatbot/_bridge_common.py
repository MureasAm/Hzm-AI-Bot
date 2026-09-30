"""bili/weibo 两个推送桥的公共小件（消重复）。

状态读写、图片下载逻辑两边曾各抄一份，这里收敛成一个实现。
cookie 会话策略刻意不同（B站塞 Header / 微博用 httpx cookie jar 续期），不在此统一——
那是有注释解释的差异，不是重复。
"""
import json
import re
import tempfile
import time
from pathlib import Path

# ==================== 写进记忆前的整理 ====================
# 通知模板是**播报腔**：`灰泽满刚刚发了动态哦！\n\n动态内容：…\n\n<链接>`。
# 照原样写进她的短期记忆有两个问题：
#  ① 记忆行是"灰泽满：{内容}"，于是变成「灰泽满：灰泽满刚刚发了动态哦！」——
#     名字出现两遍，读起来像旁白（正是提示词里明令禁止的那种句式）；
#  ② 短期记忆会作为 few-shot 喂回模型，播报句会被她学走。
# 所以剥掉播报开头/正文前缀/链接，留一句人话。
_NOTICE_HEADS = {
    "灰泽满刚刚发了动态哦！": "刚发了条动态",
    "灰泽满刚刚发了微博哦！": "刚发了条微博",
    "灰泽满宣布开播！": "刚宣布开播",
}
_BODY_PREFIXES = ("动态内容：", "微博内容：", "今天的内容是：")
_TRAILING_URL_RE = re.compile(r"\s*https?://\S+\s*$")


def to_memory_line(content: str) -> str:
    """把推出去的文案整理成"记忆里她说过的那句"。

    她本人的话（主动发言）原样返回；通知模板转成 `（刚发了条动态：…）`。
    """
    t = (content or "").strip()
    head = next((h for h in _NOTICE_HEADS if t.startswith(h)), "")
    if not head:
        return t
    body = t[len(head):].strip()
    for p in _BODY_PREFIXES:
        if body.startswith(p):
            body = body[len(p):].strip()
    body = _TRAILING_URL_RE.sub("", body).strip()
    what = _NOTICE_HEADS[head]
    return f"（{what}：{body}）" if body else f"（{what}）"


def is_newer_id(new_id, last_id) -> bool:
    """新抓到的 id 是否比"已见过的最大 id"更新（推送去重的水位线判据）。

    B站动态 id / 微博 mid **都随时间严格递增**（实测各取 10~20 条验证过），
    所以"更大 = 更新"。

    踩坑（为什么不能只判"和上一条不同"）：状态里本来只记**一条** id，
    判据是 `新 id != 记的那条`。她**撤回**最新那条动态后，接口返回的"最新"
    退回成上一条（更旧、id 更小），与记的那条不同 → 被判成新动态，
    **把昨天的动态又推了一遍**（用户实际遇到：撤回后重推了一条"晚安"）。
    置顶/删博回退同理。改成水位线后，任何"比见过的更旧"的 id 一律不推。

    数字不可比时（异常数据）退回"不同即新"，保持老行为、不倒退。
    """
    n, l = str(new_id or ""), str(last_id or "")
    if not n:
        return False
    if not l:
        return True                      # 还没记录过，当作新的
    if n.isdigit() and l.isdigit():
        return int(n) > int(l)
    return n != l


def load_state_file(path: Path) -> dict:
    """读去重状态文件；缺失/损坏返回空 dict。"""
    if path.exists():
        try:
            return json.loads(path.read_text("utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def save_state_file(path: Path, state: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, ensure_ascii=False, indent=2), "utf-8")
    except OSError as e:
        print(f"⚠️ 状态写入失败: {e}")


async def download_image_to_tmp(url: str, prefix: str) -> Path:
    """下载图片并统一转 JPEG 到临时文件；失败抛异常由调用方兜底。"""
    from .vision import _read_image_bytes, _normalize_image
    data = await _read_image_bytes(url)
    data = _normalize_image(data)
    tmp = Path(tempfile.gettempdir()) / f"{prefix}_{time.time_ns()}.jpg"
    tmp.write_bytes(data)
    return tmp
