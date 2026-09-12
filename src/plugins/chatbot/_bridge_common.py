"""bili/weibo 两个推送桥的公共小件（消重复）。

状态读写、图片下载逻辑两边曾各抄一份，这里收敛成一个实现。
cookie 会话策略刻意不同（B站塞 Header / 微博用 httpx cookie jar 续期），不在此统一——
那是有注释解释的差异，不是重复。
"""
import json
import tempfile
import time
from pathlib import Path


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
