"""bili/weibo 两个推送桥的公共小件（消重复）。

状态读写、图片下载逻辑两边曾各抄一份，这里收敛成一个实现。
cookie 会话策略刻意不同（B站塞 Header / 微博用 httpx cookie jar 续期），不在此统一——
那是有注释解释的差异，不是重复。
"""
import json
import tempfile
import time
from pathlib import Path


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
