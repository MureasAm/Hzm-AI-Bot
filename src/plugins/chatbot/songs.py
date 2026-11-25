"""按文件名索引可供点唱的预录音频。"""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal
from unicodedata import normalize

from nonebot import logger

from .constants import PROJECT_ROOT
from .voice import send_voice_file

SONG_LIBRARY_ROOT = PROJECT_ROOT / "assets" / "audio" / "songs"
_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg", ".opus"}


@dataclass(frozen=True)
class Song:
    title: str
    clip: Path | None
    full: Path | None


@dataclass(frozen=True)
class SongDelivery:
    title: str
    mode: Literal["clip", "full"]
    path: Path
    as_file: bool = False


def song_key(title: str) -> str:
    return normalize("NFC", normalize("NFC", title.strip()).casefold())


@lru_cache(maxsize=256)
def _warn_library(message: str) -> None:
    """曲库会随请求刷新，重复的运维提示只记录一次。"""
    logger.warning("[唱歌] {}", message)


def _scan_folder(root: Path, folder_name: str) -> dict[str, tuple[str, Path]]:
    folder = root / folder_name
    try:
        directory = folder.resolve(strict=True)
        if not directory.is_relative_to(root):
            _warn_library(f"曲库子目录指向曲库外，已忽略 {folder}")
            return {}
        candidates = sorted(directory.iterdir(), key=lambda path: path.name)
    except (OSError, RuntimeError) as exc:
        _warn_library(f"无法读取曲库子目录 {folder}（{exc}）")
        return {}

    tracks = {}
    for candidate in candidates:
        if candidate.suffix.lower() not in _AUDIO_EXTENSIONS:
            continue
        title = normalize("NFC", candidate.stem.strip())
        if not title:
            continue
        try:
            path = candidate.resolve(strict=True)
            if not path.is_relative_to(directory):
                _warn_library(f"音频指向曲库子目录外，已忽略 {candidate}")
                continue
            if not path.is_file() or path.stat().st_size == 0:
                continue
        except (OSError, RuntimeError) as exc:
            _warn_library(f"无法读取曲库音频 {candidate}（{exc}）")
            continue

        key = song_key(title)
        if key in tracks:
            _warn_library(f"{folder} 有同名音频，保留 {tracks[key][1].name}，忽略 {candidate.name}")
            continue
        tracks[key] = (title, path)
    return tracks


def load_songs(root: Path | None = None) -> dict[str, Song]:
    """读取两种录音，保留仅有片段或仅有完整版的条目。"""
    root = SONG_LIBRARY_ROOT if root is None else root
    try:
        root = root.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        _warn_library(f"无法读取曲库 {root}（{exc}）")
        return {}

    clips = _scan_folder(root, "经典两句")
    full_songs = _scan_folder(root, "完整版")
    songs = {}
    for key in sorted(clips.keys() | full_songs.keys()):
        clip_title, clip = clips.get(key, ("", None))
        full_title, full = full_songs.get(key, ("", None))
        songs[key] = Song(clip_title or full_title, clip, full)
    return songs


async def send_song(bot, target_id: str, delivery: SongDelivery) -> bool:
    """将现成音频发到私聊，API 异常由对话层处理。"""
    if delivery.mode not in ("clip", "full"):
        raise ValueError(f"未知的唱歌发送形式 {delivery.mode!r}")
    if not delivery.as_file:
        return await send_voice_file(bot, target_id, True, str(delivery.path))
    if delivery.mode != "full":
        raise ValueError("音频文件发送只支持完整版")
    await bot.upload_private_file(
        user_id=target_id,
        file=delivery.path.resolve().as_uri(),
        name=delivery.path.name,
    )
    return True
