from unittest.mock import AsyncMock
from urllib.parse import unquote, urlsplit

import pytest

from src.plugins.chatbot import songs


def test_library_pairs_titles_across_formats_and_normalizes_lookup(tmp_path):
    clip_dir = tmp_path / "经典两句"
    full_dir = tmp_path / "完整版"
    clip_dir.mkdir()
    full_dir.mkdir()
    clip = clip_dir / "Cafe\u0301.wav"
    full = full_dir / "CAFÉ.MP3"
    clip.write_bytes(b"audio")
    full.write_bytes(b"audio")

    assert songs.song_key("café") == songs.song_key("CAFÉ") == "café"
    assert songs.load_songs(tmp_path) == {"café": songs.Song("Café", clip, full)}


def test_library_keeps_single_versions_and_ignores_unusable_entries(tmp_path):
    clip_dir = tmp_path / "经典两句"
    clip_dir.mkdir()
    clip = clip_dir / "海风.mp3"
    (clip_dir / "海风.wav").write_bytes(b"alternate")
    clip.write_bytes(b"audio")
    (clip_dir / "空文件.wav").touch()
    (clip_dir / "备注.txt").write_text("not audio")
    (clip_dir / "目录.wav").mkdir()
    (tmp_path / "根目录.flac").write_bytes(b"outside the two directories")

    assert songs.load_songs(tmp_path) == {"海风": songs.Song("海风", clip, None)}

    full_dir = tmp_path / "完整版"
    full_dir.mkdir()
    full = full_dir / "山风.m4a"
    full.write_bytes(b"audio")

    assert songs.load_songs(tmp_path) == {
        "山风": songs.Song("山风", None, full),
        "海风": songs.Song("海风", clip, None),
    }


def test_library_ignores_symlinks_to_audio_outside_the_library(tmp_path):
    root = tmp_path / "songs"
    clip_dir = root / "经典两句"
    clip_dir.mkdir(parents=True)
    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"audio")
    try:
        (clip_dir / "这首不能发.wav").symlink_to(outside)
    except OSError as exc:
        if getattr(exc, "winerror", None) != 1314:
            raise
        pytest.skip("Windows 缺少创建符号链接的权限")

    assert songs.load_songs(root) == {}


@pytest.mark.parametrize(("mode", "extension"), [("clip", ".wav"), ("full", ".m4a")])
async def test_song_is_sent_as_a_private_qq_voice_message_by_default(tmp_path, mode, extension):
    path = tmp_path / f"星星 %20#{extension}"
    path.write_bytes(b"audio")
    bot = AsyncMock()

    assert await songs.send_song(bot, "42", songs.SongDelivery(path.stem, mode, path))

    bot.send_private_msg.assert_awaited_once()
    assert bot.send_private_msg.call_args.kwargs["user_id"] == "42"
    record = bot.send_private_msg.call_args.kwargs["message"]
    assert record.type == "record"
    uri = urlsplit(record.data["file"])
    assert uri.scheme == "file"
    assert uri.fragment == ""
    assert uri.path.isascii() and " " not in uri.path
    assert unquote(uri.path).endswith(f"/星星 %20#{extension}")
    bot.upload_private_file.assert_not_awaited()


async def test_full_song_is_uploaded_as_a_named_private_qq_file_when_explicitly_requested(tmp_path):
    full = tmp_path / "星星 100%.mp3"
    full.write_bytes(b"audio")
    bot = AsyncMock()

    assert await songs.send_song(bot, "42", songs.SongDelivery("星星 100%", "full", full, as_file=True))

    bot.upload_private_file.assert_awaited_once_with(
        user_id="42", file=full.as_uri(), name="星星 100%.mp3",
    )
    bot.send_private_msg.assert_not_awaited()


@pytest.mark.parametrize("as_file", [False, True])
async def test_send_song_exposes_delivery_failure_without_trying_the_other_format(tmp_path, as_file):
    bot = AsyncMock()
    bot.upload_private_file.side_effect = RuntimeError("offline")
    bot.send_private_msg.side_effect = RuntimeError("offline")
    full = tmp_path / "星星.mp3"
    full.write_bytes(b"audio")

    with pytest.raises(RuntimeError, match="offline"):
        await songs.send_song(bot, "42", songs.SongDelivery("星星", "full", full, as_file=as_file))
    assert bot.upload_private_file.await_count == int(as_file)
    assert bot.send_private_msg.await_count == int(not as_file)
