"""视觉理解：用智谱 glm-4.6v 描述图片，供对话注入与动态配图描述复用。

输入可以是：
- http(s) URL（QQ 图链可能 http / 会过期，必须下载后转 base64 再发）
- 本地文件路径（离线工具用）
- 已构造的 data: URI（直接透传）

任一步失败返回空串，调用方静默跳过，绝不阻塞主流程。
"""
import base64
import io
from pathlib import Path

import httpx
from PIL import Image

from .config import get_vision_model
from .constants import VISION_MAX_TOKENS, VISION_THINKING_DISABLED

VISION_PROMPT = (
    "简要描述这张图的内容和氛围：是表情包/截图/照片/文档就说清最直观的要点，"
    "包括人物动作表情、画面文字、场景氛围。60字以内，适合聊天时接梗。"
)
DOWNLOAD_TIMEOUT = 10.0
MAX_IMAGE_BYTES = 15 * 1024 * 1024  # 15MB 上限，超出跳过


_QQ_IMAGE_DOMAINS = ("qq.com", "qpic.cn")
_BILI_IMAGE_DOMAINS = ("bilibili.com", "hdslb.com", "biliimg.com")


def _referer_for(url: str) -> str:
    """按图链域名选 Referer（hotlink 校验，选错会 400/403）。

    - QQ 图链（multimedia.nt.qq.com.cn / gchat.qpic.cn 等）→ QQ 域 Referer。
      踩坑：曾统一改成 B站 Referer 导致 QQ 图片下载 400 'invalid rkey'（QQ CDN hotlink 校验）。
    - B站图链（hdslb.com 等）→ B站 Referer（B站 WAF 拦默认 UA/Referer）。
    - 其他域名 → 不强加 Referer（留空）。
    """
    host = url.split("//", 1)[1].split("/", 1)[0].lower() if "//" in url else ""
    if any(d in host for d in _BILI_IMAGE_DOMAINS):
        return "https://www.bilibili.com/"
    if any(d in host for d in _QQ_IMAGE_DOMAINS):
        return "https://qun.qq.com/"
    return ""


async def _read_image_bytes(source: str) -> bytes:
    """把各种输入归一化为图片字节。

    QQ/B站图链会拦默认 UA，必须带浏览器 UA + 对应域的 Referer（见 _referer_for）；
    也兼容 file:// 前缀与本地路径。
    """
    if source.startswith("data:"):
        return b""
    if source.startswith("file://"):
        source = source[len("file://"):]
    if source.startswith(("http://", "https://")):
        headers = {
            "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                           "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"),
        }
        referer = _referer_for(source)
        if referer:
            headers["Referer"] = referer
        # trust_env=False：强制直连，不受环境变量代理影响
        # （踩坑：bot 环境曾有失效代理 HTTP_PROXY=127.0.0.1:50454，导致 B站图链下载 ConnectError）
        async with httpx.AsyncClient(timeout=DOWNLOAD_TIMEOUT, follow_redirects=True,
                                     headers=headers, trust_env=False) as client:
            resp = await client.get(source)
            resp.raise_for_status()
            return resp.content
    # 本地文件路径
    p = Path(source)
    if p.exists():
        return p.read_bytes()
    raise ValueError(f"无法识别的图片来源: {source[:80]}")


def _normalize_image(data: bytes) -> bytes:
    """用 Pillow 把任意格式统一转成 JPEG，避开 glm-4.6v 不支持的 WEBP 等格式。

    透明（RGBA/LA/P）贴图先铺白底再转 RGB，避免透明像素变黑。
    """
    img = Image.open(io.BytesIO(data))
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        img = bg
    else:
        img = img.convert("RGB")
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=92)
    return out.getvalue()


def _detect_image_mime(data: bytes) -> str:
    """根据文件魔数判断真实 MIME；不是已知图片格式返回空串。

    QQ 图链常返回 WEBP/PNG，若硬标成 JPEG，视觉模型会解析失败返回空描述。
    """
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:2] == b"BM":
        return "image/bmp"
    return ""


def _to_data_uri(data: bytes, mime: str = "image/jpeg") -> str:
    """图片字节 → data: URI（glm 视觉走 OpenAI 兼容 image_url），带正确 MIME。"""
    return f"data:{mime};base64," + base64.b64encode(data).decode("ascii")


async def describe_image(zhipu_client, source: str, model: str = None) -> str:
    """描述图片。zhipu_client 由调用方传入（core._get_clients 里的 zhipu client）。"""
    model = model or get_vision_model()

    if source.startswith("data:"):
        data_uri = source
    else:
        try:
            print(f"[视觉] 下载图片: {source[:100]}")
            data = await _read_image_bytes(source)
            print(f"[视觉] 下载成功，{len(data)} 字节")
        except Exception as e:
            print(f"⚠️ 图片下载失败: {e}")
            return ""
        if not data or len(data) > MAX_IMAGE_BYTES:
            print("⚠️ 图片为空或超 15MB，跳过视觉描述")
            return ""
        mime = _detect_image_mime(data)
        print(f"[视觉] 原始格式: {mime or '未知'}")
        try:
            data = _normalize_image(data)  # 统一转 JPEG，避开模型不支持的 WEBP 等
            print(f"[视觉] 已转码 JPEG，{len(data)} 字节")
        except Exception as e:
            print(f"⚠️ 图片解码/转码失败（忽略）: {e}")
            return ""
        data_uri = _to_data_uri(data, "image/jpeg")

    try:
        resp = await zhipu_client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": VISION_PROMPT},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            }],
            max_tokens=VISION_MAX_TOKENS,
            **VISION_THINKING_DISABLED,
        )
        desc = (resp.choices[0].message.content or "").strip()
        return desc
    except Exception as e:
        print(f"⚠️ 视觉模型调用失败: {e}")
        return ""
