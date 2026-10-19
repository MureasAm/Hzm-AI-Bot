"""视觉理解：把图片转成一段文字描述，供对话注入与动态配图描述复用。

默认视觉模型跟主模型同一条线路（gemini）；配 `VISION_MODEL=glm-4.6v` 可切回智谱。
无论哪家，**输出都是文字描述**——主模型从来看不到图片像素，所以"换视觉模型"
改变的是描述的准确度/风格，不改变主模型的看图能力。

输入可以是：
- http(s) URL（QQ 图链可能 http / 会过期，必须下载后转 base64 再发）
- 本地文件路径（离线工具用）
- 已构造的 data: URI（直接透传）

任一步失败返回空串，调用方静默跳过，绝不阻塞主流程。
"""
import asyncio
import base64
import io
import re
from pathlib import Path

import httpx
from PIL import Image

from .config import get_vision_model, extract_chat_content
from .constants import VISION_MAX_TOKENS, VISION_EXTRA_PARAMS

VISION_PROMPT = (
    "先判断图片类型，第一行只输出一个标签：[表情包] / [截图] / [照片] / [文档] / [其他]。\n"
    "第二行简要描述画面内容和文字：人物动作表情、画面里的文字、场景，60字以内。\n"
    "只描述看到的东西，不要写“适合接梗/适合吐槽”这类用途判断。"
)
DOWNLOAD_TIMEOUT = 10.0
MAX_IMAGE_BYTES = 15 * 1024 * 1024  # 15MB 上限，超出跳过


_QQ_IMAGE_DOMAINS = ("qq.com", "qpic.cn")
_BILI_IMAGE_DOMAINS = ("bilibili.com", "hdslb.com", "biliimg.com")
# 微博图床：sinaimg.cn 的 wx1-4 / ww1-4 / tvax1-4 等子域，全走这一个主域。
# 踩坑：微博动态能抓到图链、图却发不出去——因为这里没登记，Referer 为空被图床 403
# 挡掉（实测 wx1.sinaimg.cn 不带 Referer = 403/238字节，带 weibo.com = 200/331KB）。
_WEIBO_IMAGE_DOMAINS = ("sinaimg.cn",)


def _referer_for(url: str) -> str:
    """按图链域名选 Referer（hotlink 校验，选错或漏选会 400/403）。

    - QQ 图链（multimedia.nt.qq.com.cn / gchat.qpic.cn 等）→ QQ 域 Referer。
      踩坑：曾统一改成 B站 Referer 导致 QQ 图片下载 400 'invalid rkey'（QQ CDN hotlink 校验）。
    - B站图链（hdslb.com 等）→ B站 Referer（B站 WAF 拦默认 UA/Referer）。
    - 微博图床（sinaimg.cn 等）→ 微博 Referer（不带直接 403）。
    - 其他域名 → 不强加 Referer（留空）。

    教训：**每加一个图源就要回来登记一次**——漏一个的表现是"文字发得出去、图全丢"，
    因为下载失败只打日志、不阻塞推送（见 weibo_bridge._check_posts 的兜底）。
    """
    host = url.split("//", 1)[1].split("/", 1)[0].lower() if "//" in url else ""
    if any(d in host for d in _BILI_IMAGE_DOMAINS):
        return "https://www.bilibili.com/"
    if any(d in host for d in _QQ_IMAGE_DOMAINS):
        return "https://qun.qq.com/"
    if any(d in host for d in _WEIBO_IMAGE_DOMAINS):
        return "https://weibo.com/"
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
        ua = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
              "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")
        referer = _referer_for(source)
        # trust_env=False：强制直连，不受环境变量代理影响
        # （踩坑：bot 环境曾有失效代理 HTTP_PROXY=127.0.0.1:50454，导致 B站图链下载 ConnectError）
        # 重试策略：先带 Referer；400/403/404 时降级为不带 Referer 再试一次
        # （QQ CDN hotlink 校验策略会变，曾统一 referer 后导致 400）
        attempts = [{"User-Agent": ua, "Referer": referer} if referer else {"User-Agent": ua},
                    {"User-Agent": ua}]
        last_err: Exception | None = None
        for hdrs in attempts:
            try:
                async with httpx.AsyncClient(timeout=DOWNLOAD_TIMEOUT, follow_redirects=True,
                                             headers=hdrs, trust_env=False) as client:
                    resp = await client.get(source)
                    resp.raise_for_status()
                    return resp.content
            except httpx.HTTPStatusError as e:
                last_err = e
                if e.response.status_code in (400, 403, 404):
                    continue  # 换 referer 策略重试一次
                raise
        assert last_err is not None
        raise last_err
    # 本地文件路径
    p = Path(source)
    if p.exists():
        return p.read_bytes()
    raise ValueError(f"无法识别的图片来源: {source[:80]}")


def _normalize_image(data: bytes) -> bytes:
    """用 Pillow 把任意格式统一转成 JPEG，避开视觉模型不支持的 WEBP 等格式。

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
    """图片字节 → data: URI（视觉模型走 OpenAI 兼容 image_url），带正确 MIME。"""
    return f"data:{mime};base64," + base64.b64encode(data).decode("ascii")


def _extra_params(model: str) -> dict:
    """按视觉模型给额外参数（glm 要关思考，gemini 不认识这个字段）。"""
    return VISION_EXTRA_PARAMS.get(model, {})


_IMAGE_KIND_RE = re.compile(r"^\s*\[?(表情包|截图|照片|文档|其他)\]?")


def image_kind(desc: str) -> str:
    """视觉描述 → 图片类型；没有标签的旧描述按“表情包”前缀兜底。"""
    text = (desc or "").strip()
    m = _IMAGE_KIND_RE.match(text)
    if m:
        return m.group(1)
    if text.startswith(("表情包", "一张表情包")):
        return "表情包"
    return "其他"


async def describe_image_bytes(client, data: bytes, model: str = None) -> str:
    """描述已下载的图片字节（供 chat_window 的"收到即缓存"路径复用，避免重复下载）。"""
    model = model or get_vision_model()
    if not data or len(data) > MAX_IMAGE_BYTES:
        print("⚠️ 图片为空或超 15MB，跳过视觉描述")
        return ""
    mime = _detect_image_mime(data)
    try:
        # 统一转 JPEG，避开模型不支持的 WEBP 等。解码+重编码是纯 CPU（大图近百毫秒），
        # 放线程里跑，别卡住事件循环（一批多图是 gather 并发进来的）。
        data = await asyncio.to_thread(_normalize_image, data)
    except Exception as e:
        print(f"⚠️ 图片解码/转码失败（忽略）: {e}")
        return ""
    data_uri = _to_data_uri(data, "image/jpeg")
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": VISION_PROMPT},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            }],
            max_tokens=VISION_MAX_TOKENS,
            **_extra_params(model),
        )
        return extract_chat_content(resp)
    except Exception as e:
        print(f"⚠️ 视觉模型调用失败: {e}")
        return ""


async def describe_image(client, source: str, model: str = None) -> str:
    """描述图片。client 由调用方传入（运行时用 config.get_vision_client()）。"""
    if source.startswith("data:"):
        data_uri = source
        model = model or get_vision_model()
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": VISION_PROMPT},
                        {"type": "image_url", "image_url": {"url": data_uri}},
                    ],
                }],
                max_tokens=VISION_MAX_TOKENS,
                **_extra_params(model),
            )
            return extract_chat_content(resp)
        except Exception as e:
            print(f"⚠️ 视觉模型调用失败: {e}")
            return ""
    try:
        print(f"[视觉] 下载图片: {source[:100]}")
        data = await _read_image_bytes(source)
        print(f"[视觉] 下载成功，{len(data)} 字节")
    except Exception as e:
        print(f"⚠️ 图片下载失败: {e}")
        return ""
    return await describe_image_bytes(client, data, model)
