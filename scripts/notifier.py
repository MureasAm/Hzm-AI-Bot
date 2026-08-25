#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SMTP 邮件通知：把文本/图片发到 QQ 邮箱。

用途：watchdog 在被踢需要扫码时，把二维码截图推送到你的 QQ 邮箱（手机 QQ 的
"新邮件提醒"会实时推送），以及发掉线/恢复告警。

配置写在 .env.prod（gitignore 里，安全）：
    SMTP_USER=2891616516@qq.com
    SMTP_AUTH_CODE=<QQ邮箱授权码>   # 必填。mail.qq.com 网页版：设置→账号→
                                    # 开启 SMTP 服务→生成授权码（16位，不是QQ密码）

用法（供 watchdog 等脚本调用）：
    from notifier import send_email
    send_email("主题", "正文", [image_path1, image_path2])
"""
import smtplib
import ssl
import sys
from email.header import Header
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formataddr
from pathlib import Path

# Windows 控制台默认 GBK，打 emoji 会崩；统一成 UTF-8（pythonw 下 stdout 是 None，忽略）
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env.prod"
SMTP_HOST = "smtp.qq.com"
SMTP_PORT = 465
DEFAULT_FROM = "2891616516@qq.com"


def _load_env() -> dict:
    """极简 .env 解析（不依赖 python-dotenv）。"""
    env = {}
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text("utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            env[k.strip()] = v.strip()
    return env


def _config() -> dict:
    env = _load_env()
    user = env.get("SMTP_USER") or DEFAULT_FROM
    return {
        "user": user,
        "code": env.get("SMTP_AUTH_CODE", ""),
        "to": env.get("SMTP_TO") or user,
    }


def send_email(subject: str, body: str, attachments: list[str] | None = None) -> bool:
    """发送一封邮件。attachments 里的图片会作为附件附上。返回是否发送成功。

    QQ 邮箱 SMTP 要求：登录账号 = 发件人；授权码代替密码。
    本机若走代理才连得上外网，注意 smtp.qq.com 是国内服务，一般可直连。
    """
    cfg = _config()
    if not cfg["code"]:
        print("❌ SMTP_AUTH_CODE 未配置（.env.prod 里加 SMTP_AUTH_CODE=你的授权码）")
        return False

    msg = MIMEMultipart()
    # 邮箱地址部分不能被 RFC2047 编码（QQ 邮箱会拒收），用 formataddr 组合
    msg["From"] = formataddr((str(Header("灰泽满Bot", "utf-8")), cfg["user"]))
    msg["To"] = cfg["to"]
    msg["Subject"] = Header(subject, "utf-8")
    msg.attach(MIMEText(body, "plain", "utf-8"))

    for path in attachments or []:
        p = Path(path)
        if not p.exists():
            print(f"⚠️ 附件不存在，跳过: {p}")
            continue
        with open(p, "rb") as f:
            img = MIMEImage(f.read())
            # 附件文件名用 ASCII 安全名，避免 QQ 邮箱解析问题
            img.add_header("Content-Disposition", "attachment",
                           filename=("utf-8", "", p.name))
            msg.attach(img)

    ctx = ssl.create_default_context()
    try:
        with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, context=ctx, timeout=20) as s:
            s.login(cfg["user"], cfg["code"])
            s.sendmail(cfg["user"], [cfg["to"]], msg.as_string())
        return True
    except Exception as e:  # noqa: BLE001  通知失败不致命，打日志即可
        print(f"❌ SMTP 发送失败: {e}")
        return False


if __name__ == "__main__":
    # 自测：python scripts/notifier.py "测试" "正文" [可选图片路径]
    import sys
    subj = sys.argv[1] if len(sys.argv) > 1 else "测试邮件"
    body = sys.argv[2] if len(sys.argv) > 2 else "这是一封测试邮件。"
    imgs = sys.argv[3:] or None
    ok = send_email(subj, body, imgs)
    print("✅ 发送成功" if ok else "❌ 发送失败")
