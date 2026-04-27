from __future__ import annotations

from email.message import EmailMessage
import mimetypes
from pathlib import Path
import smtplib

import requests
from twilio.rest import Client


def send_violation_email(
    smtp_host: str,
    smtp_port: int,
    username: str,
    password: str,
    sender: str,
    recipient: str,
    subject: str,
    body: str,
    attachment_path: Path | None = None,
) -> None:
    message = EmailMessage()
    message["From"] = sender
    message["To"] = recipient
    message["Subject"] = subject
    message.set_content(body)

    if attachment_path and attachment_path.exists():
        mime_type, _ = mimetypes.guess_type(str(attachment_path))
        maintype, subtype = (mime_type or "application/octet-stream").split("/", 1)
        with attachment_path.open("rb") as handle:
            message.add_attachment(
                handle.read(),
                maintype=maintype,
                subtype=subtype,
                filename=attachment_path.name,
            )

    with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as server:
        server.starttls()
        if username:
            server.login(username, password)
        server.send_message(message)


def send_violation_whatsapp(
    account_sid: str,
    auth_token: str,
    from_whatsapp: str,
    to_whatsapp: str,
    body: str,
    media_url: str | None = None,
) -> str:
    client = Client(account_sid, auth_token)
    payload: dict[str, str] = {
        "from_": from_whatsapp.strip(),
        "body": body,
        "to": to_whatsapp.strip(),
    }
    if media_url:
        payload["media_url"] = [media_url.strip()]
    message = client.messages.create(**payload)
    return message.sid


def send_violation_telegram(
    bot_token: str,
    chat_id: str,
    message_text: str,
    photo_path: Path | None = None,
) -> dict:
    base_url = f"https://api.telegram.org/bot{bot_token.strip()}"

    if photo_path and photo_path.exists():
        with photo_path.open("rb") as handle:
            response = requests.post(
                f"{base_url}/sendPhoto",
                data={
                    "chat_id": chat_id.strip(),
                    "caption": message_text,
                },
                files={"photo": (photo_path.name, handle, "image/jpeg")},
                timeout=30,
            )
    else:
        response = requests.post(
            f"{base_url}/sendMessage",
            data={
                "chat_id": chat_id.strip(),
                "text": message_text,
            },
            timeout=30,
        )

    response.raise_for_status()
    payload = response.json()
    if not payload.get("ok"):
        raise RuntimeError(payload.get("description", "Telegram API request failed."))
    return payload
