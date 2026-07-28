from __future__ import annotations

from email import policy
from email.message import EmailMessage
from email.parser import BytesParser
from typing import cast


def extract_image(content_type: str, body: bytes) -> tuple[bytes, str]:
    if len(body) > (2 * 1024 * 1024) + 65_536:
        raise OverflowError("request exceeds the upload limit.")
    if content_type.startswith("image/"):
        return body, content_type.split(";", maxsplit=1)[0].strip().lower()
    if not content_type.startswith("multipart/form-data"):
        raise TypeError("request must contain one PNG or JPEG image.")
    message = cast(EmailMessage, BytesParser(policy=policy.default).parsebytes(  # type: ignore[arg-type]
        b"Content-Type: "
        + content_type.encode("ascii", errors="strict")
        + b"\r\nMIME-Version: 1.0\r\n\r\n"
        + body
    ))
    candidates = [
        part
        for part in message.iter_parts()
        if part.get_content_disposition() == "form-data"
        and part.get_param("name", header="content-disposition") == "file"
    ]
    if len(candidates) != 1:
        raise ValueError("multipart request must contain exactly one file field.")
    payload = candidates[0].get_payload(decode=True)
    if not isinstance(payload, bytes):
        raise ValueError("uploaded file could not be decoded.")
    return payload, candidates[0].get_content_type().lower()
