"""
Alerting Module

Sends email alerts for critical system events (model drift, retraining
outcomes, batch errors). Uses aiosmtplib for async SMTP delivery.

Configuration (all via environment variables):
    SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, ALERT_EMAIL
"""

import logging
import os
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

try:
    import aiosmtplib
    _SMTP_AVAILABLE = True
except ImportError:
    _SMTP_AVAILABLE = False

logger = logging.getLogger(__name__)


async def send_email_alert(
    subject: str,
    body: str,
    alert_type: str = "warning",
) -> bool:
    """
    Send an HTML email alert to the configured admin address.

    Args:
        subject: Alert subject line.
        body: Plain-text body (rendered inside a simple HTML wrapper).
        alert_type: One of 'info', 'warning', 'error' (affects heading colour).

    Returns:
        True if sent successfully, False otherwise.
    """
    if not _SMTP_AVAILABLE:
        logger.warning(
            "aiosmtplib is not installed — email alert skipped. "
            "Install with: pip install aiosmtplib"
        )
        return False

    smtp_host = os.getenv("SMTP_HOST")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER")
    smtp_password = os.getenv("SMTP_PASSWORD")
    alert_email = os.getenv("ALERT_EMAIL")

    if not all([smtp_host, smtp_user, smtp_password, alert_email]):
        logger.warning(
            "Email alert configuration incomplete — skipping alert: %s", subject
        )
        return False

    colour_map = {"info": "#17a2b8", "warning": "#ffc107", "error": "#dc3545"}
    heading_colour = colour_map.get(alert_type, "#ffc107")

    html_body = f"""
    <html>
      <body style="font-family: Arial, sans-serif; margin: 24px;">
        <h2 style="color: {heading_colour};">[{alert_type.upper()}] {subject}</h2>
        <p style="white-space: pre-wrap;">{body}</p>
        <hr>
        <small style="color: #666;">Email Categorization System — automated alert</small>
      </body>
    </html>
    """

    message = MIMEMultipart("alternative")
    message["From"] = smtp_user
    message["To"] = alert_email
    message["Subject"] = f"[{alert_type.upper()}] Email Categorizer — {subject}"
    message.attach(MIMEText(html_body, "html"))

    try:
        await aiosmtplib.send(
            message,
            hostname=smtp_host,
            port=smtp_port,
            username=smtp_user,
            password=smtp_password,
            start_tls=True,
        )
        logger.info("Alert email sent: %s", subject)
        return True
    except Exception as exc:
        logger.error("Failed to send alert email '%s': %s", subject, exc)
        return False


def send_alert_sync(subject: str, body: str, alert_type: str = "warning") -> None:
    """
    Synchronous wrapper — used from background scheduler threads.
    Schedules a coroutine via asyncio.run() if no event loop is active.
    """
    import asyncio

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(send_email_alert(subject, body, alert_type))
        else:
            loop.run_until_complete(send_email_alert(subject, body, alert_type))
    except Exception as exc:
        logger.error("Alert dispatch error: %s", exc)
