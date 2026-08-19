"""Slack notifications for the three events a launcher cannot resolve on its own.

A job starting is worth knowing because it is the handshake that the launch worked; a
preemption and an expired gcloud credential are worth knowing because both need a person.
Everything else the launcher does — a run finishing, a step failing, memory climbing — is
visible in the log it already writes, and paging on it trained the reader to ignore the
channel. Credential expiry is sent from :mod:`openpi.tpu.gcloud`, where it is detected.
"""

import logging
import os

import requests

logger = logging.getLogger(__name__)


class SlackNotifier:
    """Sends Slack notifications for TPU job events via DM or webhook."""

    def __init__(
        self,
        bot_token: str | None = None,
        user_id: str | None = None,
        webhook_url: str | None = None,
    ):
        """Initialize the Slack notifier.

        Prefers DM via bot token if available, falls back to webhook.

        Args:
            bot_token: Slack bot token (xoxb-...). If not provided, uses SLACK_BOT_TOKEN env var.
            user_id: Slack user ID to DM. If not provided, uses SLACK_USER_ID env var.
            webhook_url: Slack webhook URL (fallback). If not provided, uses SLACK_WEBHOOK_URL env var.
        """
        self.bot_token = bot_token or os.environ.get("SLACK_BOT_TOKEN")
        self.user_id = user_id or os.environ.get("SLACK_USER_ID")
        self.webhook_url = webhook_url or os.environ.get("SLACK_WEBHOOK_URL")

    def send(self, message: str) -> bool:
        """Send a message to Slack via DM (preferred) or webhook (fallback).

        Args:
            message: Message text (supports Slack markdown)

        Returns:
            True if message was sent successfully, False otherwise
        """
        if self.bot_token and self.user_id:
            return self._send_dm(message)
        if self.webhook_url:
            return self._send_webhook(message)

        logger.debug("No Slack credentials configured, skipping notification")
        return False

    def _send_dm(self, message: str) -> bool:
        """Send a DM via Slack API."""
        try:
            response = requests.post(
                "https://slack.com/api/chat.postMessage",
                headers={"Authorization": f"Bearer {self.bot_token}"},
                json={"channel": self.user_id, "text": message},
                timeout=10,
            )
            data = response.json()
            if data.get("ok"):
                logger.debug("Slack DM sent")
                return True
            logger.warning("Slack DM failed: %s", data.get("error"))
            return False
        except Exception as e:
            logger.warning("Failed to send Slack DM: %s", e)
            return False

    def _send_webhook(self, message: str) -> bool:
        """Send via webhook (fallback)."""
        try:
            response = requests.post(
                self.webhook_url,
                json={"text": message},
                timeout=10,
            )
            if response.status_code == 200:
                logger.debug("Slack notification sent via webhook")
                return True
            logger.warning("Slack webhook failed: %s", response.text)
            return False
        except Exception as e:
            logger.warning("Failed to send Slack webhook: %s", e)
            return False

    def notify_started(self, tpu_name: str, tpu_type: str, zone: str, run_id: str, command: str) -> bool:
        """A job has just been started on a pod."""
        return self.send(
            f":rocket: *TPU job started*\n"
            f"• pod: {tpu_name} ({tpu_type}, {zone})\n"
            f"• run: `{run_id}`\n"
            f"• command: `{command}`"
        )

    def notify_preemption(self, tpu_name: str, run_id: str, retry_count: int, max_retries: int | None) -> bool:
        """A pod was preempted out from under a running job."""
        budget = f"{retry_count}" if max_retries is None else f"{retry_count}/{max_retries}"
        return self.send(f":warning: *TPU preempted* (retry {budget})\n• pod: {tpu_name}\n• run: `{run_id}`")
