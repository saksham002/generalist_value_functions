"""Slack notification utilities for TPU jobs."""

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

    def notify_started(self, tpu_name: str, tpu_type: str, command: str) -> bool:
        """Notify that a job has started.

        Args:
            tpu_name: TPU VM name
            tpu_type: TPU type (e.g., "v6e-8")
            command: Command being run

        Returns:
            True if notification was sent
        """
        message = f":rocket: *TPU Job Started*\nTPU: {tpu_name} | Type: {tpu_type}\nCommand: `{command}`"
        return self.send(message)

    def notify_error(self, tpu_name: str, error: str, command: str) -> bool:
        """Notify that a job encountered an error.

        Args:
            tpu_name: TPU VM name
            error: Error message
            command: Command that failed

        Returns:
            True if notification was sent
        """
        message = f":x: *TPU Job Error*\nTPU: {tpu_name}\nError: {error}\nCommand: `{command}`"
        return self.send(message)

    def notify_preemption(
        self,
        tpu_name: str,
        command: str,
        retry_count: int,
        max_retries: int | None,
    ) -> bool:
        """Notify that a TPU was preempted.

        Args:
            tpu_name: TPU VM name
            command: Command that was interrupted
            retry_count: Current retry attempt number
            max_retries: Maximum number of retries (None for infinite)

        Returns:
            True if notification was sent
        """
        retry_info = f"retry {retry_count}"
        if max_retries is not None:
            retry_info += f"/{max_retries}"

        message = (
            f":warning: *TPU Preempted* ({retry_info})\n"
            f"TPU: {tpu_name}\n"
            f"Creating new TPU and retrying...\n"
            f"Command: `{command}`"
        )
        return self.send(message)

    def notify_completion(
        self,
        tpu_name: str,
        command: str,
        duration_seconds: float,
        *,
        success: bool,
        output_tail: str = "",
    ) -> bool:
        """Notify that a job completed.

        Args:
            tpu_name: TPU VM name
            command: Command that completed
            duration_seconds: How long the job ran
            success: Whether the job succeeded
            output_tail: Last lines of output (included in failure messages)

        Returns:
            True if notification was sent
        """
        duration_str = _format_duration(duration_seconds)

        if success:
            emoji = ":white_check_mark:"
            title = "Job Completed"
        else:
            emoji = ":x:"
            title = "Job Failed"

        message = f"{emoji} *{title}*\nTPU: {tpu_name} | Duration: {duration_str}\nCommand: `{command}`"

        # Include error output for failed jobs
        if not success and output_tail:
            # Truncate to last 500 chars to fit Slack message limits
            truncated = output_tail[-500:] if len(output_tail) > 500 else output_tail
            message += f"\n```\n{truncated}\n```"

        return self.send(message)


def _format_duration(seconds: float) -> str:
    """Format duration in human-readable form."""
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        minutes = int(seconds / 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    hours = int(seconds / 3600)
    minutes = int((seconds % 3600) / 60)
    return f"{hours}h {minutes}m"
