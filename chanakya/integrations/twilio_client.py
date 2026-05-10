"""
twilio_client.py — Twilio voice and SMS integration.

Provides:
  make_call(to, twiml_url) -> call_sid
  send_sms(to, body) -> message_sid

Raises TwilioError on API failure; callers fall back to Telegram text.
"""

import logging

logger = logging.getLogger(__name__)


class TwilioError(Exception):
    """Raised when a Twilio API call fails."""


class TwilioClient:
    """Thin wrapper around the Twilio REST client."""

    def __init__(self) -> None:
        from twilio.rest import Client as TwilioRestClient
        from chanakya.config import TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN

        self._client = TwilioRestClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

    def send_sms(self, to: str, body: str) -> str:
        """Send an SMS message.

        Args:
            to: Destination phone number in E.164 format.
            body: Message body text.

        Returns:
            The Twilio message SID.

        Raises:
            TwilioError: If the Twilio API call fails for any reason.
        """
        from chanakya.config import TWILIO_PHONE_NUMBER

        try:
            msg = self._client.messages.create(
                body=body,
                to=to,
                from_=TWILIO_PHONE_NUMBER,
            )
            logger.info("SMS sent to %s, sid=%s", to, msg.sid)
            return msg.sid
        except Exception as exc:
            raise TwilioError(f"SMS to {to} failed: {exc}") from exc

    def make_call(self, to: str, twiml_url: str) -> str:
        """Initiate an outbound voice call.

        Args:
            to: Destination phone number in E.164 format.
            twiml_url: URL that Twilio will fetch for TwiML instructions.

        Returns:
            The Twilio call SID.

        Raises:
            TwilioError: If the Twilio API call fails for any reason.
        """
        from chanakya.config import TWILIO_PHONE_NUMBER

        try:
            call = self._client.calls.create(
                url=twiml_url,
                to=to,
                from_=TWILIO_PHONE_NUMBER,
            )
            logger.info("Call initiated to %s, sid=%s", to, call.sid)
            return call.sid
        except Exception as exc:
            raise TwilioError(f"Call to {to} failed: {exc}") from exc
