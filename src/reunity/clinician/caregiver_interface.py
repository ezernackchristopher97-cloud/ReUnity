"""
ReUnity Clinician and Caregiver Interface (CCI)

This module provides a secure interface for mental health professionals
and authorized caregivers to access aggregate patterns and support
coordination, always with explicit user consent.

The CCI enables:
- Secure sharing of anonymized patterns with treatment providers
- Crisis coordination with emergency contacts
- Progress tracking for therapeutic goals
- Collaborative care coordination

All access is governed by strict consent controls. Users maintain
complete sovereignty over their data and can revoke access at any time.

DISCLAIMER: This is not a clinical or treatment tool. It is a theoretical
and support framework only. All clinical decisions must be made by
qualified mental health professionals.

Author: Christopher Ezernack
"""

from __future__ import annotations

import hashlib
import secrets
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class AccessLevel(Enum):
    """Levels of access for clinicians/caregivers."""

    NONE = "none"  # No access
    CRISIS_ONLY = "crisis_only"  # Only notified during crisis
    SUMMARY = "summary"  # Access to aggregate summaries only
    DETAILED = "detailed"  # Access to detailed patterns
    FULL = "full"  # Full access (rare, requires explicit consent)


class ProviderType(Enum):
    """Types of care providers."""

    THERAPIST = "therapist"
    PSYCHIATRIST = "psychiatrist"
    CASE_MANAGER = "case_manager"
    CRISIS_COUNSELOR = "crisis_counselor"
    EMERGENCY_CONTACT = "emergency_contact"
    CAREGIVER = "caregiver"
    PEER_SUPPORT = "peer_support"


class ConsentStatus(Enum):
    """Status of consent for data sharing."""

    PENDING = "pending"
    GRANTED = "granted"
    REVOKED = "revoked"
    EXPIRED = "expired"


@dataclass
class ProviderProfile:
    """Profile for a care provider."""

    provider_id: str
    name: str
    provider_type: ProviderType
    organization: str | None = None
    credentials: str | None = None
    contact_info: dict[str, str] = field(default_factory=dict)
    access_level: AccessLevel = AccessLevel.NONE
    verified: bool = False
    created_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConsentRecord:
    """Record of consent for data sharing."""

    consent_id: str
    user_id: str
    provider_id: str
    access_level: AccessLevel
    data_types: list[str]  # What types of data are shared
    status: ConsentStatus
    granted_at: float | None = None
    expires_at: float | None = None
    revoked_at: float | None = None
    conditions: str = ""  # Any conditions on the consent
    signature_hash: str = ""  # Hash of consent signature


@dataclass
class SharedReport:
    """A report shared with a provider."""

    report_id: str
    provider_id: str
    report_type: str
    content: dict[str, Any]
    anonymized: bool = True
    created_at: float = field(default_factory=time.time)
    accessed_at: float | None = None
    access_log: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class CrisisAlert:
    """Alert sent during crisis situations."""

    alert_id: str
    user_id: str
    provider_ids: list[str]
    severity: str  # "low", "medium", "high", "critical"
    message: str
    entropy_level: float
    triggered_at: float = field(default_factory=time.time)
    acknowledged_by: list[str] = field(default_factory=list)
    resolved_at: float | None = None


class ClinicianCaregiverInterface:
    """
    Clinician and Caregiver Interface (CCI).

    This interface enables secure, consent-based sharing of information
    with mental health professionals and authorized caregivers. All
    sharing is governed by explicit user consent and can be revoked
    at any time.

    Key Principles:
    1. User sovereignty: Users control all data sharing
    2. Minimal disclosure: Share only what's necessary
    3. Transparency: Users can see all shared data
    4. Revocability: Consent can be withdrawn at any time

    DISCLAIMER: This is not a clinical or treatment tool.
    """

    def __init__(
        self,
        user_id: str,
        enable_crisis_alerts: bool = True,
        default_consent_duration_days: int = 90,
    ) -> None:
        """
        Initialize the Clinician/Caregiver Interface.

        Args:
            user_id: ID of the user whose data may be shared.
            enable_crisis_alerts: Whether to enable crisis alerting.
            default_consent_duration_days: Default consent duration.
        """
        self.user_id = user_id
        self.enable_crisis_alerts = enable_crisis_alerts
        self.default_consent_duration = default_consent_duration_days * 86400

        # Storage
        self._providers: dict[str, ProviderProfile] = {}
        self._consents: dict[str, ConsentRecord] = {}
        self._reports: dict[str, SharedReport] = {}
        self._alerts: list[CrisisAlert] = []
        self._access_log: list[dict[str, Any]] = []

    def register_provider(
        self,
        name: str,
        provider_type: ProviderType,
        organization: str | None = None,
        credentials: str | None = None,
        contact_info: dict[str, str] | None = None,
    ) -> ProviderProfile:
        """
        Register a new care provider.

        Args:
            name: Provider's name.
            provider_type: Type of provider.
            organization: Provider's organization.
            credentials: Professional credentials.
            contact_info: Contact information.

        Returns:
            The created ProviderProfile.
        """
        provider = ProviderProfile(
            provider_id=str(uuid.uuid4()),
            name=name,
            provider_type=provider_type,
            organization=organization,
            credentials=credentials,
            contact_info=contact_info or {},
        )

        self._providers[provider.provider_id] = provider
        return provider

    def get_provider(self, provider_id: str) -> ProviderProfile | None:
        """Get a provider by ID."""
        return self._providers.get(provider_id)

    def list_providers(self) -> list[ProviderProfile]:
        """List all registered providers."""
        return list(self._providers.values())

    def verify_provider(self, provider_id: str, verification_code: str) -> bool:
        """
        Verify a provider's identity.

        In production, this would involve credential verification.

        Args:
            provider_id: ID of the provider.
            verification_code: Verification code.

        Returns:
            True if verification successful.
        """
        if provider_id not in self._providers:
            return False

        # Simplified verification (production would be more robust)
        if len(verification_code) >= 6:
            self._providers[provider_id].verified = True
            return True

        return False

    def grant_consent(
        self,
        provider_id: str,
        access_level: AccessLevel,
        data_types: list[str],
        duration_days: int | None = None,
        conditions: str = "",
    ) -> ConsentRecord | None:
        """
        Grant consent for a provider to access data.

        Args:
            provider_id: ID of the provider.
            access_level: Level of access to grant.
            data_types: Types of data to share.
            duration_days: How long consent is valid.
            conditions: Any conditions on the consent.

        Returns:
            The created ConsentRecord or None if failed.
        """
        if provider_id not in self._providers:
            return None

        duration = (duration_days or self.default_consent_duration // 86400) * 86400
        now = time.time()

        # Create consent signature
        signature_data = f"{self.user_id}{provider_id}{access_level.value}{now}"
        signature_hash = hashlib.sha256(signature_data.encode()).hexdigest()

        consent = ConsentRecord(
            consent_id=str(uuid.uuid4()),
            user_id=self.user_id,
            provider_id=provider_id,
            access_level=access_level,
            data_types=data_types,
            status=ConsentStatus.GRANTED,
            granted_at=now,
            expires_at=now + duration,
            conditions=conditions,
            signature_hash=signature_hash,
        )

        self._consents[consent.consent_id] = consent

        # Update provider access level
        self._providers[provider_id].access_level = access_level

        # Log the consent grant
        self._log_access(
            provider_id=provider_id,
            action="consent_granted",
            details={"access_level": access_level.value, "data_types": data_types},
        )

        return consent

    def revoke_consent(self, consent_id: str) -> bool:
        """
        Revoke a previously granted consent.

        Args:
            consent_id: ID of the consent to revoke.

        Returns:
            True if revocation successful.
        """
        if consent_id not in self._consents:
            return False

        consent = self._consents[consent_id]
        consent.status = ConsentStatus.REVOKED
        consent.revoked_at = time.time()

        # Update provider access level
        provider_id = consent.provider_id
        if provider_id in self._providers:
            # Check if there are other active consents
            active_consents = [
                c for c in self._consents.values()
                if c.provider_id == provider_id
                and c.status == ConsentStatus.GRANTED
                and c.consent_id != consent_id
            ]

            if not active_consents:
                self._providers[provider_id].access_level = AccessLevel.NONE

        self._log_access(
            provider_id=provider_id,
            action="consent_revoked",
            details={"consent_id": consent_id},
        )

        return True

    def check_consent(
        self,
        provider_id: str,
        data_type: str,
    ) -> tuple[bool, AccessLevel]:
        """
        Check if a provider has consent to access specific data.

        Args:
            provider_id: ID of the provider.
            data_type: Type of data being accessed.

        Returns:
            Tuple of (has_consent, access_level).
        """
        now = time.time()

        for consent in self._consents.values():
            if consent.provider_id != provider_id:
                continue

            if consent.status != ConsentStatus.GRANTED:
                continue

            if consent.expires_at and consent.expires_at < now:
                consent.status = ConsentStatus.EXPIRED
                continue

            if data_type in consent.data_types or "all" in consent.data_types:
                return True, consent.access_level

        return False, AccessLevel.NONE

    def generate_summary_report(
        self,
        provider_id: str,
        report_type: str,
        data: dict[str, Any],
        anonymize: bool = True,
    ) -> SharedReport | None:
        """
        Generate a summary report for a provider.

        Args:
            provider_id: ID of the provider.
            report_type: Type of report.
            data: Data to include in the report.
            anonymize: Whether to anonymize the data.

        Returns:
            The created SharedReport or None if no consent.
        """
        has_consent, access_level = self.check_consent(provider_id, report_type)

        if not has_consent:
            return None

        # Anonymize if required
        content = self._anonymize_data(data) if anonymize else data

        # Filter based on access level
        content = self._filter_by_access_level(content, access_level)

        report = SharedReport(
            report_id=str(uuid.uuid4()),
            provider_id=provider_id,
            report_type=report_type,
            content=content,
            anonymized=anonymize,
        )

        self._reports[report.report_id] = report

        self._log_access(
            provider_id=provider_id,
            action="report_generated",
            details={"report_id": report.report_id, "report_type": report_type},
        )

        return report

    def _anonymize_data(self, data: dict[str, Any]) -> dict[str, Any]:
        """Anonymize sensitive data in a report."""
        anonymized = {}

        # Fields to anonymize
        sensitive_fields = {"name", "email", "phone", "address", "ssn", "dob"}

        for key, value in data.items():
            if key.lower() in sensitive_fields:
                anonymized[key] = "[REDACTED]"
            elif isinstance(value, dict):
                anonymized[key] = self._anonymize_data(value)
            elif isinstance(value, list):
                anonymized[key] = [
                    self._anonymize_data(v) if isinstance(v, dict) else v
                    for v in value
                ]
            else:
                anonymized[key] = value

        return anonymized

    def _filter_by_access_level(
        self,
        data: dict[str, Any],
        access_level: AccessLevel,
    ) -> dict[str, Any]:
        """Filter data based on access level."""
        if access_level == AccessLevel.FULL:
            return data

        if access_level == AccessLevel.DETAILED:
            # Remove raw data, keep patterns
            filtered = {k: v for k, v in data.items() if not k.startswith("raw_")}
            return filtered

        if access_level == AccessLevel.SUMMARY:
            # Only include summary statistics
            summary_keys = {"summary", "statistics", "trends", "risk_level", "recommendations"}
            return {k: v for k, v in data.items() if k in summary_keys}

        if access_level == AccessLevel.CRISIS_ONLY:
            # Only crisis-relevant information
            crisis_keys = {"crisis_level", "risk_assessment", "emergency_contacts"}
            return {k: v for k, v in data.items() if k in crisis_keys}

        return {}

    def trigger_crisis_alert(
        self,
        severity: str,
        message: str,
        entropy_level: float,
        provider_ids: list[str] | None = None,
    ) -> CrisisAlert | None:
        """
        Trigger a crisis alert to designated providers.

        Args:
            severity: Severity level ("low", "medium", "high", "critical").
            message: Alert message.
            entropy_level: Current entropy level.
            provider_ids: Specific providers to alert (None = all with crisis access).

        Returns:
            The created CrisisAlert or None if alerts disabled.
        """
        if not self.enable_crisis_alerts:
            return None

        # Determine which providers to alert
        if provider_ids is None:
            provider_ids = [
                p.provider_id for p in self._providers.values()
                if p.access_level in [AccessLevel.CRISIS_ONLY, AccessLevel.SUMMARY,
                                      AccessLevel.DETAILED, AccessLevel.FULL]
            ]

        if not provider_ids:
            return None

        alert = CrisisAlert(
            alert_id=str(uuid.uuid4()),
            user_id=self.user_id,
            provider_ids=provider_ids,
            severity=severity,
            message=message,
            entropy_level=entropy_level,
        )

        self._alerts.append(alert)

        self._log_access(
            provider_id="system",
            action="crisis_alert_triggered",
            details={
                "alert_id": alert.alert_id,
                "severity": severity,
                "providers_notified": len(provider_ids),
            },
        )

        return alert

    def acknowledge_alert(self, alert_id: str, provider_id: str) -> bool:
        """
        Acknowledge a crisis alert.

        Args:
            alert_id: ID of the alert.
            provider_id: ID of the acknowledging provider.

        Returns:
            True if acknowledgment successful.
        """
        for alert in self._alerts:
            if alert.alert_id == alert_id:
                if provider_id not in alert.acknowledged_by:
                    alert.acknowledged_by.append(provider_id)

                    self._log_access(
                        provider_id=provider_id,
                        action="alert_acknowledged",
                        details={"alert_id": alert_id},
                    )

                return True

        return False

    def resolve_alert(self, alert_id: str, provider_id: str) -> bool:
        """
        Mark a crisis alert as resolved.

        Args:
            alert_id: ID of the alert.
            provider_id: ID of the resolving provider.

        Returns:
            True if resolution successful.
        """
        for alert in self._alerts:
            if alert.alert_id == alert_id:
                alert.resolved_at = time.time()

                self._log_access(
                    provider_id=provider_id,
                    action="alert_resolved",
                    details={"alert_id": alert_id},
                )

                return True

        return False

    def get_active_alerts(self) -> list[CrisisAlert]:
        """Get all unresolved crisis alerts."""
        return [a for a in self._alerts if a.resolved_at is None]

    def _log_access(
        self,
        provider_id: str,
        action: str,
        details: dict[str, Any],
    ) -> None:
        """Log an access event."""
        self._access_log.append({
            "timestamp": time.time(),
            "provider_id": provider_id,
            "action": action,
            "details": details,
        })

    def get_access_log(
        self,
        provider_id: str | None = None,
        since_timestamp: float | None = None,
    ) -> list[dict[str, Any]]:
        """
        Get the access log.

        Args:
            provider_id: Filter by provider ID.
            since_timestamp: Only return entries after this timestamp.

        Returns:
            List of access log entries.
        """
        log = self._access_log

        if provider_id:
            log = [e for e in log if e["provider_id"] == provider_id]

        if since_timestamp:
            log = [e for e in log if e["timestamp"] > since_timestamp]

        return log

    def generate_progress_report(
        self,
        provider_id: str,
        metrics: dict[str, Any],
        period_days: int = 30,
    ) -> SharedReport | None:
        """
        Generate a progress report for therapeutic goals.

        Args:
            provider_id: ID of the provider.
            metrics: Progress metrics to include.
            period_days: Reporting period in days.

        Returns:
            The created SharedReport or None if no consent.
        """
        has_consent, access_level = self.check_consent(provider_id, "progress_report")

        if not has_consent:
            return None

        # Structure the progress report
        content = {
            "period_days": period_days,
            "summary": {
                "entropy_trend": metrics.get("entropy_trend", "stable"),
                "crisis_events": metrics.get("crisis_count", 0),
                "stability_score": metrics.get("stability_score", 0.5),
            },
            "trends": {
                "emotional_regulation": metrics.get("emotional_regulation_trend", []),
                "grounding_effectiveness": metrics.get("grounding_effectiveness", []),
            },
            "recommendations": self._generate_recommendations(metrics),
            "disclaimer": (
                "This report is generated by an AI support system and is not "
                "a clinical assessment. All treatment decisions should be made "
                "by qualified mental health professionals."
            ),
        }

        return self.generate_summary_report(
            provider_id=provider_id,
            report_type="progress_report",
            data=content,
            anonymize=False,  # Progress reports are already structured
        )

    def _generate_recommendations(self, metrics: dict[str, Any]) -> list[str]:
        """Generate recommendations based on metrics."""
        recommendations = []

        entropy_trend = metrics.get("entropy_trend", "stable")
        if entropy_trend == "increasing":
            recommendations.append(
                "Consider increasing session frequency or implementing "
                "additional grounding strategies."
            )

        crisis_count = metrics.get("crisis_count", 0)
        if crisis_count > 2:
            recommendations.append(
                "Multiple crisis events detected. Safety planning review "
                "may be beneficial."
            )

        stability_score = metrics.get("stability_score", 0.5)
        if stability_score > 0.7:
            recommendations.append(
                "Stability metrics are positive. Consider exploring "
                "deeper therapeutic work if appropriate."
            )

        return recommendations

    def export_consent_records(self) -> list[dict[str, Any]]:
        """
        Export all consent records for user review.

        Returns:
            List of consent records as dictionaries.
        """
        return [
            {
                "consent_id": c.consent_id,
                "provider_name": self._providers.get(c.provider_id, {}).name
                    if c.provider_id in self._providers else "Unknown",
                "access_level": c.access_level.value,
                "data_types": c.data_types,
                "status": c.status.value,
                "granted_at": c.granted_at,
                "expires_at": c.expires_at,
                "revoked_at": c.revoked_at,
                "conditions": c.conditions,
            }
            for c in self._consents.values()
        ]
