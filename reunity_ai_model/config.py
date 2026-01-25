"""
ReUnity Configuration Module
============================

Centralized configuration for all ReUnity components.

DISCLAIMER: ReUnity is NOT a clinical or treatment tool.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum


class Environment(Enum):
    """Deployment environment."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"


@dataclass
class EntropyConfig:
    """Configuration for entropy analysis."""
    
    # Entropy state thresholds
    crisis_threshold: float = 0.85
    high_threshold: float = 0.70
    moderate_threshold: float = 0.50
    low_threshold: float = 0.30
    stable_threshold: float = 0.15
    
    # Lyapunov stability thresholds
    chaotic_threshold: float = 0.5
    unstable_threshold: float = 0.1
    marginal_threshold: float = 0.0
    stable_lyapunov_threshold: float = -0.1
    very_stable_threshold: float = -0.5
    
    # Analysis parameters
    history_size: int = 100
    trend_window: int = 5
    epsilon: float = 1e-10


@dataclass
class MemoryConfig:
    """Configuration for continuity memory store."""
    
    # RIME weights
    alpha_episodic: float = 0.4
    beta_semantic: float = 0.35
    gamma_context: float = 0.25
    
    # Memory parameters
    max_memories: int = 10000
    default_consent_scope: str = "self_only"
    
    # Decay parameters
    episodic_decay_days: float = 7.0
    semantic_decay_days: float = 30.0


@dataclass
class RegimeConfig:
    """Configuration for regime controller."""
    
    # Regime thresholds
    apostasis_threshold: float = 0.3
    regeneration_threshold: float = 0.5
    divergence_constraint: float = 0.7
    
    # Utility calculation weights
    recency_weight: float = 0.4
    intensity_weight: float = 0.4
    link_weight: float = 0.2
    
    # Pruning parameters
    utility_threshold: float = 0.2
    max_pruned_features: int = 1000


@dataclass
class SecurityConfig:
    """Configuration for security and encryption."""
    
    # Encryption
    encryption_algorithm: str = "AES-256-GCM"
    key_derivation: str = "PBKDF2-SHA256"
    key_iterations: int = 100000
    
    # Quantum-resistant (future-proofing)
    enable_quantum_resistant: bool = False
    kyber_security_level: int = 3
    
    # Session
    session_timeout_minutes: int = 60
    max_failed_attempts: int = 5


@dataclass
class APIConfig:
    """Configuration for API server."""
    
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    
    # CORS
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    cors_methods: List[str] = field(default_factory=lambda: ["*"])
    
    # Rate limiting
    rate_limit_per_minute: int = 60
    
    # Documentation
    docs_url: str = "/docs"
    redoc_url: str = "/redoc"


@dataclass
class GroundingConfig:
    """Configuration for grounding techniques."""
    
    # Recommendation parameters
    max_recommendations: int = 3
    personalization_weight: float = 0.5
    
    # Technique categories
    categories: List[str] = field(default_factory=lambda: [
        "sensory", "breathing", "physical", "mindfulness",
        "visualization", "cognitive", "bilateral"
    ])


@dataclass
class PatternConfig:
    """Configuration for pattern recognition."""
    
    # Detection thresholds
    confidence_threshold: float = 0.5
    high_confidence_threshold: float = 0.8
    
    # History
    max_detection_history: int = 500
    
    # Alert settings
    alert_on_high_confidence: bool = True
    alert_cooldown_minutes: int = 30


@dataclass
class AlterConfig:
    """Configuration for alter-aware subsystem."""
    
    # Profile limits
    max_alter_profiles: int = 100
    
    # Switch detection
    switch_detection_enabled: bool = True
    switch_cooldown_seconds: int = 60
    
    # Communication
    enable_inter_alter_messaging: bool = True
    message_retention_days: int = 30


@dataclass
class ClinicianConfig:
    """Configuration for clinician interface."""
    
    # Access control
    require_consent_for_access: bool = True
    audit_all_access: bool = True
    
    # Session limits
    max_concurrent_sessions: int = 10
    session_timeout_minutes: int = 120
    
    # Data sharing
    allow_anonymized_export: bool = True
    require_explicit_consent: bool = True


@dataclass
class ExportConfig:
    """Configuration for data export."""
    
    # Bundle settings
    include_provenance: bool = True
    include_hash: bool = True
    hash_algorithm: str = "SHA-256"
    
    # Anonymization
    anonymize_by_default: bool = False
    anonymization_salt_length: int = 32


@dataclass
class ReUnityConfig:
    """Master configuration for ReUnity system."""
    
    # Environment
    environment: Environment = Environment.DEVELOPMENT
    version: str = "1.0.0"
    
    # Component configs
    entropy: EntropyConfig = field(default_factory=EntropyConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    api: APIConfig = field(default_factory=APIConfig)
    grounding: GroundingConfig = field(default_factory=GroundingConfig)
    pattern: PatternConfig = field(default_factory=PatternConfig)
    alter: AlterConfig = field(default_factory=AlterConfig)
    clinician: ClinicianConfig = field(default_factory=ClinicianConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Storage
    data_directory: str = "./reunity_data"
    enable_local_storage: bool = True
    
    @classmethod
    def from_environment(cls) -> "ReUnityConfig":
        """Create configuration from environment variables."""
        config = cls()
        
        # Override from environment
        env = os.getenv("REUNITY_ENV", "development")
        config.environment = Environment(env)
        
        if config.environment == Environment.PRODUCTION:
            config.api.debug = False
            config.security.enable_quantum_resistant = True
            config.log_level = "WARNING"
        
        # API settings
        config.api.host = os.getenv("REUNITY_HOST", config.api.host)
        config.api.port = int(os.getenv("REUNITY_PORT", config.api.port))
        
        # Data directory
        config.data_directory = os.getenv(
            "REUNITY_DATA_DIR", 
            config.data_directory
        )
        
        return config
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary."""
        from dataclasses import asdict
        result = asdict(self)
        result["environment"] = self.environment.value
        return result


# Global configuration instance
_config: Optional[ReUnityConfig] = None


def get_config() -> ReUnityConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = ReUnityConfig.from_environment()
    return _config


def set_config(config: ReUnityConfig) -> None:
    """Set the global configuration instance."""
    global _config
    _config = config


# Convenience exports
__all__ = [
    "Environment",
    "EntropyConfig",
    "MemoryConfig",
    "RegimeConfig",
    "SecurityConfig",
    "APIConfig",
    "GroundingConfig",
    "PatternConfig",
    "AlterConfig",
    "ClinicianConfig",
    "ExportConfig",
    "ReUnityConfig",
    "get_config",
    "set_config",
]
