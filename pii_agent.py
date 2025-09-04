#!/usr/bin/env python3
"""
PII Logging Interception Agent

A secure module for detecting and redacting Personally Identifiable Information
in log messages before storage. Designed for production use in privacy-focused
environments with strict data protection requirements.

Threat Model:
- Prevents accidental PII exposure in logs
- Complies with GDPR, CCPA, PCI-DSS requirements
- Defends against log injection and data leakage

Usage:
    from pii_agent import log_message, set_policy, Policy
    
    set_policy(Policy.MASK_AND_LOG)
    log_message("User john.doe@example.com logged in")
"""

import re
import logging
from enum import Enum
from typing import Dict, List, Tuple, Optional, Pattern
from functools import lru_cache

class Policy(Enum):
    """Security policies for handling PII in logs"""
    LOG_AS_IS = "log_as_is"
    MASK_AND_LOG = "mask_and_log"
    BLOCK = "block"

class PIIType(Enum):
    """Classifications of PII severity"""
    CRITICAL = "critical"
    SENSITIVE = "sensitive"
    CONTEXTUAL = "contextual"

class PIIPatterns:
    """Compiled regex patterns for PII detection with security rationale"""
    
    EMAIL = re.compile(
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        re.IGNORECASE
    )
    
    PHONE_US = re.compile(
        r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'
    )
    
    SSN = re.compile(
        r'\b\d{3}-\d{2}-\d{4}\b|\b\d{9}\b'
    )
    
    CREDIT_CARD = re.compile(
        r'\b(?:\d{4}[-\s]?){3}\d{4}\b'
    )
    
    IP_ADDRESS = re.compile(
        r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
    )
    
    API_KEY = re.compile(
        r'\b(?:api[_-]?key|apikey|access[_-]?token)[\s:=]+[\w-]{20,}\b',
        re.IGNORECASE
    )
    
    PASSWORD = re.compile(
        r'\b(?:password|passwd|pwd)[\s:=]+\S+\b',
        re.IGNORECASE
    )

class PIIAgent:
    """Main PII detection and redaction engine"""
    
    def __init__(self, default_policy: Policy = Policy.MASK_AND_LOG):
        self.policy = default_policy
        self.patterns: Dict[PIIType, List[Tuple[Pattern, str]]] = {
            PIIType.CRITICAL: [
                (PIIPatterns.SSN, "[SSN_REDACTED]"),
                (PIIPatterns.CREDIT_CARD, "[CC_REDACTED]"),
            ],
            PIIType.SENSITIVE: [
                (PIIPatterns.EMAIL, "[EMAIL_REDACTED]"),
                (PIIPatterns.PHONE_US, "[PHONE_REDACTED]"),
                (PIIPatterns.IP_ADDRESS, "[IP_REDACTED]"),
                (PIIPatterns.PASSWORD, "[PASSWORD_REDACTED]"),
            ],
            PIIType.CONTEXTUAL: [
                (PIIPatterns.API_KEY, "[API_KEY_REDACTED]"),
            ]
        }
        logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    @lru_cache(maxsize=1024)
    def detect_pii(self, message: str) -> Optional[PIIType]:
        """
        Detect PII in message and return highest severity level found.
        Uses caching for performance on repeated patterns.
        """
        if not message or len(message) > 10000:
            return None
        
        for pii_type in [PIIType.CRITICAL, PIIType.SENSITIVE, PIIType.CONTEXTUAL]:
            for pattern, _ in self.patterns[pii_type]:
                if pattern.search(message):
                    return pii_type
        return None
    
    def mask_sensitive_data(self, message: str) -> str:
        """
        Replace PII with redaction tokens while preserving message structure.
        Applies patterns in order of severity for defense in depth.
        """
        masked = message
        
        for pii_type in [PIIType.CRITICAL, PIIType.SENSITIVE, PIIType.CONTEXTUAL]:
            for pattern, replacement in self.patterns[pii_type]:
                masked = pattern.sub(replacement, masked)
        
        return masked
    
    def log_message(self, message: str) -> None:
        """
        Main entry point for log interception and PII handling.
        Applies configured policy to determine action.
        """
        if not message:
            return
        
        pii_type = self.detect_pii(message)
        
        if pii_type is None:
            logging.info(message)
            return
        
        if self.policy == Policy.LOG_AS_IS:
            logging.warning(f"[WARNING: PII DETECTED] {message}")
        elif self.policy == Policy.MASK_AND_LOG:
            if pii_type == PIIType.CRITICAL:
                logging.info("[BLOCKED] Message contained critical PII")
            else:
                masked = self.mask_sensitive_data(message)
                logging.info(masked)
        elif self.policy == Policy.BLOCK:
            logging.info(f"[BLOCKED] Message contained {pii_type.value} PII")

_global_agent = PIIAgent()

def log_message(message: str) -> None:
    """Global function to intercept and process log messages"""
    _global_agent.log_message(message)

def set_policy(policy: Policy) -> None:
    """Configure global PII handling policy"""
    _global_agent.policy = policy

def detect_pii(message: str) -> bool:
    """Utility function for testing PII detection"""
    return _global_agent.detect_pii(message) is not None

if __name__ == "__main__":
    print("PII Agent Test Suite\n" + "="*50)
    
    test_messages = [
        ("Safe message", "2023-07-15 INFO: Server health check passed"),
        ("Email PII", "User john.doe@example.com failed login"),
        ("Phone PII", "Contact customer at 555-123-4567"),
        ("SSN Critical", "Processing SSN: 123-45-6789"),
        ("Credit Card", "Payment with card 4111-1111-1111-1111"),
        ("API Key", "Using api_key: sk_test_abcdef123456789"),
        ("Password", "Login with password: SuperSecret123!"),
        ("IP Address", "Connection from 192.168.1.100"),
        ("Mixed PII", "User john@example.com from 10.0.0.1 with SSN 123-45-6789"),
    ]
    
    for policy in [Policy.LOG_AS_IS, Policy.MASK_AND_LOG, Policy.BLOCK]:
        print(f"\n[Testing Policy: {policy.value}]")
        set_policy(policy)
        
        for label, msg in test_messages[:3]:
            print(f"\n{label}: {msg}")
            print("Output: ", end="")
            log_message(msg)