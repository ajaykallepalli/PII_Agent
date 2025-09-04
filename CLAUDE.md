# PII Logging Interception Agent

## Project Overview
A Python module that intercepts and analyzes log messages for Personally Identifiable Information (PII) before storage, preventing sensitive data exposure in plaintext logs.

## Role & Context
- **Developer Role**: Senior backend developer and security engineer
- **Industry**: Privacy-focused fintech company
- **Focus**: Secure, production-ready code following data protection regulations

## Core Requirements

### Technical Specifications
- **Language**: Python 3.10+
- **Dependencies**: Standard library only (no third-party packages)
- **Main Function**: `log_message(message: str) -> None`
- **Detection Targets**: Email addresses, phone numbers, names, SSNs, credit cards, etc.

### Security Policies
1. **LOG_AS_IS**: Pass benign messages through unmodified
2. **MASK_AND_LOG**: Redact sensitive portions while preserving structure
3. **BLOCK**: Completely discard messages with critical PII

### Design Constraints
- Single-file module when possible
- Type-hinted functions following PEP8
- No intermediate storage of sensitive data
- Lightweight and performant regex patterns
- Comprehensive security comments

## PII Detection Patterns

### Critical PII (Block by default)
- Social Security Numbers (SSN)
- Credit Card Numbers
- Bank Account Numbers
- Driver's License Numbers
- Passport Numbers

### Sensitive PII (Mask by default)
- Email addresses
- Phone numbers
- IP addresses
- Names (contextual detection)
- Dates of birth
- Addresses

### Contextual PII (Configurable)
- User IDs
- Session tokens
- API keys
- URLs with sensitive parameters

## Testing Strategy

### Unit Tests
- Test each PII pattern independently
- Verify false positive rates
- Benchmark performance

### Integration Tests
- Real log message samples
- Policy configuration testing
- Edge case handling

### Security Tests
- Regex DoS prevention
- Unicode handling
- Injection resistance

## Code Structure

```
pii_agent.py
├── Configuration
│   ├── PolicyEnum
│   └── PIIPatterns
├── Detection Layer
│   ├── detect_pii()
│   └── classify_pii_type()
├── Redaction Engine
│   ├── mask_sensitive_data()
│   └── generate_mask()
└── Main Pipeline
    └── log_message()
```

## Example Usage

```python
from pii_agent import log_message, set_policy, Policy

# Configure policy
set_policy(Policy.MASK_AND_LOG)

# Safe message - passes through
log_message("2023-07-15 INFO: Server health check passed")

# PII detected - gets masked
log_message("2023-07-15 ERROR: User john.doe@example.com failed login")
# Output: "2023-07-15 ERROR: User [EMAIL_REDACTED] failed login"

# Critical PII - gets blocked
log_message("SSN: 123-45-6789 was processed")
# Output: "[BLOCKED] Message contained critical PII"
```

## Testing Commands
```bash
# Run unit tests
python -m pytest test_pii_agent.py -v

# Run with example logs
python pii_agent.py --test-mode

# Benchmark performance
python -m timeit -s "from pii_agent import log_message" "log_message('test@example.com')"
```

## Security Considerations

### Threat Model
- **Internal Threats**: Developers accidentally logging sensitive data
- **Compliance**: GDPR, CCPA, PCI-DSS requirements
- **Data Leakage**: Preventing PII in logs, monitoring systems, backups

### Mitigation Strategies
1. Defense in depth - multiple detection layers
2. Fail-safe defaults - block when uncertain
3. Audit logging - track redaction events
4. Regular pattern updates
5. Performance monitoring to prevent DoS

## Future Enhancements
- ML-based PII detection
- Contextual analysis
- Multilingual support
- Custom PII patterns via configuration
- Integration with logging frameworks
- Metrics and alerting

## Compliance Notes
- GDPR Article 32: Technical measures for data protection
- PCI-DSS 3.4: Render PAN unreadable
- HIPAA: Safeguards for PHI
- CCPA: Consumer data protection

## Review Checklist
- [ ] All PII patterns tested
- [ ] False positive rate < 1%
- [ ] Performance impact < 5ms per message
- [ ] Security comments added
- [ ] Edge cases handled
- [ ] Documentation complete