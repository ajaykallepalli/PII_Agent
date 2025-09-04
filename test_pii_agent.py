#!/usr/bin/env python3
"""
Comprehensive test suite for PII Agent
Tests detection accuracy, performance, and edge cases
"""

import unittest
import time
from io import StringIO
import logging
import sys
from pii_agent import PIIAgent, Policy, PIIType, detect_pii, log_message, set_policy

class TestPIIDetection(unittest.TestCase):
    """Test PII pattern detection accuracy"""
    
    def setUp(self):
        self.agent = PIIAgent()
    
    def test_email_detection(self):
        """Test email address detection patterns"""
        test_cases = [
            ("john.doe@example.com", True),
            ("user+tag@domain.co.uk", True),
            ("admin@192.168.1.1", True),
            ("not.an.email", False),
            ("@example.com", False),
            ("user@", False),
        ]
        
        for text, expected in test_cases:
            with self.subTest(text=text):
                result = self.agent.detect_pii(text) is not None
                self.assertEqual(result, expected, f"Failed for: {text}")
    
    def test_phone_detection(self):
        """Test phone number detection patterns"""
        test_cases = [
            ("555-123-4567", True),
            ("(555) 123-4567", True),
            ("+1 555 123 4567", True),
            ("5551234567", True),
            ("123-45", False),
            ("12345", False),
        ]
        
        for text, expected in test_cases:
            with self.subTest(text=text):
                result = self.agent.detect_pii(text) is not None
                self.assertEqual(result, expected, f"Failed for: {text}")
    
    def test_ssn_detection(self):
        """Test SSN detection with critical severity"""
        test_cases = [
            ("123-45-6789", PIIType.CRITICAL),
            ("123456789", PIIType.CRITICAL),
            ("12-345-6789", None),
            ("1234-5678", None),
        ]
        
        for text, expected in test_cases:
            with self.subTest(text=text):
                result = self.agent.detect_pii(text)
                self.assertEqual(result, expected, f"Failed for: {text}")
    
    def test_credit_card_detection(self):
        """Test credit card number patterns"""
        test_cases = [
            ("4111-1111-1111-1111", PIIType.CRITICAL),
            ("4111 1111 1111 1111", PIIType.CRITICAL),
            ("4111111111111111", PIIType.CRITICAL),
            ("1234-5678-9012", None),
        ]
        
        for text, expected in test_cases:
            with self.subTest(text=text):
                result = self.agent.detect_pii(text)
                self.assertEqual(result, expected, f"Failed for: {text}")
    
    def test_mixed_pii(self):
        """Test messages with multiple PII types"""
        message = "User john@example.com with SSN 123-45-6789 called from 555-123-4567"
        result = self.agent.detect_pii(message)
        self.assertEqual(result, PIIType.CRITICAL)

class TestRedaction(unittest.TestCase):
    """Test PII masking and redaction"""
    
    def setUp(self):
        self.agent = PIIAgent()
    
    def test_email_masking(self):
        """Test email address masking"""
        original = "Contact john.doe@example.com for details"
        masked = self.agent.mask_sensitive_data(original)
        self.assertEqual(masked, "Contact [EMAIL_REDACTED] for details")
    
    def test_multiple_redactions(self):
        """Test masking multiple PII in one message"""
        original = "Email: test@example.com Phone: 555-123-4567"
        masked = self.agent.mask_sensitive_data(original)
        self.assertIn("[EMAIL_REDACTED]", masked)
        self.assertIn("[PHONE_REDACTED]", masked)
        self.assertNotIn("test@example.com", masked)
        self.assertNotIn("555-123-4567", masked)
    
    def test_preserve_structure(self):
        """Test that non-PII content is preserved"""
        original = "2023-07-15 ERROR: User john@example.com failed login attempt #3"
        masked = self.agent.mask_sensitive_data(original)
        self.assertIn("2023-07-15 ERROR:", masked)
        self.assertIn("failed login attempt #3", masked)
        self.assertIn("[EMAIL_REDACTED]", masked)

class TestPolicies(unittest.TestCase):
    """Test different security policies"""
    
    def setUp(self):
        self.agent = PIIAgent()
        self.old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        logger = logging.getLogger()
        self.log_stream = StringIO()
        handler = logging.StreamHandler(self.log_stream)
        handler.setLevel(logging.INFO)
        logger.handlers = [handler]
    
    def tearDown(self):
        sys.stdout = self.old_stdout
    
    def test_log_as_is_policy(self):
        """Test LOG_AS_IS policy behavior"""
        self.agent.policy = Policy.LOG_AS_IS
        self.agent.log_message("Email: test@example.com")
        output = self.log_stream.getvalue()
        self.assertIn("WARNING: PII DETECTED", output)
        self.assertIn("test@example.com", output)
    
    def test_mask_and_log_policy(self):
        """Test MASK_AND_LOG policy behavior"""
        self.agent.policy = Policy.MASK_AND_LOG
        self.agent.log_message("Email: test@example.com")
        output = self.log_stream.getvalue()
        self.assertIn("[EMAIL_REDACTED]", output)
        self.assertNotIn("test@example.com", output)
    
    def test_block_policy(self):
        """Test BLOCK policy behavior"""
        self.agent.policy = Policy.BLOCK
        self.agent.log_message("Email: test@example.com")
        output = self.log_stream.getvalue()
        self.assertIn("[BLOCKED]", output)
        self.assertNotIn("test@example.com", output)
    
    def test_critical_pii_blocking(self):
        """Test that critical PII is always blocked in MASK_AND_LOG"""
        self.agent.policy = Policy.MASK_AND_LOG
        self.agent.log_message("SSN: 123-45-6789")
        output = self.log_stream.getvalue()
        self.assertIn("[BLOCKED]", output)
        self.assertIn("critical", output.lower())

class TestPerformance(unittest.TestCase):
    """Test performance requirements"""
    
    def setUp(self):
        self.agent = PIIAgent()
    
    def test_detection_speed(self):
        """Ensure detection is under 5ms per message"""
        message = "User john.doe@example.com logged in from 192.168.1.100"
        
        start = time.perf_counter()
        for _ in range(100):
            self.agent.detect_pii(message)
        elapsed = time.perf_counter() - start
        
        avg_time_ms = (elapsed / 100) * 1000
        self.assertLess(avg_time_ms, 5, f"Detection too slow: {avg_time_ms:.2f}ms")
    
    def test_cache_effectiveness(self):
        """Test LRU cache improves performance"""
        message = "test@example.com"
        
        start = time.perf_counter()
        self.agent.detect_pii(message)
        first_call = time.perf_counter() - start
        
        start = time.perf_counter()
        self.agent.detect_pii(message)
        cached_call = time.perf_counter() - start
        
        self.assertLess(cached_call, first_call * 0.5)

class TestEdgeCases(unittest.TestCase):
    """Test edge cases and security concerns"""
    
    def setUp(self):
        self.agent = PIIAgent()
    
    def test_empty_message(self):
        """Test handling of empty messages"""
        self.agent.log_message("")
        self.agent.log_message(None)
    
    def test_very_long_message(self):
        """Test handling of very long messages"""
        long_msg = "a" * 20000
        result = self.agent.detect_pii(long_msg)
        self.assertIsNone(result)
    
    def test_unicode_handling(self):
        """Test Unicode characters don't break detection"""
        messages = [
            "User 李明@example.com logged in",
            "Контакт: test@тест.com",
            "📧 john@example.com",
        ]
        
        for msg in messages:
            try:
                self.agent.detect_pii(msg)
                self.agent.mask_sensitive_data(msg)
            except Exception as e:
                self.fail(f"Unicode handling failed: {e}")
    
    def test_regex_dos_prevention(self):
        """Test protection against regex DoS attacks"""
        evil_input = "a" * 1000 + "@" * 1000
        
        start = time.perf_counter()
        self.agent.detect_pii(evil_input)
        elapsed = time.perf_counter() - start
        
        self.assertLess(elapsed, 0.1, "Potential regex DoS vulnerability")
    
    def test_false_positives(self):
        """Test common false positive scenarios"""
        safe_messages = [
            "Error code: 404-12-3456",
            "Version 1.2.3.4",
            "Port number: 8080",
            "UUID: 550e8400-e29b-41d4-a716-446655440000",
        ]
        
        for msg in safe_messages:
            with self.subTest(msg=msg):
                result = self.agent.detect_pii(msg)
                if result and result != PIIType.CONTEXTUAL:
                    self.fail(f"False positive for: {msg}")

class TestIntegration(unittest.TestCase):
    """Integration tests with global functions"""
    
    def setUp(self):
        self.old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        logger = logging.getLogger()
        self.log_stream = StringIO()
        handler = logging.StreamHandler(self.log_stream)
        handler.setLevel(logging.INFO)
        logger.handlers = [handler]
    
    def tearDown(self):
        sys.stdout = self.old_stdout
    
    def test_global_functions(self):
        """Test global function interface"""
        set_policy(Policy.MASK_AND_LOG)
        
        self.assertTrue(detect_pii("test@example.com"))
        self.assertFalse(detect_pii("safe message"))
        
        log_message("User test@example.com logged in")
        output = self.log_stream.getvalue()
        self.assertIn("[EMAIL_REDACTED]", output)

if __name__ == "__main__":
    unittest.main(verbosity=2)