#!/usr/bin/env python3
"""
Comprehensive test suite for Advanced PII Detection Agent
Tests NER, Proximity Analysis, Graph Theory components and integration
"""

import unittest
import tempfile
import json
import pandas as pd
from pathlib import Path
import shutil
import numpy as np
import networkx as nx
from unittest.mock import patch, MagicMock

from advanced_pii_agent import (
    PIIType, RiskLevel, PIIEntity, PIINERDetector, 
    ProximityAnalyzer, PIIGraphBuilder, AdvancedPIIAgent
)

class TestPIIEntity(unittest.TestCase):
    """Test PIIEntity data structure"""
    
    def test_pii_entity_creation(self):
        """Test creating PIIEntity with all fields"""
        entity = PIIEntity(
            text="john.doe@example.com",
            pii_type=PIIType.EMAIL,
            start_pos=0,
            end_pos=19,
            confidence=0.9,
            context="User email: john.doe@example.com",
            row_index=1,
            column_name="email",
            risk_level=RiskLevel.MEDIUM
        )
        
        self.assertEqual(entity.text, "john.doe@example.com")
        self.assertEqual(entity.pii_type, PIIType.EMAIL)
        self.assertEqual(entity.risk_level, RiskLevel.MEDIUM)
        self.assertEqual(len(entity.related_entities), 0)
    
    def test_pii_entity_to_dict(self):
        """Test converting PIIEntity to dictionary"""
        entity = PIIEntity(
            text="123-45-6789",
            pii_type=PIIType.SSN,
            start_pos=0,
            end_pos=11,
            confidence=0.95,
            context="SSN: 123-45-6789",
            risk_level=RiskLevel.CRITICAL
        )
        
        entity_dict = entity.to_dict()
        
        self.assertIsInstance(entity_dict, dict)
        self.assertEqual(entity_dict['pii_type'], 'social_security_number')
        self.assertEqual(entity_dict['risk_level'], 'critical')
        self.assertEqual(entity_dict['confidence'], 0.95)

class TestPIINERDetector(unittest.TestCase):
    """Test NER-based PII detection"""
    
    def setUp(self):
        """Setup test environment"""
        # Mock spaCy to avoid model loading in tests
        with patch('advanced_pii_agent.spacy') as mock_spacy:
            mock_nlp = MagicMock()
            mock_spacy.load.return_value = mock_nlp
            self.detector = PIINERDetector()
    
    def test_regex_email_detection(self):
        """Test email detection using regex patterns"""
        text = "Contact john.doe@example.com for more information"
        entities = self.detector.detect_pii_in_text(text)
        
        email_entities = [e for e in entities if e.pii_type == PIIType.EMAIL]
        self.assertEqual(len(email_entities), 1)
        self.assertEqual(email_entities[0].text, "john.doe@example.com")
        self.assertEqual(email_entities[0].start_pos, 8)
        self.assertEqual(email_entities[0].detection_method, "regex")
    
    def test_regex_phone_detection(self):
        """Test phone number detection"""
        test_cases = [
            "Call me at 555-123-4567",
            "Phone: (555) 123-4567",
            "Mobile: +1 555 123 4567",
            "Number: 5551234567"
        ]
        
        for text in test_cases:
            with self.subTest(text=text):
                entities = self.detector.detect_pii_in_text(text)
                phone_entities = [e for e in entities if e.pii_type == PIIType.PHONE]
                self.assertGreater(len(phone_entities), 0, f"Failed to detect phone in: {text}")
    
    def test_regex_ssn_detection(self):
        """Test SSN detection with high confidence"""
        text = "SSN: 123-45-6789"
        entities = self.detector.detect_pii_in_text(text)
        
        ssn_entities = [e for e in entities if e.pii_type == PIIType.SSN]
        self.assertEqual(len(ssn_entities), 1)
        self.assertEqual(ssn_entities[0].text, "123-45-6789")
        self.assertEqual(ssn_entities[0].confidence, 0.9)
    
    def test_credit_card_detection(self):
        """Test credit card number detection"""
        test_cases = [
            "4111-1111-1111-1111",
            "4111 1111 1111 1111",
            "4111111111111111"
        ]
        
        for cc_number in test_cases:
            with self.subTest(cc_number=cc_number):
                text = f"Payment card: {cc_number}"
                entities = self.detector.detect_pii_in_text(text)
                cc_entities = [e for e in entities if e.pii_type == PIIType.CREDIT_CARD]
                self.assertEqual(len(cc_entities), 1)
                self.assertEqual(cc_entities[0].text, cc_number)
    
    def test_multiple_pii_types(self):
        """Test detection of multiple PII types in one text"""
        text = "Contact John Smith at john.smith@company.com or call 555-123-4567"
        entities = self.detector.detect_pii_in_text(text)
        
        pii_types_found = {e.pii_type for e in entities}
        self.assertIn(PIIType.EMAIL, pii_types_found)
        self.assertIn(PIIType.PHONE, pii_types_found)
    
    def test_context_extraction(self):
        """Test context extraction around detected PII"""
        text = "This is a longer text with john.doe@example.com in the middle of it"
        entities = self.detector.detect_pii_in_text(text)
        
        email_entity = entities[0]
        self.assertIn("john.doe@example.com", email_entity.context)
        self.assertGreater(len(email_entity.context), len(email_entity.text))
    
    def test_deduplication(self):
        """Test that overlapping entities are deduplicated"""
        # This would require overlapping patterns - simplified test
        text = "Email: test@example.com"
        entities = self.detector.detect_pii_in_text(text)
        
        # Should not have duplicate entities for the same text span
        positions = [(e.start_pos, e.end_pos) for e in entities]
        self.assertEqual(len(positions), len(set(positions)))
    
    def test_empty_text(self):
        """Test handling of empty or invalid text"""
        test_cases = ["", None, " ", "\n\t"]
        
        for text in test_cases:
            with self.subTest(text=repr(text)):
                entities = self.detector.detect_pii_in_text(text)
                self.assertEqual(len(entities), 0)

class TestProximityAnalyzer(unittest.TestCase):
    """Test proximity analysis functionality"""
    
    def setUp(self):
        """Setup test environment"""
        self.analyzer = ProximityAnalyzer(window_size=50)
    
    def test_proximity_grouping(self):
        """Test grouping of proximate entities"""
        entities = [
            PIIEntity("John Doe", PIIType.PERSON, 0, 8, 0.9, "John Doe works"),
            PIIEntity("john@example.com", PIIType.EMAIL, 15, 31, 0.9, "at john@example.com"),
            PIIEntity("555-123-4567", PIIType.PHONE, 35, 47, 0.9, "call 555-123-4567")
        ]
        
        text = "John Doe works at john@example.com call 555-123-4567"
        analyzed_entities = self.analyzer.analyze_proximity(entities, text)
        
        # All entities should be updated with related entities info
        for entity in analyzed_entities:
            if entity.pii_type == PIIType.PERSON:
                self.assertGreater(len(entity.related_entities), 0)
                self.assertEqual(entity.detection_method, "proximity")
    
    def test_risk_level_update(self):
        """Test risk level updates based on proximity"""
        # High-risk combination: Person + SSN
        entities = [
            PIIEntity("John Smith", PIIType.PERSON, 0, 10, 0.9, "John Smith"),
            PIIEntity("123-45-6789", PIIType.SSN, 20, 31, 0.9, "123-45-6789")
        ]
        
        text = "John Smith SSN: 123-45-6789"
        analyzed_entities = self.analyzer.analyze_proximity(entities, text)
        
        # Person entity should have elevated risk due to proximity to SSN
        person_entity = next(e for e in analyzed_entities if e.pii_type == PIIType.PERSON)
        self.assertIn(person_entity.risk_level, [RiskLevel.HIGH, RiskLevel.CRITICAL])
    
    def test_isolated_entities(self):
        """Test handling of isolated entities (no proximity)"""
        entities = [
            PIIEntity("test@example.com", PIIType.EMAIL, 0, 16, 0.9, "test@example.com"),
            PIIEntity("555-123-4567", PIIType.PHONE, 200, 212, 0.9, "555-123-4567")  # Far apart
        ]
        
        text = "test@example.com" + " " * 180 + "555-123-4567"
        analyzed_entities = self.analyzer.analyze_proximity(entities, text)
        
        # Entities should remain with base risk levels
        for entity in analyzed_entities:
            self.assertEqual(len(entity.related_entities), 0)
    
    def test_risk_matrix_scoring(self):
        """Test risk matrix scoring for different PII combinations"""
        # Test high-risk combination
        high_risk_pair = (PIIType.PERSON, PIIType.SSN)
        high_risk_score = self.analyzer.risk_matrix.get(high_risk_pair, 0)
        self.assertGreaterEqual(high_risk_score, 0.8)
        
        # Test medium-risk combination
        medium_risk_pair = (PIIType.PERSON, PIIType.EMAIL)
        medium_risk_score = self.analyzer.risk_matrix.get(medium_risk_pair, 0)
        self.assertGreaterEqual(medium_risk_score, 0.5)

class TestPIIGraphBuilder(unittest.TestCase):
    """Test graph construction and analysis"""
    
    def setUp(self):
        """Setup test environment"""
        self.graph_builder = PIIGraphBuilder()
    
    def test_graph_construction(self):
        """Test basic graph construction"""
        entities = [
            PIIEntity("John Doe", PIIType.PERSON, 0, 8, 0.9, "John Doe", row_index=0),
            PIIEntity("john@example.com", PIIType.EMAIL, 15, 31, 0.9, "john@example.com", row_index=0),
            PIIEntity("Acme Corp", PIIType.ORGANIZATION, 40, 49, 0.8, "Acme Corp", row_index=1)
        ]
        
        graph = self.graph_builder.build_graph(entities)
        
        self.assertEqual(len(graph.nodes()), 3)
        self.assertGreater(len(graph.edges()), 0)
    
    def test_node_attributes(self):
        """Test node attributes are correctly set"""
        entities = [
            PIIEntity("test@example.com", PIIType.EMAIL, 0, 16, 0.95, "email", 
                     risk_level=RiskLevel.MEDIUM, detection_method="regex")
        ]
        
        graph = self.graph_builder.build_graph(entities)
        node_id = list(graph.nodes())[0]
        node_data = graph.nodes[node_id]
        
        self.assertEqual(node_data['pii_type'], 'email_address')
        self.assertEqual(node_data['confidence'], 0.95)
        self.assertEqual(node_data['risk_level'], 'medium')
        self.assertEqual(node_data['detection_method'], 'regex')
    
    def test_semantic_edges(self):
        """Test semantic relationship edge creation"""
        entities = [
            PIIEntity("John Doe", PIIType.PERSON, 0, 8, 0.9, "John Doe"),
            PIIEntity("john@example.com", PIIType.EMAIL, 15, 31, 0.9, "john@example.com")
        ]
        
        graph = self.graph_builder.build_graph(entities)
        
        # Should have semantic edge between person and email
        edges = list(graph.edges(data=True))
        semantic_edges = [e for e in edges if e[2].get('edge_type') == 'semantic']
        self.assertGreater(len(semantic_edges), 0)
    
    def test_co_occurrence_edges(self):
        """Test co-occurrence edge creation"""
        entities = [
            PIIEntity("John", PIIType.PERSON, 0, 4, 0.9, "John", row_index=0, column_name="name"),
            PIIEntity("jane@example.com", PIIType.EMAIL, 0, 16, 0.9, "jane@example.com", 
                     row_index=0, column_name="email")
        ]
        
        graph = self.graph_builder.build_graph(entities)
        
        # Should have row co-occurrence edge
        edges = list(graph.edges(data=True))
        cooccurrence_edges = [e for e in edges if 'co_occurrence' in e[2].get('edge_type', '')]
        self.assertGreater(len(cooccurrence_edges), 0)
    
    def test_graph_analysis(self):
        """Test comprehensive graph analysis"""
        entities = [
            PIIEntity("John", PIIType.PERSON, 0, 4, 0.9, "John"),
            PIIEntity("john@example.com", PIIType.EMAIL, 10, 26, 0.9, "email"),
            PIIEntity("555-1234", PIIType.PHONE, 30, 38, 0.8, "phone")
        ]
        
        self.graph_builder.build_graph(entities)
        analysis = self.graph_builder.analyze_graph()
        
        # Check required analysis components
        self.assertIn('basic_metrics', analysis)
        self.assertIn('connected_components', analysis)
        self.assertIn('risk_clusters', analysis)
        self.assertIn('edge_analysis', analysis)
        
        # Verify basic metrics
        self.assertEqual(analysis['basic_metrics']['num_nodes'], 3)
        self.assertIsInstance(analysis['basic_metrics']['density'], float)
    
    def test_risk_cluster_identification(self):
        """Test identification of high-risk entity clusters"""
        entities = [
            PIIEntity("John Doe", PIIType.PERSON, 0, 8, 0.9, "John Doe", 
                     risk_level=RiskLevel.MEDIUM),
            PIIEntity("123-45-6789", PIIType.SSN, 15, 26, 0.95, "SSN", 
                     risk_level=RiskLevel.CRITICAL)
        ]
        
        self.graph_builder.build_graph(entities)
        analysis = self.graph_builder.analyze_graph()
        
        risk_clusters = analysis['risk_clusters']
        if risk_clusters:  # Only test if clusters exist
            high_risk_cluster = max(risk_clusters, key=lambda x: x['overall_risk'])
            self.assertGreater(high_risk_cluster['overall_risk'], 0.5)
    
    def test_empty_graph(self):
        """Test handling of empty graph"""
        analysis = self.graph_builder.analyze_graph()
        self.assertIn('error', analysis)
    
    def test_visualization_creation(self):
        """Test graph visualization creation"""
        entities = [
            PIIEntity("test@example.com", PIIType.EMAIL, 0, 16, 0.9, "email"),
            PIIEntity("555-1234", PIIType.PHONE, 20, 28, 0.8, "phone")
        ]
        
        self.graph_builder.build_graph(entities)
        
        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as tmp_file:
            result_path = self.graph_builder.visualize_graph(tmp_file.name)
            self.assertTrue(Path(result_path).exists())
            Path(result_path).unlink()  # Cleanup

class TestAdvancedPIIAgent(unittest.TestCase):
    """Test the main Advanced PII Agent integration"""
    
    def setUp(self):
        """Setup test environment"""
        # Mock spaCy to avoid model loading
        with patch('advanced_pii_agent.spacy') as mock_spacy:
            mock_nlp = MagicMock()
            mock_spacy.load.return_value = mock_nlp
            self.agent = AdvancedPIIAgent()
            
        # Create temporary directory for outputs
        self.test_dir = Path(tempfile.mkdtemp())
    
    def tearDown(self):
        """Cleanup test environment"""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_csv_processing_workflow(self):
        """Test complete CSV processing workflow"""
        # Create test CSV
        test_data = {
            'name': ['John Doe', 'Jane Smith', 'Bob Johnson'],
            'email': ['john@example.com', 'jane@test.org', 'bob@company.net'],
            'phone': ['555-123-4567', '555-987-6543', '555-555-5555'],
            'notes': ['Regular customer', 'VIP client', 'New prospect']
        }
        
        test_df = pd.DataFrame(test_data)
        test_csv_path = self.test_dir / 'test_data.csv'
        test_df.to_csv(test_csv_path, index=False)
        
        # Process CSV
        results = self.agent.process_csv(str(test_csv_path), str(self.test_dir))
        
        # Verify results structure
        if "error" not in results:
            self.assertIn('summary', results)
            self.assertIn('json_report_path', results)
            self.assertIn('masked_csv_path', results)
            
            # Verify output files exist
            self.assertTrue(Path(results['json_report_path']).exists())
            self.assertTrue(Path(results['masked_csv_path']).exists())
            
            # Verify masked CSV has redacted content
            masked_df = pd.read_csv(results['masked_csv_path'])
            self.assertTrue(any('[EMAIL_REDACTED]' in str(cell) 
                              for cell in masked_df.values.flatten() 
                              if isinstance(cell, str)))
    
    def test_safe_csv_reading(self):
        """Test secure CSV reading with various encodings"""
        # Create test CSV with special characters
        test_data = pd.DataFrame({
            'name': ['José María', 'François', 'München'],
            'email': ['jose@example.com', 'francois@test.fr', 'munchen@test.de']
        })
        
        test_csv_path = self.test_dir / 'test_encoding.csv'
        test_data.to_csv(test_csv_path, index=False, encoding='utf-8')
        
        # Test reading
        df = self.agent._safe_read_csv(str(test_csv_path))
        self.assertEqual(len(df), 3)
        self.assertIn('José María', df['name'].values)
    
    def test_large_file_handling(self):
        """Test handling of large files"""
        # Create a file that exceeds size limit (mock the size check)
        test_csv_path = self.test_dir / 'large_file.csv'
        pd.DataFrame({'col1': ['data']}).to_csv(test_csv_path, index=False)
        
        # Mock file size to exceed limit
        with patch('pathlib.Path.stat') as mock_stat:
            mock_stat.return_value.st_size = 200 * 1024 * 1024  # 200MB
            
            with self.assertRaises(ValueError) as context:
                self.agent._safe_read_csv(str(test_csv_path))
            
            self.assertIn("File too large", str(context.exception))
    
    def test_pii_masking(self):
        """Test PII masking in cell values"""
        entities = [
            PIIEntity("john@example.com", PIIType.EMAIL, 5, 21, 0.9, "context"),
            PIIEntity("555-1234", PIIType.PHONE, 25, 33, 0.8, "context")
        ]
        
        cell_value = "Call john@example.com at 555-1234 for info"
        masked_value = self.agent._mask_cell_pii(cell_value, entities)
        
        self.assertIn('[EMAIL_REDACTED]', masked_value)
        self.assertIn('[PHONE_REDACTED]', masked_value)
        self.assertNotIn('john@example.com', masked_value)
        self.assertNotIn('555-1234', masked_value)
    
    def test_risk_distribution_calculation(self):
        """Test risk distribution calculation"""
        entities = [
            PIIEntity("email1", PIIType.EMAIL, 0, 6, 0.9, "ctx", risk_level=RiskLevel.LOW),
            PIIEntity("email2", PIIType.EMAIL, 10, 16, 0.9, "ctx", risk_level=RiskLevel.MEDIUM),
            PIIEntity("ssn", PIIType.SSN, 20, 23, 0.95, "ctx", risk_level=RiskLevel.CRITICAL),
            PIIEntity("phone", PIIType.PHONE, 30, 35, 0.8, "ctx", risk_level=RiskLevel.HIGH)
        ]
        
        distribution = self.agent._calculate_risk_distribution(entities)
        
        self.assertEqual(distribution['low'], 1)
        self.assertEqual(distribution['medium'], 1)
        self.assertEqual(distribution['high'], 1)
        self.assertEqual(distribution['critical'], 1)
    
    def test_error_handling(self):
        """Test error handling in CSV processing"""
        # Test with non-existent file
        results = self.agent.process_csv('nonexistent_file.csv')
        self.assertIn('error', results)
        
        # Test with invalid CSV format
        invalid_file = self.test_dir / 'invalid.csv'
        invalid_file.write_text('invalid,csv,content\nwith"broken"quotes')
        
        # This should handle the error gracefully
        results = self.agent.process_csv(str(invalid_file), str(self.test_dir))
        # Should either succeed with warnings or fail gracefully

class TestIntegrationScenarios(unittest.TestCase):
    """Test integration scenarios and edge cases"""
    
    def setUp(self):
        """Setup integration test environment"""
        with patch('advanced_pii_agent.spacy') as mock_spacy:
            mock_nlp = MagicMock()
            mock_spacy.load.return_value = mock_nlp
            self.agent = AdvancedPIIAgent()
        
        self.test_dir = Path(tempfile.mkdtemp())
    
    def tearDown(self):
        """Cleanup"""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_healthcare_data_scenario(self):
        """Test scenario with healthcare-related PII"""
        healthcare_data = {
            'patient_id': ['P001', 'P002', 'P003'],
            'patient_name': ['John Doe', 'Jane Smith', 'Bob Wilson'],
            'ssn': ['123-45-6789', '987-65-4321', '456-78-9123'],
            'dob': ['1980-01-15', '1975-06-22', '1992-03-08'],
            'email': ['john@email.com', 'jane@email.com', 'bob@email.com'],
            'diagnosis': ['Diabetes', 'Hypertension', 'Asthma']
        }
        
        df = pd.DataFrame(healthcare_data)
        csv_path = self.test_dir / 'healthcare.csv'
        df.to_csv(csv_path, index=False)
        
        results = self.agent.process_csv(str(csv_path), str(self.test_dir))
        
        if "error" not in results:
            summary = results['summary']
            
            # Should detect multiple high-risk PII types
            expected_types = {'social_security_number', 'email_address', 'person_name'}
            actual_types = set(summary['pii_types_detected'])
            
            # At least some overlap expected
            self.assertTrue(len(expected_types.intersection(actual_types)) > 0)
            
            # Should have high-risk entities due to SSN
            self.assertGreater(len(results.get('high_risk_entities', [])), 0)
    
    def test_financial_data_scenario(self):
        """Test scenario with financial PII"""
        financial_data = {
            'customer_name': ['Alice Johnson', 'Bob Smith'],
            'account_number': ['1234567890123456', '9876543210987654'],
            'credit_card': ['4111-1111-1111-1111', '5555-4444-3333-2222'],
            'phone': ['555-123-4567', '555-987-6543'],
            'address': ['123 Main St', '456 Oak Ave']
        }
        
        df = pd.DataFrame(financial_data)
        csv_path = self.test_dir / 'financial.csv'
        df.to_csv(csv_path, index=False)
        
        results = self.agent.process_csv(str(csv_path), str(self.test_dir))
        
        if "error" not in results:
            # Should detect credit card numbers as critical PII
            high_risk = results.get('high_risk_entities', [])
            cc_entities = [e for e in high_risk if e.get('pii_type') == 'credit_card']
            self.assertGreater(len(cc_entities), 0)
    
    def test_mixed_content_scenario(self):
        """Test with mixed content including safe and PII data"""
        mixed_data = {
            'product_id': ['PROD001', 'PROD002', 'PROD003'],
            'description': ['Widget A', 'Gadget B', 'Tool C'],
            'contact_info': [
                'For questions, email support@company.com',
                'Call John at 555-123-4567 for details',
                'Visit our website at https://example.com'
            ],
            'metadata': ['Category: Electronics', 'Stock: 50', 'Price: $29.99']
        }
        
        df = pd.DataFrame(mixed_data)
        csv_path = self.test_dir / 'mixed.csv'
        df.to_csv(csv_path, index=False)
        
        results = self.agent.process_csv(str(csv_path), str(self.test_dir))
        
        if "error" not in results:
            # Should detect some PII in contact_info but not in other columns
            summary = results['summary']
            self.assertGreater(summary['total_entities'], 0)
            
            # Masked CSV should preserve non-PII content
            masked_df = pd.read_csv(results['masked_csv_path'])
            self.assertEqual(len(masked_df), len(df))
            
            # Product IDs should be unchanged
            self.assertEqual(masked_df['product_id'].iloc[0], 'PROD001')

class TestPerformanceAndSecurity(unittest.TestCase):
    """Test performance and security aspects"""
    
    def setUp(self):
        """Setup performance test environment"""
        with patch('advanced_pii_agent.spacy') as mock_spacy:
            mock_nlp = MagicMock()
            mock_spacy.load.return_value = mock_nlp
            self.detector = PIINERDetector()
    
    def test_detection_performance(self):
        """Test detection performance on larger text"""
        import time
        
        # Create large text with scattered PII
        large_text = "This is a large text document. " * 1000
        large_text += "Contact john@example.com for more information. "
        large_text += "Call 555-123-4567 for support. " * 100
        
        start_time = time.perf_counter()
        entities = self.detector.detect_pii_in_text(large_text)
        end_time = time.perf_counter()
        
        processing_time = end_time - start_time
        
        # Should complete within reasonable time (adjust threshold as needed)
        self.assertLess(processing_time, 5.0, 
                       f"Processing took too long: {processing_time:.2f}s")
        
        # Should still detect PII correctly
        self.assertGreater(len(entities), 0)
    
    def test_memory_usage(self):
        """Test memory usage with large datasets"""
        # Create moderately large dataset
        large_data = {
            'col1': [f'user{i}@example.com' for i in range(1000)],
            'col2': [f'555-{i:03d}-{i:04d}' for i in range(1000)],
            'col3': [f'User Name {i}' for i in range(1000)]
        }
        
        df = pd.DataFrame(large_data)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
            df.to_csv(tmp_file.name, index=False)
            
            try:
                # This should complete without memory errors
                with patch('advanced_pii_agent.spacy') as mock_spacy:
                    mock_nlp = MagicMock()
                    mock_spacy.load.return_value = mock_nlp
                    agent = AdvancedPIIAgent()
                
                # Process without running out of memory
                results = agent.process_csv(tmp_file.name)
                
                # Basic verification that it completed
                self.assertTrue(isinstance(results, dict))
                
            finally:
                Path(tmp_file.name).unlink()
    
    def test_input_sanitization(self):
        """Test input sanitization and injection prevention"""
        # Test potentially malicious inputs
        malicious_inputs = [
            "'; DROP TABLE users; --",
            "<script>alert('xss')</script>",
            "../../../../etc/passwd",
            "null\x00byte",
            "very" * 10000 + "long" * 10000  # Extremely long input
        ]
        
        for malicious_input in malicious_inputs:
            with self.subTest(input=malicious_input[:50]):
                try:
                    entities = self.detector.detect_pii_in_text(malicious_input)
                    # Should handle malicious input without crashing
                    self.assertIsInstance(entities, list)
                except Exception as e:
                    # If it fails, it should fail gracefully
                    self.assertIsInstance(e, (ValueError, TypeError))

def run_test_suite():
    """Run the complete test suite"""
    test_classes = [
        TestPIIEntity,
        TestPIINERDetector,
        TestProximityAnalyzer,
        TestPIIGraphBuilder,
        TestAdvancedPIIAgent,
        TestIntegrationScenarios,
        TestPerformanceAndSecurity
    ]
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result

if __name__ == "__main__":
    # Run specific test class if provided as argument
    import sys
    
    if len(sys.argv) > 1:
        test_class_name = sys.argv[1]
        if hasattr(sys.modules[__name__], test_class_name):
            suite = unittest.TestLoader().loadTestsFromTestCase(
                getattr(sys.modules[__name__], test_class_name)
            )
            runner = unittest.TextTestRunner(verbosity=2)
            runner.run(suite)
        else:
            print(f"Test class '{test_class_name}' not found")
    else:
        # Run all tests
        run_test_suite()