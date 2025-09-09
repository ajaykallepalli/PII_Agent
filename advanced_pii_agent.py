#!/usr/bin/env python3
"""
Advanced PII Detection Agent
Combines NER, Proximity Analysis, and Graph Theory for comprehensive PII detection
"""

import json
import logging
import re
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PIIType(Enum):
    """Enhanced PII classification with severity levels"""
    # Critical PII - high re-identification risk
    SSN = "social_security_number"
    CREDIT_CARD = "credit_card"
    BANK_ACCOUNT = "bank_account"
    DRIVERS_LICENSE = "drivers_license"
    PASSPORT = "passport"
    
    # Personal Identifiers
    PERSON = "person_name"
    EMAIL = "email_address"
    PHONE = "phone_number"
    DATE_OF_BIRTH = "date_of_birth"
    
    # Location Data
    ADDRESS = "physical_address"
    ZIP_CODE = "zip_code"
    LOCATION = "location"
    GPS_COORD = "gps_coordinates"
    
    # Organizational
    ORGANIZATION = "organization"
    WEBSITE = "website_url"
    
    # Technical
    IP_ADDRESS = "ip_address"
    MAC_ADDRESS = "mac_address"
    API_KEY = "api_key"
    PASSWORD = "password"
    
    # Financial
    IBAN = "iban"
    BITCOIN_ADDRESS = "bitcoin_address"
    
    # Medical
    MEDICAL_ID = "medical_identifier"
    
    # Other
    USERNAME = "username"
    OTHER = "other_pii"

class RiskLevel(Enum):
    """Risk assessment levels for PII exposure"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class PIIEntity:
    """Structured representation of a detected PII entity"""
    text: str
    pii_type: PIIType
    start_pos: int
    end_pos: int
    confidence: float
    context: str
    row_index: Optional[int] = None
    column_name: Optional[str] = None
    risk_level: RiskLevel = RiskLevel.LOW
    detection_method: str = "regex"  # "regex", "ner", "proximity", "graph"
    related_entities: List[str] = None
    
    def __post_init__(self):
        if self.related_entities is None:
            self.related_entities = []
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        result = asdict(self)
        result['pii_type'] = self.pii_type.value
        result['risk_level'] = self.risk_level.value
        return result

class PIINERDetector:
    """
    Named Entity Recognition-based PII detector using spaCy
    Handles person names, organizations, locations, and custom patterns
    """
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        """
        Initialize NER detector with spaCy model
        
        Args:
            model_name: spaCy model to use for NER
        """
        self.model_name = model_name
        self.nlp = None
        self._load_model()
        self._setup_patterns()
        
    def _load_model(self):
        """Load spaCy model with error handling"""
        try:
            import spacy
            self.nlp = spacy.load(self.model_name)
            logger.info(f"Loaded spaCy model: {self.model_name}")
        except ImportError:
            logger.error("spaCy not installed. Run: pip install spacy")
            raise ImportError("spaCy is required for NER functionality")
        except OSError:
            logger.error(f"spaCy model '{self.model_name}' not found. Run: python -m spacy download {self.model_name}")
            raise OSError(f"spaCy model '{self.model_name}' not available")
    
    def _setup_patterns(self):
        """Setup regex patterns for non-NER PII detection"""
        self.patterns = {
            PIIType.EMAIL: re.compile(
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                re.IGNORECASE
            ),
            PIIType.PHONE: re.compile(
                r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'
            ),
            PIIType.SSN: re.compile(
                r'\b\d{3}-\d{2}-\d{4}\b|\b\d{9}\b'
            ),
            PIIType.CREDIT_CARD: re.compile(
                r'\b(?:\d{4}[-\s]?){3}\d{4}\b'
            ),
            PIIType.ZIP_CODE: re.compile(
                r'\b\d{5}(?:-\d{4})?\b'
            ),
            PIIType.IP_ADDRESS: re.compile(
                r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
            ),
            PIIType.DATE_OF_BIRTH: re.compile(
                r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2})\b'
            ),
            PIIType.API_KEY: re.compile(
                r'\b(?:api[_-]?key|apikey|access[_-]?token)[\s:=]+[\w-]{20,}\b',
                re.IGNORECASE
            ),
            PIIType.PASSWORD: re.compile(
                r'\b(?:password|passwd|pwd)[\s:=]+\S+\b',
                re.IGNORECASE
            ),
        }
    
    def detect_pii_in_text(self, text: str, context: Dict = None) -> List[PIIEntity]:
        """
        Detect PII entities in text using both NER and regex patterns
        
        Args:
            text: Input text to analyze
            context: Additional context (row_index, column_name, etc.)
            
        Returns:
            List of detected PII entities
        """
        entities = []
        context = context or {}
        
        if not text or not isinstance(text, str):
            return entities
        
        # Regex-based detection
        entities.extend(self._detect_regex_pii(text, context))
        
        # NER-based detection
        if self.nlp:
            entities.extend(self._detect_ner_pii(text, context))
        
        # Remove duplicates and overlaps
        entities = self._deduplicate_entities(entities)
        
        return entities
    
    def _detect_regex_pii(self, text: str, context: Dict) -> List[PIIEntity]:
        """Detect PII using regex patterns"""
        entities = []
        
        for pii_type, pattern in self.patterns.items():
            matches = pattern.finditer(text)
            for match in matches:
                entity = PIIEntity(
                    text=match.group(),
                    pii_type=pii_type,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=0.9,  # High confidence for regex matches
                    context=self._extract_context(text, match.start(), match.end()),
                    row_index=context.get('row_index'),
                    column_name=context.get('column_name'),
                    detection_method="regex"
                )
                entities.append(entity)
        
        return entities
    
    def _detect_ner_pii(self, text: str, context: Dict) -> List[PIIEntity]:
        """Detect PII using spaCy NER"""
        entities = []
        
        try:
            doc = self.nlp(text)
            
            for ent in doc.ents:
                pii_type = self._map_ner_label_to_pii_type(ent.label_)
                if pii_type:
                    entity = PIIEntity(
                        text=ent.text,
                        pii_type=pii_type,
                        start_pos=ent.start_char,
                        end_pos=ent.end_char,
                        confidence=0.8,  # NER confidence can vary
                        context=self._extract_context(text, ent.start_char, ent.end_char),
                        row_index=context.get('row_index'),
                        column_name=context.get('column_name'),
                        detection_method="ner"
                    )
                    entities.append(entity)
        
        except Exception as e:
            logger.warning(f"NER processing error: {e}")
        
        return entities
    
    def _map_ner_label_to_pii_type(self, label: str) -> Optional[PIIType]:
        """Map spaCy NER labels to PII types"""
        label_mapping = {
            "PERSON": PIIType.PERSON,
            "ORG": PIIType.ORGANIZATION,
            "GPE": PIIType.LOCATION,  # Geopolitical entity
            "LOC": PIIType.LOCATION,
            "FAC": PIIType.LOCATION,  # Facility
            "DATE": None,  # Generally not PII unless specific format
            "TIME": None,
            "MONEY": None,
            "PERCENT": None,
            "ORDINAL": None,
            "CARDINAL": None,
        }
        return label_mapping.get(label)
    
    def _extract_context(self, text: str, start: int, end: int, window: int = 50) -> str:
        """Extract context around detected entity"""
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)
        return text[context_start:context_end]
    
    def _deduplicate_entities(self, entities: List[PIIEntity]) -> List[PIIEntity]:
        """Remove overlapping and duplicate entities, keeping highest confidence"""
        if not entities:
            return entities
        
        # Sort by start position
        entities.sort(key=lambda x: x.start_pos)
        
        deduplicated = []
        for entity in entities:
            # Check for overlap with existing entities
            overlaps = False
            for existing in deduplicated:
                if (entity.start_pos < existing.end_pos and 
                    entity.end_pos > existing.start_pos):
                    # Overlapping entities - keep the one with higher confidence
                    if entity.confidence > existing.confidence:
                        deduplicated.remove(existing)
                        deduplicated.append(entity)
                    overlaps = True
                    break
            
            if not overlaps:
                deduplicated.append(entity)
        
        return sorted(deduplicated, key=lambda x: x.start_pos)

class ProximityAnalyzer:
    """
    Analyzes proximity relationships between entities to identify contextual PII risks
    Uses sliding window approach to find related entities that increase re-identification risk
    """
    
    def __init__(self, window_size: int = 100, risk_threshold: float = 0.7):
        """
        Initialize proximity analyzer
        
        Args:
            window_size: Character window size for proximity analysis
            risk_threshold: Threshold for high-risk proximity relationships
        """
        self.window_size = window_size
        self.risk_threshold = risk_threshold
        
        # Define risk relationships between PII types
        self.risk_matrix = self._build_risk_matrix()
    
    def _build_risk_matrix(self) -> Dict[Tuple[PIIType, PIIType], float]:
        """Build matrix of risk scores for PII type combinations"""
        risk_matrix = {}
        
        # High-risk combinations (score >= 0.8)
        high_risk_pairs = [
            (PIIType.PERSON, PIIType.SSN),
            (PIIType.PERSON, PIIType.DATE_OF_BIRTH),
            (PIIType.PERSON, PIIType.ADDRESS),
            (PIIType.EMAIL, PIIType.PHONE),
            (PIIType.PERSON, PIIType.PHONE),
            (PIIType.ORGANIZATION, PIIType.EMAIL),
            (PIIType.ZIP_CODE, PIIType.ADDRESS),
            (PIIType.PERSON, PIIType.CREDIT_CARD),
        ]
        
        # Medium-risk combinations (score >= 0.5)
        medium_risk_pairs = [
            (PIIType.PERSON, PIIType.ORGANIZATION),
            (PIIType.EMAIL, PIIType.ORGANIZATION),
            (PIIType.PHONE, PIIType.ADDRESS),
            (PIIType.LOCATION, PIIType.ZIP_CODE),
            (PIIType.PERSON, PIIType.EMAIL),
            (PIIType.PERSON, PIIType.LOCATION),
        ]
        
        # Assign risk scores
        for pair in high_risk_pairs:
            risk_matrix[pair] = 0.9
            risk_matrix[(pair[1], pair[0])] = 0.9  # Symmetric
        
        for pair in medium_risk_pairs:
            risk_matrix[pair] = 0.6
            risk_matrix[(pair[1], pair[0])] = 0.6  # Symmetric
        
        return risk_matrix
    
    def analyze_proximity(self, entities: List[PIIEntity], text: str) -> List[PIIEntity]:
        """
        Analyze proximity relationships and update entity risk levels
        
        Args:
            entities: List of detected PII entities
            text: Original text for context analysis
            
        Returns:
            Updated entities with proximity-based risk assessments
        """
        if len(entities) < 2:
            return entities
        
        # Create proximity groups
        proximity_groups = self._create_proximity_groups(entities)
        
        # Analyze each group for risk relationships
        for group in proximity_groups:
            self._analyze_group_risk(group, text)
        
        # Update entity risk levels based on proximity analysis
        updated_entities = []
        for entity in entities:
            updated_entity = self._update_entity_risk(entity, proximity_groups)
            updated_entities.append(updated_entity)
        
        return updated_entities
    
    def _create_proximity_groups(self, entities: List[PIIEntity]) -> List[List[PIIEntity]]:
        """Group entities that are within proximity window of each other"""
        groups = []
        used_entities = set()
        
        for i, entity in enumerate(entities):
            if i in used_entities:
                continue
            
            group = [entity]
            used_entities.add(i)
            
            # Find all entities within proximity window
            for j, other_entity in enumerate(entities[i+1:], i+1):
                if j in used_entities:
                    continue
                
                if self._are_proximate(entity, other_entity):
                    group.append(other_entity)
                    used_entities.add(j)
            
            if len(group) > 1:  # Only include groups with multiple entities
                groups.append(group)
        
        return groups
    
    def _are_proximate(self, entity1: PIIEntity, entity2: PIIEntity) -> bool:
        """Check if two entities are within proximity window"""
        return abs(entity1.start_pos - entity2.start_pos) <= self.window_size
    
    def _analyze_group_risk(self, group: List[PIIEntity], text: str):
        """Analyze risk level for a group of proximate entities"""
        group_types = [entity.pii_type for entity in group]
        
        # Calculate maximum risk score for the group
        max_risk = 0.0
        risk_pairs = []
        
        for i, entity1 in enumerate(group):
            for entity2 in group[i+1:]:
                pair_key = (entity1.pii_type, entity2.pii_type)
                risk_score = self.risk_matrix.get(pair_key, 0.3)  # Default low risk
                
                if risk_score > max_risk:
                    max_risk = risk_score
                
                if risk_score >= 0.5:  # Medium or higher risk
                    risk_pairs.append((entity1, entity2, risk_score))
        
        # Update related entities information
        for entity in group:
            entity.related_entities = [
                f"{other.pii_type.value}:{other.text[:20]}..." 
                for other in group if other != entity
            ]
    
    def _update_entity_risk(self, entity: PIIEntity, proximity_groups: List[List[PIIEntity]]) -> PIIEntity:
        """Update entity risk level based on proximity analysis"""
        # Find which group this entity belongs to
        entity_group = None
        for group in proximity_groups:
            if entity in group:
                entity_group = group
                break
        
        if not entity_group or len(entity_group) == 1:
            # No proximity relationships
            entity.risk_level = self._get_base_risk_level(entity.pii_type)
            return entity
        
        # Calculate risk based on proximity relationships
        max_proximity_risk = 0.0
        for other_entity in entity_group:
            if other_entity != entity:
                pair_key = (entity.pii_type, other_entity.pii_type)
                risk = self.risk_matrix.get(pair_key, 0.3)
                max_proximity_risk = max(max_proximity_risk, risk)
        
        # Combine base risk with proximity risk
        base_risk = self._get_base_risk_score(entity.pii_type)
        combined_risk = min(1.0, base_risk + max_proximity_risk * 0.5)
        
        entity.risk_level = self._score_to_risk_level(combined_risk)
        entity.detection_method = "proximity"
        
        return entity
    
    def _get_base_risk_level(self, pii_type: PIIType) -> RiskLevel:
        """Get base risk level for a PII type"""
        critical_types = {PIIType.SSN, PIIType.CREDIT_CARD, PIIType.PASSPORT, 
                         PIIType.DRIVERS_LICENSE, PIIType.BANK_ACCOUNT}
        high_types = {PIIType.DATE_OF_BIRTH, PIIType.ADDRESS, PIIType.MEDICAL_ID}
        medium_types = {PIIType.PERSON, PIIType.EMAIL, PIIType.PHONE}
        
        if pii_type in critical_types:
            return RiskLevel.CRITICAL
        elif pii_type in high_types:
            return RiskLevel.HIGH
        elif pii_type in medium_types:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def _get_base_risk_score(self, pii_type: PIIType) -> float:
        """Get base risk score for a PII type"""
        risk_level = self._get_base_risk_level(pii_type)
        score_mapping = {
            RiskLevel.CRITICAL: 0.9,
            RiskLevel.HIGH: 0.7,
            RiskLevel.MEDIUM: 0.5,
            RiskLevel.LOW: 0.3
        }
        return score_mapping[risk_level]
    
    def _score_to_risk_level(self, score: float) -> RiskLevel:
        """Convert risk score to risk level"""
        if score >= 0.85:
            return RiskLevel.CRITICAL
        elif score >= 0.65:
            return RiskLevel.HIGH
        elif score >= 0.45:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW

class PIIGraphBuilder:
    """
    Builds and analyzes entity graphs using networkx to identify PII clusters
    and relationships that may increase re-identification risks
    """
    
    def __init__(self, min_edge_weight: float = 0.1):
        """
        Initialize graph builder
        
        Args:
            min_edge_weight: Minimum weight for edges to be included in graph
        """
        self.min_edge_weight = min_edge_weight
        self.graph = nx.Graph()
        self.entity_metadata = {}
    
    def build_graph(self, entities: List[PIIEntity], text: str = None) -> nx.Graph:
        """
        Build entity relationship graph
        
        Args:
            entities: List of PII entities
            text: Original text for context (optional)
            
        Returns:
            NetworkX graph with entities as nodes and relationships as edges
        """
        self.graph.clear()
        self.entity_metadata.clear()
        
        if not entities:
            return self.graph
        
        # Add nodes
        for i, entity in enumerate(entities):
            node_id = f"{entity.pii_type.value}_{i}"
            
            self.graph.add_node(
                node_id,
                pii_type=entity.pii_type.value,
                text=entity.text[:50],  # Truncate for privacy
                confidence=entity.confidence,
                risk_level=entity.risk_level.value,
                row_index=entity.row_index,
                column_name=entity.column_name,
                detection_method=entity.detection_method
            )
            
            self.entity_metadata[node_id] = entity
        
        # Add edges based on relationships
        self._add_proximity_edges(entities)
        self._add_semantic_edges(entities)
        self._add_co_occurrence_edges(entities)
        
        return self.graph
    
    def _add_proximity_edges(self, entities: List[PIIEntity]):
        """Add edges between entities that are spatially close"""
        proximity_analyzer = ProximityAnalyzer()
        
        for i, entity1 in enumerate(entities):
            node1_id = f"{entity1.pii_type.value}_{i}"
            
            for j, entity2 in enumerate(entities[i+1:], i+1):
                node2_id = f"{entity2.pii_type.value}_{j}"
                
                # Calculate proximity weight
                distance = abs(entity1.start_pos - entity2.start_pos)
                if distance <= proximity_analyzer.window_size:
                    # Inverse distance weighting
                    weight = max(0.1, 1.0 - (distance / proximity_analyzer.window_size))
                    
                    if weight >= self.min_edge_weight:
                        self.graph.add_edge(
                            node1_id, node2_id,
                            weight=weight,
                            edge_type="proximity",
                            distance=distance
                        )
    
    def _add_semantic_edges(self, entities: List[PIIEntity]):
        """Add edges between semantically related entities"""
        semantic_relationships = {
            (PIIType.PERSON, PIIType.EMAIL): 0.8,
            (PIIType.PERSON, PIIType.PHONE): 0.8,
            (PIIType.PERSON, PIIType.ADDRESS): 0.9,
            (PIIType.ORGANIZATION, PIIType.EMAIL): 0.7,
            (PIIType.ORGANIZATION, PIIType.ADDRESS): 0.8,
            (PIIType.ADDRESS, PIIType.ZIP_CODE): 0.9,
            (PIIType.EMAIL, PIIType.USERNAME): 0.6,
        }
        
        for i, entity1 in enumerate(entities):
            node1_id = f"{entity1.pii_type.value}_{i}"
            
            for j, entity2 in enumerate(entities[i+1:], i+1):
                node2_id = f"{entity2.pii_type.value}_{j}"
                
                # Check for semantic relationship
                pair_key = (entity1.pii_type, entity2.pii_type)
                reverse_key = (entity2.pii_type, entity1.pii_type)
                
                weight = semantic_relationships.get(pair_key) or semantic_relationships.get(reverse_key)
                
                if weight and weight >= self.min_edge_weight:
                    self.graph.add_edge(
                        node1_id, node2_id,
                        weight=weight,
                        edge_type="semantic"
                    )
    
    def _add_co_occurrence_edges(self, entities: List[PIIEntity]):
        """Add edges between entities that co-occur in the same row/column"""
        # Group entities by row and column
        row_groups = defaultdict(list)
        col_groups = defaultdict(list)
        
        for i, entity in enumerate(entities):
            node_id = f"{entity.pii_type.value}_{i}"
            
            if entity.row_index is not None:
                row_groups[entity.row_index].append(node_id)
            
            if entity.column_name is not None:
                col_groups[entity.column_name].append(node_id)
        
        # Add edges within row groups
        for row_entities in row_groups.values():
            if len(row_entities) > 1:
                for i, node1 in enumerate(row_entities):
                    for node2 in row_entities[i+1:]:
                        if not self.graph.has_edge(node1, node2):
                            self.graph.add_edge(
                                node1, node2,
                                weight=0.3,
                                edge_type="row_co_occurrence"
                            )
        
        # Add edges within column groups (lighter weight)
        for col_entities in col_groups.values():
            if len(col_entities) > 1:
                for i, node1 in enumerate(col_entities):
                    for node2 in col_entities[i+1:]:
                        if not self.graph.has_edge(node1, node2):
                            self.graph.add_edge(
                                node1, node2,
                                weight=0.2,
                                edge_type="column_co_occurrence"
                            )
    
    def analyze_graph(self) -> Dict[str, Any]:
        """
        Perform comprehensive graph analysis
        
        Returns:
            Dictionary with graph analysis results
        """
        if not self.graph.nodes():
            return {"error": "Empty graph - no entities to analyze"}
        
        analysis = {}
        
        # Basic graph metrics
        analysis["basic_metrics"] = {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "density": nx.density(self.graph),
            "is_connected": nx.is_connected(self.graph)
        }
        
        # Connected components analysis
        components = list(nx.connected_components(self.graph))
        analysis["connected_components"] = {
            "count": len(components),
            "sizes": [len(comp) for comp in components],
            "largest_component_size": max([len(comp) for comp in components]) if components else 0
        }
        
        # Centrality measures
        if self.graph.number_of_nodes() > 1:
            analysis["centrality"] = {
                "degree_centrality": dict(nx.degree_centrality(self.graph)),
                "betweenness_centrality": dict(nx.betweenness_centrality(self.graph)),
                "closeness_centrality": dict(nx.closeness_centrality(self.graph)),
                "eigenvector_centrality": dict(nx.eigenvector_centrality(self.graph, max_iter=1000))
            }
        
        # Risk cluster identification
        analysis["risk_clusters"] = self._identify_risk_clusters(components)
        
        # Edge analysis
        analysis["edge_analysis"] = self._analyze_edges()
        
        return analysis
    
    def _identify_risk_clusters(self, components: List[Set]) -> List[Dict]:
        """Identify high-risk clusters of connected entities"""
        risk_clusters = []
        
        for i, component in enumerate(components):
            if len(component) < 2:
                continue
            
            # Calculate cluster risk score
            cluster_entities = [self.entity_metadata[node_id] for node_id in component]
            risk_scores = [self._get_risk_score(entity.risk_level) for entity in cluster_entities]
            avg_risk = np.mean(risk_scores)
            max_risk = max(risk_scores)
            
            # Analyze PII type diversity
            pii_types = {entity.pii_type for entity in cluster_entities}
            type_diversity = len(pii_types) / len(cluster_entities)
            
            cluster_info = {
                "cluster_id": i,
                "size": len(component),
                "entities": [
                    {
                        "type": entity.pii_type.value,
                        "text_preview": entity.text[:20] + "..." if len(entity.text) > 20 else entity.text,
                        "risk_level": entity.risk_level.value,
                        "confidence": entity.confidence
                    }
                    for entity in cluster_entities
                ],
                "average_risk_score": avg_risk,
                "max_risk_score": max_risk,
                "type_diversity": type_diversity,
                "overall_risk": self._calculate_cluster_risk(avg_risk, max_risk, type_diversity, len(component))
            }
            
            risk_clusters.append(cluster_info)
        
        # Sort by overall risk
        risk_clusters.sort(key=lambda x: x["overall_risk"], reverse=True)
        
        return risk_clusters
    
    def _get_risk_score(self, risk_level: RiskLevel) -> float:
        """Convert risk level to numeric score"""
        score_mapping = {
            RiskLevel.LOW: 0.25,
            RiskLevel.MEDIUM: 0.5,
            RiskLevel.HIGH: 0.75,
            RiskLevel.CRITICAL: 1.0
        }
        return score_mapping[risk_level]
    
    def _calculate_cluster_risk(self, avg_risk: float, max_risk: float, 
                               type_diversity: float, cluster_size: int) -> float:
        """Calculate overall cluster risk score"""
        # Weighted combination of factors
        size_factor = min(1.0, cluster_size / 5.0)  # Larger clusters are riskier
        diversity_factor = type_diversity  # More diverse types increase risk
        
        overall_risk = (
            0.4 * avg_risk +
            0.3 * max_risk +
            0.2 * diversity_factor +
            0.1 * size_factor
        )
        
        return min(1.0, overall_risk)
    
    def _analyze_edges(self) -> Dict[str, Any]:
        """Analyze edge patterns and types"""
        edge_data = []
        
        for u, v, data in self.graph.edges(data=True):
            edge_data.append({
                "source": u,
                "target": v,
                "weight": data.get("weight", 0),
                "type": data.get("edge_type", "unknown")
            })
        
        if not edge_data:
            return {"total_edges": 0}
        
        # Edge type distribution
        edge_types = [edge["type"] for edge in edge_data]
        type_counts = {edge_type: edge_types.count(edge_type) for edge_type in set(edge_types)}
        
        # Weight distribution
        weights = [edge["weight"] for edge in edge_data]
        
        return {
            "total_edges": len(edge_data),
            "edge_type_distribution": type_counts,
            "weight_statistics": {
                "mean": np.mean(weights),
                "std": np.std(weights),
                "min": np.min(weights),
                "max": np.max(weights)
            }
        }
    
    def visualize_graph(self, output_path: str = "pii_graph.html", 
                       show_labels: bool = True) -> str:
        """
        Create interactive visualization of the PII graph
        
        Args:
            output_path: Path to save the visualization
            show_labels: Whether to show entity labels
            
        Returns:
            Path to the generated visualization file
        """
        if not self.graph.nodes():
            logger.warning("Cannot visualize empty graph")
            return ""
        
        # Create layout
        pos = nx.spring_layout(self.graph, k=3, iterations=50)
        
        # Extract node information
        node_trace = self._create_node_trace(pos, show_labels)
        edge_trace = self._create_edge_trace(pos)
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title="PII Entity Relationship Graph",
                           titlefont_size=16,
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           annotations=[ dict(
                               text="Nodes represent PII entities, edges show relationships",
                               showarrow=False,
                               xref="paper", yref="paper",
                               x=0.005, y=-0.002,
                               xanchor='left', yanchor='bottom',
                               font=dict(size=12)
                           )],
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                       )
        
        # Save visualization
        fig.write_html(output_path)
        logger.info(f"Graph visualization saved to: {output_path}")
        
        return output_path
    
    def _create_node_trace(self, pos: Dict, show_labels: bool) -> go.Scatter:
        """Create node trace for plotly visualization"""
        node_x = []
        node_y = []
        node_text = []
        node_info = []
        node_color = []
        
        # Color mapping for PII types
        type_colors = {
            'person_name': '#ff4444',
            'email_address': '#44ff44',
            'phone_number': '#4444ff',
            'social_security_number': '#ff0000',
            'credit_card': '#ff0000',
            'address': '#ffaa44',
            'organization': '#aa44ff',
            'other': '#888888'
        }
        
        for node in self.graph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            node_data = self.graph.nodes[node]
            pii_type = node_data.get('pii_type', 'other')
            text_preview = node_data.get('text', '')[:20]
            
            node_color.append(type_colors.get(pii_type, '#888888'))
            
            if show_labels:
                node_text.append(f"{pii_type}<br>{text_preview}")
            else:
                node_text.append('')
            
            # Hover info
            info = f"Type: {pii_type}<br>"
            info += f"Text: {text_preview}<br>"
            info += f"Risk: {node_data.get('risk_level', 'unknown')}<br>"
            info += f"Confidence: {node_data.get('confidence', 0):.2f}<br>"
            info += f"Method: {node_data.get('detection_method', 'unknown')}"
            
            node_info.append(info)
        
        return go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            hovertext=node_info,
            textposition="middle center",
            marker=dict(
                size=20,
                color=node_color,
                line=dict(width=2, color='white')
            )
        )
    
    def _create_edge_trace(self, pos: Dict) -> go.Scatter:
        """Create edge trace for plotly visualization"""
        edge_x = []
        edge_y = []
        
        for edge in self.graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        return go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='#888'),
            hoverinfo='none',
            mode='lines'
        )

class AdvancedPIIAgent:
    """
    Main agent class that orchestrates NER, Proximity Analysis, and Graph Theory
    for comprehensive PII detection and risk assessment
    """
    
    def __init__(self, spacy_model: str = "en_core_web_sm"):
        """
        Initialize the advanced PII agent
        
        Args:
            spacy_model: spaCy model name for NER
        """
        self.ner_detector = PIINERDetector(spacy_model)
        self.proximity_analyzer = ProximityAnalyzer()
        self.graph_builder = PIIGraphBuilder()
        
        logger.info("Advanced PII Agent initialized successfully")
    
    def process_csv(self, input_path: str, output_dir: str = "./output") -> Dict[str, Any]:
        """
        Process CSV file for comprehensive PII detection and analysis
        
        Args:
            input_path: Path to input CSV file
            output_dir: Directory for output files
            
        Returns:
            Dictionary containing analysis results and file paths
        """
        try:
            # Validate input
            input_file = Path(input_path)
            if not input_file.exists():
                raise FileNotFoundError(f"Input file not found: {input_path}")
            
            # Create output directory
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Processing CSV file: {input_path}")
            
            # Read CSV with security measures
            df = self._safe_read_csv(input_path)
            
            # Process data
            all_entities = []
            masked_df = df.copy()
            
            # Process each cell
            for row_idx, row in df.iterrows():
                for col_name, cell_value in row.items():
                    if pd.isna(cell_value) or cell_value == '':
                        continue
                    
                    cell_str = str(cell_value)
                    context = {
                        'row_index': row_idx,
                        'column_name': col_name
                    }
                    
                    # Detect PII in cell
                    entities = self.ner_detector.detect_pii_in_text(cell_str, context)
                    
                    if entities:
                        all_entities.extend(entities)
                        # Mask PII in the cell
                        masked_value = self._mask_cell_pii(cell_str, entities)
                        masked_df.at[row_idx, col_name] = masked_value
            
            logger.info(f"Detected {len(all_entities)} PII entities")
            
            # Proximity analysis
            logger.info("Performing proximity analysis...")
            all_entities = self.proximity_analyzer.analyze_proximity(all_entities, "")
            
            # Graph analysis
            logger.info("Building entity relationship graph...")
            graph = self.graph_builder.build_graph(all_entities)
            graph_analysis = self.graph_builder.analyze_graph()
            
            # Generate outputs
            results = self._generate_output_files(
                all_entities, masked_df, graph_analysis, 
                output_path, input_file.stem
            )
            
            # Create visualization
            viz_path = output_path / f"{input_file.stem}_graph.html"
            self.graph_builder.visualize_graph(str(viz_path))
            results["visualization_path"] = str(viz_path)
            
            logger.info("Processing completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Error processing CSV: {e}")
            return {"error": str(e)}
    
    def _safe_read_csv(self, file_path: str, max_size_mb: int = 100) -> pd.DataFrame:
        """Safely read CSV file with size limits and error handling"""
        file_size = Path(file_path).stat().st_size / (1024 * 1024)  # MB
        
        if file_size > max_size_mb:
            raise ValueError(f"File too large: {file_size:.1f}MB (max: {max_size_mb}MB)")
        
        try:
            # Read with various encodings
            encodings = ['utf-8', 'latin-1', 'cp1252']
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding, dtype=str, na_filter=False)
                    logger.info(f"Successfully read CSV with {encoding} encoding")
                    logger.info(f"DataFrame shape: {df.shape}")
                    return df
                except UnicodeDecodeError:
                    continue
            
            raise ValueError("Could not decode CSV file with any supported encoding")
            
        except Exception as e:
            raise ValueError(f"Error reading CSV file: {e}")
    
    def _mask_cell_pii(self, cell_value: str, entities: List[PIIEntity]) -> str:
        """Mask PII in a cell value"""
        if not entities:
            return cell_value
        
        # Sort entities by position (reverse order for proper replacement)
        entities.sort(key=lambda x: x.start_pos, reverse=True)
        
        masked_value = cell_value
        for entity in entities:
            mask_token = f"[{entity.pii_type.value.upper()}_REDACTED]"
            masked_value = (
                masked_value[:entity.start_pos] + 
                mask_token + 
                masked_value[entity.end_pos:]
            )
        
        return masked_value
    
    def _generate_output_files(self, entities: List[PIIEntity], masked_df: pd.DataFrame,
                              graph_analysis: Dict, output_path: Path, 
                              base_name: str) -> Dict[str, Any]:
        """Generate output files and return results summary"""
        
        # Generate JSON report
        report = {
            "summary": {
                "total_entities": len(entities),
                "pii_types_detected": list(set(e.pii_type.value for e in entities)),
                "risk_distribution": self._calculate_risk_distribution(entities),
                "detection_methods": list(set(e.detection_method for e in entities))
            },
            "entities": [entity.to_dict() for entity in entities],
            "graph_analysis": graph_analysis,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
        # Save JSON report
        json_path = output_path / f"{base_name}_pii_report.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Save masked CSV
        csv_path = output_path / f"{base_name}_masked.csv"
        masked_df.to_csv(csv_path, index=False)
        
        # Save detailed analysis
        analysis_path = output_path / f"{base_name}_analysis.json"
        with open(analysis_path, 'w', encoding='utf-8') as f:
            json.dump(graph_analysis, f, indent=2, default=str)
        
        results = {
            "json_report_path": str(json_path),
            "masked_csv_path": str(csv_path),
            "analysis_path": str(analysis_path),
            "summary": report["summary"],
            "high_risk_entities": [
                e.to_dict() for e in entities 
                if e.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]
            ]
        }
        
        logger.info(f"Output files generated in: {output_path}")
        return results
    
    def _calculate_risk_distribution(self, entities: List[PIIEntity]) -> Dict[str, int]:
        """Calculate distribution of entities by risk level"""
        risk_counts = defaultdict(int)
        for entity in entities:
            risk_counts[entity.risk_level.value] += 1
        return dict(risk_counts)

def main():
    """Example usage of the Advanced PII Agent"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Advanced PII Detection Agent")
    parser.add_argument("input_csv", help="Path to input CSV file")
    parser.add_argument("--output-dir", default="./output", help="Output directory")
    parser.add_argument("--spacy-model", default="en_core_web_sm", help="spaCy model to use")
    
    args = parser.parse_args()
    
    # Initialize agent
    agent = AdvancedPIIAgent(spacy_model=args.spacy_model)
    
    # Process CSV
    results = agent.process_csv(args.input_csv, args.output_dir)
    
    if "error" in results:
        print(f"Error: {results['error']}")
        return 1
    
    # Print summary
    print("\n" + "="*50)
    print("PII DETECTION RESULTS")
    print("="*50)
    
    summary = results["summary"]
    print(f"Total PII entities detected: {summary['total_entities']}")
    print(f"PII types found: {', '.join(summary['pii_types_detected'])}")
    print(f"Risk distribution: {summary['risk_distribution']}")
    
    if results["high_risk_entities"]:
        print(f"\nHigh-risk entities: {len(results['high_risk_entities'])}")
        for entity in results["high_risk_entities"][:5]:  # Show first 5
            print(f"  - {entity['pii_type']}: {entity['text'][:30]}... (Risk: {entity['risk_level']})")
    
    print(f"\nFiles generated:")
    print(f"  - Report: {results['json_report_path']}")
    print(f"  - Masked CSV: {results['masked_csv_path']}")
    print(f"  - Visualization: {results.get('visualization_path', 'N/A')}")
    
    return 0

if __name__ == "__main__":
    exit(main())