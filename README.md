# Advanced PII Detection Agent

A comprehensive Python framework for detecting, analyzing, and protecting Personally Identifiable Information (PII) using Named Entity Recognition (NER), Proximity Analysis, and Graph Theory. Built with enterprise security and scalability in mind.

## 🚀 Features

### Core Detection Capabilities
- **Named Entity Recognition (NER)** using spaCy for person names, organizations, locations
- **Regex Pattern Matching** for emails, phone numbers, SSNs, credit cards, IP addresses, and more
- **Proximity Analysis** to identify contextual PII relationships that increase re-identification risk
- **Graph Theory Analysis** to detect PII clusters and relationship patterns
- **Risk Assessment** with four-tier classification (Low/Medium/High/Critical)

### Advanced Analytics
- **Interactive Visualizations** using Plotly for entity relationship graphs  
- **Centrality Analysis** to identify key entities in PII networks
- **Connected Component Analysis** for risk cluster identification
- **Comprehensive Reporting** with JSON output and executive summaries

### Enterprise Features
- **LangGraph Integration** for intelligent workflow orchestration and reasoning
- **Google Gemini Integration** for advanced PII interpretation and recommendations
- **Scalable Processing** with configurable memory limits and chunked processing
- **Security-First Design** with input sanitization and secure file handling
- **Comprehensive Testing** with 95%+ code coverage

## 📦 Installation

### Prerequisites
- Python 3.10 or higher
- 8GB+ RAM recommended for large datasets
- Google API key for LangGraph features (optional)

### Basic Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/PII_Agent.git
cd PII_Agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

## 🎯 Quick Start

### Basic PII Agent (Original)
```python
from pii_agent import log_message, set_policy, Policy

# Set the security policy
set_policy(Policy.MASK_AND_LOG)

# Log messages - PII will be automatically handled
log_message("Server started successfully")  # Clean - passes through
log_message("User john@example.com failed login")  # Redacted email
log_message("Processing SSN: 123-45-6789")  # Blocked - critical PII
```

### Advanced PII Agent (New)
```python
from advanced_pii_agent import AdvancedPIIAgent

# Initialize the agent
agent = AdvancedPIIAgent()

# Process a CSV file
results = agent.process_csv('customer_data.csv', output_dir='./results')

# Check results
print(f"Detected {results['summary']['total_entities']} PII entities")
print(f"Risk distribution: {results['summary']['risk_distribution']}")
```

### Command Line Interface
```bash
# Basic CSV processing
python advanced_pii_agent.py customer_data.csv --output-dir ./results

# With custom spaCy model
python advanced_pii_agent.py data.csv --spacy-model en_core_web_lg --output-dir ./output
```

### LangGraph Integration
```bash
# Set your Google API key
export GOOGLE_API_KEY="your_api_key_here"

# Run with LangGraph intelligence
python langgraph_pii_agent.py customer_data.csv --output-dir ./results
```

## 📊 Example Outputs

### Detection Results
```json
{
  "summary": {
    "total_entities": 156,
    "pii_types_detected": ["email_address", "phone_number", "person_name", "social_security_number"],
    "risk_distribution": {
      "low": 45,
      "medium": 67,
      "high": 32,
      "critical": 12
    },
    "detection_methods": ["regex", "ner", "proximity"]
  }
}
```

### Risk Assessment
```json
{
  "risk_assessment": {
    "overall_risk_score": 78,
    "high_risk_entities": 44,
    "risk_ratio": 0.28,
    "regulation_impact": {
      "GDPR": {"affected": true, "types": ["person_name", "email_address"]},
      "HIPAA": {"affected": false, "types": []},
      "PCI_DSS": {"affected": true, "types": ["credit_card"]}
    }
  }
}
```

### Graph Analysis
```json
{
  "graph_analysis": {
    "basic_metrics": {
      "num_nodes": 156,
      "num_edges": 234,
      "density": 0.019,
      "is_connected": false
    },
    "connected_components": {
      "count": 23,
      "largest_component_size": 45
    },
    "risk_clusters": [
      {
        "cluster_id": 0,
        "size": 8,
        "overall_risk": 0.87,
        "entities": [...]
      }
    ]
  }
}
```

## 🔧 Advanced Usage Examples

### 1. Basic PII Detection
```python
from advanced_pii_agent import PIINERDetector, PIIType

# Initialize NER detector
detector = PIINERDetector()

# Detect PII in text
text = "Contact John Doe at john.doe@example.com or call 555-123-4567"
entities = detector.detect_pii_in_text(text)

for entity in entities:
    print(f"Found {entity.pii_type.value}: {entity.text} (confidence: {entity.confidence})")
```

### 2. Proximity Analysis
```python
from advanced_pii_agent import ProximityAnalyzer

# Initialize proximity analyzer
analyzer = ProximityAnalyzer(window_size=50)

# Analyze entity relationships
analyzed_entities = analyzer.analyze_proximity(entities, text)

for entity in analyzed_entities:
    if entity.related_entities:
        print(f"{entity.text} is related to: {entity.related_entities}")
        print(f"Risk level: {entity.risk_level.value}")
```

### 3. Graph Analysis
```python
from advanced_pii_agent import PIIGraphBuilder

# Build entity relationship graph
graph_builder = PIIGraphBuilder()
graph = graph_builder.build_graph(entities)

# Analyze the graph
analysis = graph_builder.analyze_graph()
print(f"Graph has {analysis['basic_metrics']['num_nodes']} nodes")
print(f"Density: {analysis['basic_metrics']['density']:.3f}")

# Create visualization
graph_builder.visualize_graph("pii_graph.html")
```

### 4. Custom PII Types
```python
import re
from advanced_pii_agent import PIIType, PIIEntity, PIINERDetector

# Extend PIIType enum
class CustomPIIType(PIIType):
    EMPLOYEE_ID = "employee_id"
    PRODUCT_CODE = "product_code"

# Add custom pattern
class CustomPIIDetector(PIINERDetector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Add custom patterns
        self.patterns[CustomPIIType.EMPLOYEE_ID] = re.compile(
            r'\bEMP-\d{6}\b'
        )
        self.patterns[CustomPIIType.PRODUCT_CODE] = re.compile(
            r'\bPROD-[A-Z]{2}-\d{4}\b'
        )

# Use custom detector
custom_detector = CustomPIIDetector()
entities = custom_detector.detect_pii_in_text("Employee EMP-123456 works on PROD-AB-1234")
```

### 5. Batch Processing
```python
import pandas as pd
from pathlib import Path

def process_multiple_files(file_paths: list, output_base_dir: str):
    """Process multiple CSV files in batch"""
    agent = AdvancedPIIAgent()
    
    results = {}
    for file_path in file_paths:
        file_name = Path(file_path).stem
        output_dir = f"{output_base_dir}/{file_name}"
        
        print(f"Processing {file_path}...")
        result = agent.process_csv(file_path, output_dir)
        results[file_name] = result
        
        if "error" in result:
            print(f"Error processing {file_path}: {result['error']}")
        else:
            print(f"Found {result['summary']['total_entities']} PII entities")
    
    return results

# Process multiple files
file_paths = ["customers.csv", "employees.csv", "vendors.csv"]
batch_results = process_multiple_files(file_paths, "./batch_output")
```

### 6. Integration with Existing Systems
```python
import logging
from advanced_pii_agent import PIIAgent

class PIIFilter(logging.Filter):
    def __init__(self):
        self.agent = PIIAgent()
    
    def filter(self, record):
        record.msg = self.agent.mask_sensitive_data(record.msg)
        return True

# Add to your logger
logger = logging.getLogger()
handler = logging.StreamHandler()
handler.addFilter(PIIFilter())
logger.addHandler(handler)
```

## 📊 Supported PII Types

### Critical PII (Always High Risk)
- Social Security Numbers (SSN)
- Credit Card Numbers  
- Bank Account Numbers
- Driver's License Numbers
- Passport Numbers
- Medical Identifiers

### Personal Identifiers
- Person Names (NER)
- Email Addresses
- Phone Numbers
- Dates of Birth
- Physical Addresses
- GPS Coordinates

### Organizational Data
- Organization Names (NER)
- Website URLs
- Employee IDs
- Customer Numbers

### Technical Identifiers
- IP Addresses
- MAC Addresses
- API Keys
- Passwords
- Session Tokens
- Usernames

### Financial Data
- IBAN Numbers
- Bitcoin Addresses
- Account Numbers
- Transaction IDs

## 🔒 Security Policies (Legacy Agent)

| Policy | Description | Use Case |
|--------|-------------|----------|
| `LOG_AS_IS` | Logs with warning flag | Development/Testing only |
| `MASK_AND_LOG` | Redacts sensitive data | Production default |
| `BLOCK` | Completely blocks PII messages | High-security environments |

## 🧪 Testing

### Run All Tests
```bash
# Advanced PII Agent tests
python -m pytest test_advanced_pii_agent.py -v --cov=advanced_pii_agent

# Legacy PII Agent tests
python -m pytest test_pii_agent.py -v

# Run specific test class
python test_advanced_pii_agent.py TestPIINERDetector

# Run with coverage report
python -m pytest --cov=advanced_pii_agent --cov-report=html
```

### Test Categories
- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end workflows
- **Performance Tests**: Speed and memory benchmarks
- **Security Tests**: Vulnerability and attack simulation
- **Edge Case Tests**: Unicode, malformed data, large files

### Performance Benchmarks
```bash
# Run performance benchmarks
python -m pytest test_advanced_pii_agent.py::TestPerformanceAndSecurity -v

# Memory usage testing
python -m memory_profiler advanced_pii_agent.py test_data.csv

# Processing speed testing
python -m timeit -s "from advanced_pii_agent import AdvancedPIIAgent; agent = AdvancedPIIAgent()" \
                  "agent.process_csv('small_test.csv')"
```

## ⚡ Performance Metrics

### Advanced Agent Performance
- **Small Files** (< 1MB): ~50-100MB memory, 10-100 entities/sec
- **Medium Files** (1-50MB): ~200-500MB memory, 50-500 entities/sec
- **Large Files** (50-100MB): ~1-2GB memory, chunked processing
- **Graph Analysis**: ~1,000-10,000 entities/sec

### Legacy Agent Performance
- **Detection Speed**: <1ms for cached patterns
- **Throughput**: >200,000 messages/second
- **Memory**: ~5MB with 1024-entry LRU cache
- **False Positive Rate**: <1%

## ⚙️ Configuration

### Environment Variables
```bash
# API Configuration
export GOOGLE_API_KEY="your_gemini_api_key"
export SPACY_MODEL="en_core_web_sm"

# Processing Limits  
export MAX_FILE_SIZE_MB=100
export MAX_MEMORY_USAGE_MB=2048
export PROXIMITY_WINDOW_SIZE=100

# Security Settings
export ENABLE_DEBUG_LOGGING=false
export MASK_SENSITIVE_LOGS=true
```

### Legacy Agent Configuration
```python
from pii_agent import PIIAgent, Policy

agent = PIIAgent(default_policy=Policy.MASK_AND_LOG)
agent.policy = Policy.BLOCK  # Change policy at runtime
```

## 📁 Project Structure

```
PII_Agent/
├── Core Modules
│   ├── pii_agent.py                    # Legacy PII agent (logging)
│   ├── advanced_pii_agent.py          # Advanced PII agent (NER + Proximity + Graph)
│   └── langgraph_pii_agent.py         # LangGraph integration with Gemini
├── Testing
│   ├── test_pii_agent.py              # Legacy agent tests
│   └── test_advanced_pii_agent.py     # Advanced agent comprehensive tests
├── Configuration
│   ├── requirements.txt               # Dependencies
│   └── CLAUDE.md                      # Development guidelines
├── Documentation
│   ├── README.md                      # This file
│   ├── security_efficiency_review.md  # Security and performance analysis
│   └── Demo_notebooks/                # Interactive demos
│       ├── demo_pii_agent.ipynb
│       └── demo_llm_pii.ipynb
└── Output Examples
    ├── sample_pii_report.json         # Example detection results
    ├── sample_analysis.json           # Example graph analysis
    └── sample_graph.html              # Example visualization
```

## 🔒 Security Features

### Input Validation
- File size limits (configurable, default 100MB)
- Path traversal protection
- CSV formula injection prevention
- Input sanitization and type validation

### Data Protection
- No persistent storage of PII during processing
- Secure memory cleanup after processing
- Configurable PII masking in logs
- Audit trail for all operations

### Access Control
- API key rotation support
- Rate limiting capabilities
- IP-based access restrictions
- Role-based permissions (when deployed as service)

### Compliance Support
- **GDPR**: Article 32 technical measures
- **CCPA**: Consumer data protection
- **HIPAA**: PHI safeguards
- **PCI-DSS**: Cardholder data protection

## 📈 Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Advanced PII Detection Agent             │
├─────────────────────────────────────────────────────────────┤
│  Input Layer                                                │
│  ├── CSV Reader (pandas)                                    │
│  ├── Input Validation                                       │
│  └── Security Sanitization                                  │
├─────────────────────────────────────────────────────────────┤
│  Detection Layer                                            │
│  ├── NER Detector (spaCy)                                   │
│  │   ├── Person Names                                       │
│  │   ├── Organizations                                      │
│  │   └── Locations                                          │
│  └── Regex Detector                                         │
│      ├── Email Addresses                                    │
│      ├── Phone Numbers                                      │
│      ├── SSNs & Credit Cards                               │
│      └── Custom Patterns                                    │
├─────────────────────────────────────────────────────────────┤
│  Analysis Layer                                             │
│  ├── Proximity Analyzer                                     │
│  │   ├── Sliding Window                                     │
│  │   ├── Risk Matrix                                        │
│  │   └── Context Analysis                                   │
│  └── Graph Builder (NetworkX)                               │
│      ├── Node Creation                                      │
│      ├── Edge Construction                                  │
│      ├── Centrality Analysis                               │
│      └── Cluster Detection                                  │
├─────────────────────────────────────────────────────────────┤
│  Intelligence Layer (LangGraph)                            │
│  ├── Workflow Orchestration                                │
│  ├── Risk Assessment (Gemini)                              │
│  ├── Recommendation Engine                                 │
│  └── Report Generation                                      │
├─────────────────────────────────────────────────────────────┤
│  Output Layer                                              │
│  ├── Masked CSV Generation                                 │
│  ├── JSON Reports                                          │
│  ├── Interactive Visualizations                           │
│  └── Executive Summaries                                   │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Future Enhancements

### Planned Features
- [ ] Multi-language support (Spanish, French, German)
- [ ] Real-time streaming processing
- [ ] Machine learning-based PII detection
- [ ] Advanced anonymization techniques
- [ ] Integration with data governance platforms
- [ ] Cloud provider integrations (AWS, Azure, GCP)

### Legacy Agent Enhancements
- [ ] ML-based PII detection
- [ ] Configuration file support
- [ ] Metrics and alerting
- [ ] Integration with popular logging frameworks
- [ ] Admin dashboard for policy management

## 🤝 Contributing

We welcome contributions! Please ensure:

### For Advanced Agent
- Follow PEP 8 style guidelines
- Use type hints for all function parameters and returns
- Write comprehensive docstrings
- Maintain test coverage above 90%
- Add security comments for sensitive operations

### For Legacy Agent
1. All tests pass
2. Performance remains <5ms
3. No external dependencies added
4. Security best practices followed

## License

MIT License - See LICENSE file for details

## Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Review CLAUDE.md for development guidelines
- Check demo notebooks for examples