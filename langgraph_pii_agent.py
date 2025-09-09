#!/usr/bin/env python3
"""
LangGraph-powered PII Detection Agent
Integrates the Advanced PII Agent with LangGraph for enhanced reasoning and workflow management
"""

import json
import logging
import asyncio
from typing import Dict, List, Any, Optional, TypedDict
from pathlib import Path
import pandas as pd

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor, ToolInvocation
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# Import our advanced PII agent
from advanced_pii_agent import (
    AdvancedPIIAgent, PIIEntity, PIIType, RiskLevel,
    PIINERDetector, ProximityAnalyzer, PIIGraphBuilder
)

logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    """State for the LangGraph PII Agent"""
    messages: List[BaseMessage]
    csv_file_path: Optional[str]
    output_directory: Optional[str]
    detected_entities: List[Dict]
    analysis_results: Dict[str, Any]
    risk_assessment: Dict[str, Any]
    current_step: str
    error_message: Optional[str]
    recommendations: List[str]

class LangGraphPIIAgent:
    """
    LangGraph-powered PII detection agent that provides intelligent reasoning
    and adaptive workflow management for PII analysis
    """
    
    def __init__(self, google_api_key: str, spacy_model: str = "en_core_web_sm"):
        """
        Initialize the LangGraph PII Agent
        
        Args:
            google_api_key: Google API key for Gemini
            spacy_model: spaCy model for NER
        """
        self.google_api_key = google_api_key
        self.pii_agent = AdvancedPIIAgent(spacy_model)
        
        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=google_api_key,
            temperature=0.1  # Low temperature for consistent analysis
        )
        
        # Build the agent graph
        self.graph = self._build_graph()
        
        logger.info("LangGraph PII Agent initialized successfully")
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("validate_input", self._validate_input)
        workflow.add_node("analyze_csv", self._analyze_csv)
        workflow.add_node("assess_risk", self._assess_risk)
        workflow.add_node("generate_recommendations", self._generate_recommendations)
        workflow.add_node("create_report", self._create_report)
        workflow.add_node("handle_error", self._handle_error)
        
        # Define the workflow
        workflow.set_entry_point("validate_input")
        
        workflow.add_conditional_edges(
            "validate_input",
            self._should_continue_after_validation,
            {
                "continue": "analyze_csv",
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "analyze_csv",
            self._should_continue_after_analysis,
            {
                "continue": "assess_risk",
                "error": "handle_error"
            }
        )
        
        workflow.add_edge("assess_risk", "generate_recommendations")
        workflow.add_edge("generate_recommendations", "create_report")
        workflow.add_edge("create_report", END)
        workflow.add_edge("handle_error", END)
        
        return workflow.compile()
    
    async def process_file(self, csv_file_path: str, output_directory: str = "./output") -> Dict[str, Any]:
        """
        Process CSV file using the LangGraph workflow
        
        Args:
            csv_file_path: Path to CSV file
            output_directory: Output directory for results
            
        Returns:
            Complete analysis results
        """
        initial_state = AgentState(
            messages=[HumanMessage(content=f"Analyze PII in CSV file: {csv_file_path}")],
            csv_file_path=csv_file_path,
            output_directory=output_directory,
            detected_entities=[],
            analysis_results={},
            risk_assessment={},
            current_step="validation",
            error_message=None,
            recommendations=[]
        )
        
        try:
            final_state = await self.graph.ainvoke(initial_state)
            return self._format_final_results(final_state)
        except Exception as e:
            logger.error(f"Error in LangGraph workflow: {e}")
            return {"error": str(e), "status": "failed"}
    
    async def _validate_input(self, state: AgentState) -> AgentState:
        """Validate input parameters and file"""
        logger.info("Step 1: Validating input...")
        state["current_step"] = "validation"
        
        csv_path = Path(state["csv_file_path"])
        
        # Check file existence
        if not csv_path.exists():
            state["error_message"] = f"File not found: {state['csv_file_path']}"
            return state
        
        # Check file size
        file_size_mb = csv_path.stat().st_size / (1024 * 1024)
        if file_size_mb > 100:  # 100MB limit
            state["error_message"] = f"File too large: {file_size_mb:.1f}MB (max: 100MB)"
            return state
        
        # Create output directory
        output_path = Path(state["output_directory"])
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Use LLM to assess file characteristics
        try:
            sample_df = pd.read_csv(csv_path, nrows=5)  # Read first 5 rows
            file_info = {
                "rows": len(sample_df) + 1,  # +1 for header
                "columns": len(sample_df.columns),
                "column_names": list(sample_df.columns),
                "sample_data": sample_df.head(3).to_dict()
            }
            
            assessment_prompt = f"""
            Analyze this CSV file structure for PII risk assessment:
            
            File: {csv_path.name}
            Size: {file_size_mb:.2f}MB
            Columns: {file_info['columns']}
            Column names: {', '.join(file_info['column_names'])}
            
            Based on the column names, identify potential PII risk areas and suggest focus areas for analysis.
            Respond in JSON format with: risk_level, focus_columns, expected_pii_types.
            """
            
            response = await self.llm.ainvoke([HumanMessage(content=assessment_prompt)])
            
            state["messages"].append(AIMessage(content=f"File validation complete. {response.content}"))
            state["analysis_results"]["file_assessment"] = {
                "file_info": file_info,
                "llm_assessment": response.content
            }
            
        except Exception as e:
            logger.warning(f"Could not perform LLM assessment: {e}")
            state["analysis_results"]["file_assessment"] = {"error": str(e)}
        
        logger.info("Input validation completed successfully")
        return state
    
    async def _analyze_csv(self, state: AgentState) -> AgentState:
        """Perform comprehensive PII analysis"""
        logger.info("Step 2: Analyzing CSV for PII...")
        state["current_step"] = "analysis"
        
        try:
            # Run the advanced PII analysis
            results = self.pii_agent.process_csv(
                state["csv_file_path"], 
                state["output_directory"]
            )
            
            if "error" in results:
                state["error_message"] = results["error"]
                return state
            
            # Extract entities and convert to serializable format
            entities_data = []
            if "high_risk_entities" in results:
                entities_data.extend(results["high_risk_entities"])
            
            state["detected_entities"] = entities_data
            state["analysis_results"].update(results)
            
            # Use LLM to interpret results
            interpretation_prompt = f"""
            Analyze these PII detection results:
            
            Total entities detected: {results['summary']['total_entities']}
            PII types found: {', '.join(results['summary']['pii_types_detected'])}
            Risk distribution: {results['summary']['risk_distribution']}
            High-risk entities: {len(results.get('high_risk_entities', []))}
            
            Provide insights about the PII exposure level and potential compliance concerns.
            Focus on GDPR, CCPA, and HIPAA implications if applicable.
            """
            
            interpretation = await self.llm.ainvoke([HumanMessage(content=interpretation_prompt)])
            state["analysis_results"]["llm_interpretation"] = interpretation.content
            state["messages"].append(AIMessage(content=interpretation.content))
            
            logger.info(f"Analysis complete. Found {len(entities_data)} high-risk PII entities")
            
        except Exception as e:
            state["error_message"] = f"Analysis error: {str(e)}"
            logger.error(f"Analysis error: {e}")
        
        return state
    
    async def _assess_risk(self, state: AgentState) -> AgentState:
        """Assess overall risk using LLM reasoning"""
        logger.info("Step 3: Assessing PII risks...")
        state["current_step"] = "risk_assessment"
        
        try:
            # Prepare risk assessment data
            summary = state["analysis_results"]["summary"]
            entities = state["detected_entities"]
            
            # Calculate risk metrics
            total_entities = summary["total_entities"]
            high_risk_count = len([e for e in entities if e.get("risk_level") in ["high", "critical"]])
            risk_ratio = high_risk_count / max(total_entities, 1)
            
            # Categorize PII types by regulation impact
            regulated_types = {
                "GDPR": ["person_name", "email_address", "phone_number", "address"],
                "HIPAA": ["medical_identifier", "person_name", "address"],
                "PCI_DSS": ["credit_card", "bank_account"],
                "CCPA": ["person_name", "email_address", "phone_number", "ip_address"]
            }
            
            regulation_impact = {}
            for regulation, types in regulated_types.items():
                affected_types = [t for t in summary["pii_types_detected"] if t in types]
                regulation_impact[regulation] = {
                    "affected": len(affected_types) > 0,
                    "types": affected_types,
                    "count": len(affected_types)
                }
            
            risk_prompt = f"""
            Perform comprehensive risk assessment for PII exposure:
            
            DETECTION SUMMARY:
            - Total PII entities: {total_entities}
            - High/Critical risk entities: {high_risk_count}
            - Risk ratio: {risk_ratio:.2%}
            - PII types: {', '.join(summary['pii_types_detected'])}
            
            REGULATORY IMPACT:
            {json.dumps(regulation_impact, indent=2)}
            
            GRAPH ANALYSIS:
            {json.dumps(state['analysis_results'].get('graph_analysis', {}), indent=2)}
            
            Provide:
            1. Overall risk score (0-100)
            2. Primary risk factors
            3. Regulatory compliance concerns
            4. Data breach impact assessment
            5. Recommended security controls
            
            Format response as JSON with these fields.
            """
            
            risk_response = await self.llm.ainvoke([HumanMessage(content=risk_prompt)])
            
            # Parse and structure risk assessment
            risk_assessment = {
                "total_entities": total_entities,
                "high_risk_entities": high_risk_count,
                "risk_ratio": risk_ratio,
                "regulation_impact": regulation_impact,
                "llm_assessment": risk_response.content,
                "timestamp": pd.Timestamp.now().isoformat()
            }
            
            state["risk_assessment"] = risk_assessment
            state["messages"].append(AIMessage(content=f"Risk assessment complete. {risk_response.content}"))
            
            logger.info(f"Risk assessment complete. Risk ratio: {risk_ratio:.2%}")
            
        except Exception as e:
            state["error_message"] = f"Risk assessment error: {str(e)}"
            logger.error(f"Risk assessment error: {e}")
        
        return state
    
    async def _generate_recommendations(self, state: AgentState) -> AgentState:
        """Generate actionable recommendations using LLM"""
        logger.info("Step 4: Generating recommendations...")
        state["current_step"] = "recommendations"
        
        try:
            # Prepare context for recommendations
            risk_assessment = state["risk_assessment"]
            summary = state["analysis_results"]["summary"]
            
            recommendations_prompt = f"""
            Based on the PII analysis results, generate specific actionable recommendations:
            
            CURRENT STATE:
            - Risk Ratio: {risk_assessment.get('risk_ratio', 0):.2%}
            - High-risk entities: {risk_assessment.get('high_risk_entities', 0)}
            - Regulatory concerns: {list(risk_assessment.get('regulation_impact', {}).keys())}
            
            DETECTION METHODS USED:
            {', '.join(summary.get('detection_methods', []))}
            
            Generate recommendations for:
            1. Immediate actions (0-30 days)
            2. Short-term improvements (1-6 months)
            3. Long-term strategy (6+ months)
            4. Technical controls
            5. Process improvements
            6. Compliance measures
            
            For each recommendation, include:
            - Action description
            - Priority level (High/Medium/Low)
            - Estimated effort
            - Expected impact
            
            Format as structured JSON.
            """
            
            recommendations_response = await self.llm.ainvoke([HumanMessage(content=recommendations_prompt)])
            
            # Also generate specific technical recommendations
            technical_prompt = f"""
            Provide specific technical recommendations for PII protection:
            
            DETECTED PII TYPES: {', '.join(summary['pii_types_detected'])}
            GRAPH CLUSTERS: {state['analysis_results'].get('graph_analysis', {}).get('connected_components', {})}
            
            Recommend specific:
            1. Data masking techniques for each PII type
            2. Access control measures
            3. Encryption strategies
            4. Monitoring and alerting
            5. Data retention policies
            6. Incident response procedures
            
            Be specific and actionable.
            """
            
            technical_response = await self.llm.ainvoke([HumanMessage(content=technical_prompt)])
            
            recommendations = {
                "strategic_recommendations": recommendations_response.content,
                "technical_recommendations": technical_response.content,
                "generated_at": pd.Timestamp.now().isoformat()
            }
            
            state["recommendations"] = [recommendations_response.content, technical_response.content]
            state["analysis_results"]["recommendations"] = recommendations
            
            state["messages"].append(AIMessage(content="Comprehensive recommendations generated"))
            
            logger.info("Recommendations generated successfully")
            
        except Exception as e:
            state["error_message"] = f"Recommendations error: {str(e)}"
            logger.error(f"Recommendations error: {e}")
        
        return state
    
    async def _create_report(self, state: AgentState) -> AgentState:
        """Create final comprehensive report"""
        logger.info("Step 5: Creating final report...")
        state["current_step"] = "reporting"
        
        try:
            # Generate executive summary
            executive_prompt = f"""
            Create an executive summary for PII analysis report:
            
            KEY FINDINGS:
            - Total PII entities: {state['analysis_results']['summary']['total_entities']}
            - Risk level: {state['risk_assessment'].get('risk_ratio', 0):.2%} high-risk
            - Regulatory impact: {len(state['risk_assessment'].get('regulation_impact', {}))} regulations affected
            
            Create a professional executive summary suitable for management review.
            Include key risks, compliance concerns, and priority recommendations.
            Keep it concise but comprehensive.
            """
            
            executive_summary = await self.llm.ainvoke([HumanMessage(content=executive_prompt)])
            
            # Compile comprehensive report
            final_report = {
                "executive_summary": executive_summary.content,
                "analysis_metadata": {
                    "file_analyzed": state["csv_file_path"],
                    "analysis_timestamp": pd.Timestamp.now().isoformat(),
                    "agent_version": "1.0.0",
                    "methods_used": ["NER", "Proximity Analysis", "Graph Theory"]
                },
                "pii_detection_results": state["analysis_results"]["summary"],
                "risk_assessment": state["risk_assessment"],
                "recommendations": state["analysis_results"].get("recommendations", {}),
                "detailed_findings": {
                    "high_risk_entities": state["detected_entities"],
                    "graph_analysis": state["analysis_results"].get("graph_analysis", {}),
                    "file_assessment": state["analysis_results"].get("file_assessment", {})
                }
            }
            
            # Save comprehensive report
            output_path = Path(state["output_directory"])
            report_path = output_path / "comprehensive_pii_report.json"
            
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(final_report, f, indent=2, default=str)
            
            state["analysis_results"]["comprehensive_report_path"] = str(report_path)
            state["messages"].append(AIMessage(content="Comprehensive report generated successfully"))
            
            logger.info(f"Final report saved to: {report_path}")
            
        except Exception as e:
            state["error_message"] = f"Reporting error: {str(e)}"
            logger.error(f"Reporting error: {e}")
        
        return state
    
    async def _handle_error(self, state: AgentState) -> AgentState:
        """Handle errors in the workflow"""
        logger.error(f"Error in step {state['current_step']}: {state['error_message']}")
        
        # Generate error report with LLM assistance
        error_prompt = f"""
        An error occurred during PII analysis:
        
        Step: {state['current_step']}
        Error: {state['error_message']}
        
        Provide suggestions for resolving this error and alternative approaches.
        """
        
        try:
            error_response = await self.llm.ainvoke([HumanMessage(content=error_prompt)])
            state["messages"].append(AIMessage(content=f"Error analysis: {error_response.content}"))
        except Exception:
            state["messages"].append(AIMessage(content="Error handling failed"))
        
        return state
    
    def _should_continue_after_validation(self, state: AgentState) -> str:
        """Decide whether to continue after validation"""
        return "error" if state.get("error_message") else "continue"
    
    def _should_continue_after_analysis(self, state: AgentState) -> str:
        """Decide whether to continue after analysis"""
        return "error" if state.get("error_message") else "continue"
    
    def _format_final_results(self, state: AgentState) -> Dict[str, Any]:
        """Format final results for return"""
        if state.get("error_message"):
            return {
                "status": "error",
                "error": state["error_message"],
                "step_failed": state["current_step"]
            }
        
        return {
            "status": "success",
            "summary": state["analysis_results"].get("summary", {}),
            "risk_assessment": state["risk_assessment"],
            "recommendations": state["recommendations"],
            "output_files": {
                "comprehensive_report": state["analysis_results"].get("comprehensive_report_path"),
                "json_report": state["analysis_results"].get("json_report_path"),
                "masked_csv": state["analysis_results"].get("masked_csv_path"),
                "visualization": state["analysis_results"].get("visualization_path")
            },
            "conversation_log": [msg.content for msg in state["messages"]]
        }

async def main():
    """Example usage of LangGraph PII Agent"""
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="LangGraph PII Detection Agent")
    parser.add_argument("input_csv", help="Path to input CSV file")
    parser.add_argument("--output-dir", default="./output", help="Output directory")
    parser.add_argument("--google-api-key", help="Google API key (or set GOOGLE_API_KEY env var)")
    
    args = parser.parse_args()
    
    # Get API key
    api_key = args.google_api_key or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: Google API key required. Use --google-api-key or set GOOGLE_API_KEY environment variable")
        return 1
    
    try:
        # Initialize agent
        agent = LangGraphPIIAgent(google_api_key=api_key)
        
        # Process file
        results = await agent.process_file(args.input_csv, args.output_dir)
        
        # Display results
        print("\n" + "="*60)
        print("LANGGRAPH PII AGENT RESULTS")
        print("="*60)
        
        if results["status"] == "error":
            print(f"Error: {results['error']}")
            print(f"Failed at step: {results.get('step_failed', 'unknown')}")
            return 1
        
        summary = results["summary"]
        print(f"Status: {results['status']}")
        print(f"Total PII entities: {summary.get('total_entities', 0)}")
        print(f"PII types detected: {', '.join(summary.get('pii_types_detected', []))}")
        print(f"Risk distribution: {summary.get('risk_distribution', {})}")
        
        risk_assessment = results["risk_assessment"]
        print(f"\nRisk Assessment:")
        print(f"  Risk ratio: {risk_assessment.get('risk_ratio', 0):.2%}")
        print(f"  High-risk entities: {risk_assessment.get('high_risk_entities', 0)}")
        
        print(f"\nOutput files:")
        for file_type, path in results["output_files"].items():
            if path:
                print(f"  {file_type}: {path}")
        
        print(f"\nRecommendations generated: {len(results.get('recommendations', []))}")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    import asyncio
    exit(asyncio.run(main()))