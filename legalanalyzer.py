import asyncio
import json

# Add Azure OpenAI package (consistent with LegalAnalyzer.py coding standard)
from openai import AsyncAzureOpenAI

# ─────────────────────────────────────────────────────────────────────────────
#  Configuration  (mirrors LegalAnalyzer.py style)
# ─────────────────────────────────────────────────────────────────────────────

# Set to True to print the full response from OpenAI for each call
printFullResponse = False

# Azure OpenAI settings – from LegalAnalyzer.py
azure_oai_endpoint   = "https://b3openai26.openai.azure.com/"
azure_oai_key        = "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
azure_oai_deployment = "gpt-4o"

# Local knowledge base file (legal precedents, compliance rules, red flags)
GROUNDING_FILE = "legaldoc_grounding.txt"

# Compliance categories evaluated by all agents
COMPLIANCE_CATEGORIES = [
    "GDPR", "HIPAA", "SOX", "CCPA",
    "Contract Law", "Intellectual Property",
    "Liability Clauses", "Termination Clauses", "Data Retention",
]


# ─────────────────────────────────────────────────────────────────────────────
#  Agent 1  –  Document Ingestion Agent
#  Reads the legal document from a local file path provided by the user
# ─────────────────────────────────────────────────────────────────────────────

async def run_document_ingestion_agent(document_path: str) -> str:
    """
    Document Ingestion Agent: Reads the specified local legal document file
    and returns its full text content for downstream processing.
    """
    print(f"\n[Document Ingestion Agent] Reading '{document_path}'...")
    document_text = open(file=document_path, encoding="utf8").read().strip()
    print(f"[Document Ingestion Agent] Successfully ingested {len(document_text):,} characters.")
    return document_text


# ─────────────────────────────────────────────────────────────────────────────
#  Agent 2  –  Clause Extraction Agent
#  Uses Azure OpenAI + grounding context to identify and extract legal clauses
# ─────────────────────────────────────────────────────────────────────────────

async def run_clause_extraction_agent(
    document_text: str,
    grounding_text: str,
    client: AsyncAzureOpenAI,
) -> dict:
    """
    Clause Extraction Agent: Sends raw document text with grounding context
    to Azure OpenAI and receives a structured JSON list of identified legal
    clauses enriched with type, risk level, and key legal terms.
    """
    print("\n[Clause Extraction Agent] Extracting legal clauses from document...")

    system_prompt = """You are a legal clause extraction specialist.
Analyze the provided legal document and extract every distinct clause.
Return a JSON object with a 'clauses' array where each element contains:
  - "clause_id"   : sequential integer
  - "clause_type" : type of clause (e.g., Indemnification, Liability, IP, NDA, Termination, Arbitration)
  - "clause_text" : exact extracted text
  - "risk_level"  : preliminary risk assessment - Low | Medium | High
  - "keywords"    : array of key legal terms found in the clause
Return ONLY valid JSON. No additional commentary."""

    print("\nAdding grounding context from legaldoc_grounding.txt")
    user_prompt = (
        f"LEGAL KNOWLEDGE BASE (use as reference):\n{grounding_text}\n\n"
        f"Extract all legal clauses from the following document:\n\n{document_text}"
    )

    print("\nSending request to Azure OpenAI model for clause extraction...\n")
    response = await client.chat.completions.create(
        model=azure_oai_deployment,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=2500,
        response_format={"type": "json_object"},
    )

    if printFullResponse:
        print(response)

    clauses = json.loads(response.choices[0].message.content)
    print(f"[Clause Extraction Agent] Extracted {len(clauses.get('clauses', []))} clauses.")
    return clauses


# ─────────────────────────────────────────────────────────────────────────────
#  Agent 3  –  Compliance Validation Agent
#  Cross-references extracted clauses with compliance frameworks using
#  legaldoc_grounding.txt as the RAG knowledge base
# ─────────────────────────────────────────────────────────────────────────────

async def run_compliance_validation_agent(
    clauses: dict,
    grounding_text: str,
    client: AsyncAzureOpenAI,
) -> dict:
    """
    Compliance Validation Agent: Uses legaldoc_grounding.txt as the grounding
    knowledge base to validate each extracted clause against the configured
    compliance frameworks and regulatory categories.
    """
    print("\n[Compliance Validation Agent] Validating clauses against compliance frameworks...")

    system_prompt = f"""You are a senior legal compliance officer with expertise in multi-jurisdiction regulatory frameworks.
Evaluate each legal clause against the following compliance categories: {', '.join(COMPLIANCE_CATEGORIES)}.

Return a JSON object with:
  - "compliance_report": array of evaluations, one per clause, each containing:
      - "clause_id"            : matches input clause_id
      - "clause_type"          : from input
      - "compliance_status"    : Compliant | Non-Compliant | Needs-Review
      - "violated_frameworks"  : list of specific frameworks violated (empty if compliant)
      - "risk_level"           : Low | Medium | High | Critical
      - "issues"               : list of identified compliance issues
      - "recommendations"      : list of specific corrective actions
      - "supporting_precedent" : most relevant rule or precedent from the knowledge base
  - "overall_risk_score"      : integer 0-100 (0 = no risk, 100 = critical breach)
  - "critical_issues_count"   : integer count of Critical-risk items
  - "requires_legal_review"   : boolean - true if human legal review is mandatory
  - "compliance_summary"      : 2-3 sentence overall compliance assessment
Return ONLY valid JSON. No additional commentary."""

    print("\nAdding grounding context from legaldoc_grounding.txt")
    user_prompt = (
        f"LEGAL KNOWLEDGE BASE (use as grounding reference):\n{grounding_text}\n\n"
        f"Validate the following extracted clauses:\n\n{json.dumps(clauses, indent=2)}"
    )

    print("\nSending request to Azure OpenAI model for compliance validation...\n")
    response = await client.chat.completions.create(
        model=azure_oai_deployment,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.1,
        max_tokens=3500,
        response_format={"type": "json_object"},
    )

    if printFullResponse:
        print(response)

    compliance_report = json.loads(response.choices[0].message.content)
    risk_score     = compliance_report.get("overall_risk_score", 0)
    critical_count = compliance_report.get("critical_issues_count", 0)
    print(
        f"[Compliance Validation Agent] Validation complete. "
        f"Risk Score: {risk_score}/100  |  Critical Issues: {critical_count}"
    )
    return compliance_report


# ─────────────────────────────────────────────────────────────────────────────
#  Agent 4  –  Grounded Summary Agent
#  Produces a grounded executive summary using legaldoc_grounding.txt
# ─────────────────────────────────────────────────────────────────────────────

async def run_grounded_summary_agent(
    document_text: str,
    compliance_report: dict,
    grounding_text: str,
    client: AsyncAzureOpenAI,
) -> str:
    """
    Grounded Summary Agent: Synthesises document content and compliance findings
    into a concise executive summary grounded in legaldoc_grounding.txt.
    """
    print("\n[Grounded Summary Agent] Generating grounded executive summary...")

    system_prompt = """You are a legal analyst producing an executive summary for senior stakeholders.
Your response must be grounded in the provided compliance findings and legal knowledge base.
Structure your response as follows:

  1. Document Overview          (2-3 sentences, document type and purpose)
  2. Key Legal Clauses Found    (bulleted list of significant clauses with their risk level)
  3. Compliance Findings        (flagged issues with specific regulatory citations)
  4. Risk Assessment            (overall risk narrative with supporting knowledge base references)
  5. Recommended Actions        (numbered, prioritised, actionable steps)

Be factual, precise, and cite relevant rules or precedents from the knowledge base."""

    print("\nAdding grounding context from legaldoc_grounding.txt")
    user_prompt = (
        f"LEGAL KNOWLEDGE BASE (grounding context):\n{grounding_text}\n\n"
        f"Compliance Report:\n{json.dumps(compliance_report, indent=2)}\n\n"
        f"Original Document (first 3,000 characters):\n{document_text[:3000]}"
    )

    print("\nSending request to Azure OpenAI model for grounded summary...\n")
    response = await client.chat.completions.create(
        model=azure_oai_deployment,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.4,
        max_tokens=1800,
    )

    if printFullResponse:
        print(response)

    summary = response.choices[0].message.content
    print("[Grounded Summary Agent] Executive summary generated successfully.")
    return summary


# ─────────────────────────────────────────────────────────────────────────────
#  Orchestrator Agent
#  Coordinates all sub-agent outputs using Azure OpenAI and legaldoc_grounding.txt
# ─────────────────────────────────────────────────────────────────────────────

async def run_orchestrator_agent(
    document_name: str,
    compliance_report: dict,
    grounded_summary: str,
    grounding_text: str,
    client: AsyncAzureOpenAI,
) -> str:
    """
    Orchestrator Agent: Coordinates and synthesises all sub-agent findings using
    Azure OpenAI, grounded in legaldoc_grounding.txt. Produces a final consolidated
    legal review report with a verdict and prioritised action plan.
    """
    print("\n[Orchestrator Agent] Synthesising all agent findings...")

    system_prompt = f"""You are a senior legal document review orchestrator.
Your mission is to coordinate and synthesise findings from the following specialised agents:
  1. Document Ingestion Agent    - read the raw legal document from a local file
  2. Clause Extraction Agent     - identified and structured all legal clauses
  3. Compliance Validation Agent - validated each clause against: {', '.join(COMPLIANCE_CATEGORIES)}
  4. Grounded Summary Agent      - produced an executive summary grounded in the legal knowledge base

Your responsibilities:
  - Synthesise all agent findings into a single consolidated legal review report
  - Flag and prioritise all compliance issues by severity (Critical / High / Medium / Low)
  - Escalate immediately if overall_risk_score > 70
  - Conclude with a clear legal verdict (APPROVED / NEEDS REVISION / REJECTED)
  - Provide a numbered prioritised action plan
  - Ground all conclusions with specific references from the knowledge base"""

    risk_score = compliance_report.get("overall_risk_score", 0)
    print("\nAdding grounding context from legaldoc_grounding.txt")
    user_prompt = (
        f"LEGAL KNOWLEDGE BASE (grounding context):\n{grounding_text}\n\n"
        f"Document under review: {document_name}\n"
        f"Overall Risk Score: {risk_score}/100\n\n"
        f"Compliance Validation Report:\n{json.dumps(compliance_report, indent=2)}\n\n"
        f"Grounded Executive Summary:\n{grounded_summary}\n\n"
        "Instructions:\n"
        "1. Review all agent findings above.\n"
        "2. Flag and list all compliance issues grouped by severity.\n"
        "3. If risk score > 70, include an ESCALATION NOTICE at the top of your response.\n"
        "4. Provide a final consolidated legal review report with a clear verdict.\n"
        "5. End with a numbered prioritised action plan."
    )

    print("\nSending request to Azure OpenAI model for orchestration...\n")
    response = await client.chat.completions.create(
        model=azure_oai_deployment,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.3,
        max_tokens=2000,
    )

    if printFullResponse:
        print(response)

    print("[Orchestrator Agent] Consolidated report generated.")
    return response.choices[0].message.content


# ─────────────────────────────────────────────────────────────────────────────
#  Main  –  Entry point and multi-agent pipeline orchestration
# ─────────────────────────────────────────────────────────────────────────────

async def main():

    try:
        # ── Banner ─────────────────────────────────────────────────────────────
        print("=" * 62)
        print("  Legal Document Review  –  Multi-Agent AI System")
        print("  Powered by Azure OpenAI  |  Knowledge Base: legaldoc_grounding.txt")
        print("=" * 62)

        # Configure the Azure OpenAI client (same as LegalAnalyzer.py)
        client = AsyncAzureOpenAI(
            azure_endpoint = azure_oai_endpoint,
            api_key        = azure_oai_key,
            api_version    = "2024-02-15-preview"
        )

        # ── Interactive review loop ────────────────────────────────────────────
        while True:
            print("------------------\nPausing the app to allow you to select a legal document.")
            print("Press anything then enter to continue...")
            input()

            document_path = input("Enter the path to the legal document file: ").strip()
            if document_path.lower() == 'quit':
                print('Exiting program...')
                break
            if not document_path:
                print("No path provided. Please try again.")
                continue

            print(f"\n{'─' * 62}")
            print(f"  Processing: {document_path}")
            print(f"{'─' * 62}")

            # ── Load grounding knowledge base (legaldoc_grounding.txt) ─────────
            print("\nAdding grounding context from legaldoc_grounding.txt")
            grounding_text = open(file=GROUNDING_FILE, encoding="utf8").read().strip()

            # ── Agent 1: Document Ingestion ────────────────────────────────────
            document_text = await run_document_ingestion_agent(document_path)

            # ── Agent 2: Clause Extraction (grounded) ─────────────────────────
            clauses = await run_clause_extraction_agent(document_text, grounding_text, client)

            # ── Agent 3: Compliance Validation (grounded) ─────────────────────
            compliance_report = await run_compliance_validation_agent(
                clauses, grounding_text, client
            )

            # ── Agent 4: Grounded Executive Summary ───────────────────────────
            grounded_summary = await run_grounded_summary_agent(
                document_text, compliance_report, grounding_text, client
            )

            # ── Orchestrator: Consolidate + Produce Final Report ───────────────
            final_report = await run_orchestrator_agent(
                document_path, compliance_report, grounded_summary, grounding_text, client
            )

            # ── Output Results ─────────────────────────────────────────────────
            print(f"\n{'=' * 62}")
            print("  LEGAL DOCUMENT REVIEW  –  FINAL ORCHESTRATED REPORT")
            print(f"{'=' * 62}")
            print(f"\nResponse:\n{final_report}\n")

            # ── Risk Dashboard ─────────────────────────────────────────────────
            risk_score      = compliance_report.get("overall_risk_score", 0)
            critical_count  = compliance_report.get("critical_issues_count", 0)
            requires_review = compliance_report.get("requires_legal_review", False)
            comp_summary    = compliance_report.get("compliance_summary", "")

            print(f"{'─' * 42}")
            print(f"  Overall Risk Score    : {risk_score}/100")
            print(f"  Critical Issues       : {critical_count}")
            print(f"  Requires Legal Review : {'YES - ESCALATE' if requires_review else 'No'}")
            if comp_summary:
                print(f"\n  Compliance Summary:\n  {comp_summary}")
            print(f"{'─' * 42}\n")

    except Exception as ex:
        print(ex)


if __name__ == "__main__":
    asyncio.run(main())
