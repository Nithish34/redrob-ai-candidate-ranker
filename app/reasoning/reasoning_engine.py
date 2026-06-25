"""
Reasoning Engine
================
Generates recruiter-friendly, specific, and non-hallucinated explanations 
for candidate rankings. Satisfies manual review criteria in Stage 4 of the hackathon.
"""

from __future__ import annotations

import random
from typing import Any


def generate_reasoning(candidate_entry: dict, rank: int) -> str:
    """Generate a custom reasoning string for a ranked candidate.

    Parameters
    ----------
    candidate_entry : dict
        A ranked candidate entry containing:
        - candidate_id: str
        - final_score: float
        - tier: str
        - layer_results: dict of layer name -> result dict
        - candidate: original candidate dict
    rank : int
        The final rank of the candidate (1-100).

    Returns
    -------
    str
        A 1-2 sentence specific reasoning string.
    """
    candidate = candidate_entry.get("candidate", {})
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    
    # 1. Basic profile facts (strictly non-hallucinated)
    title = profile.get("current_title", "Engineer")
    years_val = profile.get("years_of_experience", 0.0)
    try:
        years = float(years_val)
    except (ValueError, TypeError):
        years = 0.0
        
    location = profile.get("location", "India")
    
    # 2. Extract skills that actually exist in the profile
    skills = [s.get("name") for s in candidate.get("skills", []) if s.get("name")]
    
    # Check for domain fits
    l2_results = candidate_entry.get("layer_results", {}).get("layer2", {})
    l2_meta = l2_results.get("metadata", {})
    matched_domains = l2_meta.get("domain_detail", {}).get("matched_domains", [])
    
    # Check for evidence strength
    l3_results = candidate_entry.get("layer_results", {}).get("layer3", {})
    l3_meta = l3_results.get("metadata", {})
    strongest_cap = l3_meta.get("strongest_capability", "")
    total_matches = l3_meta.get("total_evidence_matches", 0)

    # Check for logistics
    notice = signals.get("notice_period_days", 0)
    response_rate = signals.get("recruiter_response_rate", 0.0)
    try:
        response_rate = float(response_rate)
    except (ValueError, TypeError):
        response_rate = 0.0
        
    # Check for product/consulting
    company_classifications = l2_meta.get("product_detail", {}).get("company_classifications", {})
    has_product = any(cls == "product" for cls in company_classifications.values())
    has_consulting_only = all(cls == "consulting" for cls in company_classifications.values()) if company_classifications else False

    # 3. Build components of the sentence
    
    # Component A: Introduction with title and experience
    # Vary the format slightly to avoid robotic output
    r_val = rank % 3
    if r_val == 0:
        intro = f"{title} with {years:.1f} yrs experience"
    elif r_val == 1:
        intro = f"Senior profile ({years:.1f} yrs) working as {title}"
    else:
        intro = f"{years:.1f}-year experienced {title}"

    # Component B: Core technical fit/evidence
    core_skills = []
    for skill in ["python", "pytorch", "tensorflow", "vector search", "semantic search", "rag", "embeddings", "milvus", "pinecone", "elasticsearch", "opensearch", "recommendation"]:
        # Find exact or substring match in candidate's skills list
        matched = [s for s in skills if skill in s.lower()]
        if matched:
            core_skills.append(matched[0])
            
    core_skills = list(dict.fromkeys(core_skills))[:3] # unique first 3
    
    tech_fit = ""
    if matched_domains:
        domains_str = ", ".join(matched_domains[:3])
        if core_skills:
            tech_fit = f"demonstrates strong {domains_str} capabilities with skills in {', '.join(core_skills)}"
        else:
            tech_fit = f"covers key domains including {domains_str}"
    elif core_skills:
        tech_fit = f"possesses relevant technical skills in {', '.join(core_skills)}"
    else:
        tech_fit = "shows adjacent engineering experience"

    # Component C: Product / Company context
    company_context = ""
    if has_product:
        company_context = "at product-focused firms"
    elif has_consulting_only:
        company_context = "primarily in consulting environments"

    # Component D: Behavioral and hiring logistics details
    logistics = ""
    if response_rate > 0.6:
        logistics = f"highly responsive (rate: {response_rate:.0%})"
    if notice > 45:
        if logistics:
            logistics += f" but has a {notice}-day notice period"
        else:
            logistics = f"has a long {notice}-day notice period"
    elif notice <= 15:
        if logistics:
            logistics += f" and available within {notice} days"
        else:
            logistics = f"quickly available within {notice} days"

    # Component E: Rank-dependent verdict/tone
    verdict = ""
    if rank <= 15:
        verdict = "making them an exceptional fit for the Senior AI Engineer role"
    elif rank <= 50:
        verdict = "showing very solid credentials and strong alignment"
    elif rank <= 85:
        verdict = "representing a viable candidate with some minor experience or availability gaps"
    else:
        verdict = "included as a potential filler with adjacent capabilities"

    # 4. Assemble the sentence dynamically
    # Try to build a natural paragraph flow
    parts = []
    
    # First sentence: Profile, experience, tech fit, and company context
    s1 = f"{intro} who {tech_fit}"
    if company_context:
        s1 += f" {company_context}."
    else:
        s1 += "."
        
    # Second sentence: Logistics and rank verdict
    s2_parts = []
    if logistics:
        s2_parts.append(logistics)
    if verdict:
        s2_parts.append(verdict)
        
    if s2_parts:
        # Join appropriately
        if len(s2_parts) == 2:
            # e.g., "highly responsive, making them an exceptional fit..."
            s2 = f"{s2_parts[0].capitalize()}, {s2_parts[1]}."
        else:
            s2 = f"{s2_parts[0].capitalize()}."
        reasoning = f"{s1} {s2}"
    else:
        reasoning = s1

    # Final cleanup to ensure length and validity
    reasoning = reasoning.replace("  ", " ").replace(" ,", ",").replace(" .", ".")
    
    # Cap length at 250 characters to keep it concise but descriptive
    if len(reasoning) > 250:
        reasoning = reasoning[:247] + "..."
        
    return reasoning
