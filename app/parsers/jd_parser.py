"""
JD Parser — Converts the Redrob hackathon job description into a structured intent object.

Since the JD is fixed for this hackathon, we hardcode the parsed output based on
careful analysis of the job_description.docx. In a production system, this would
use NLP to extract requirements dynamically.
"""


def parse_jd() -> dict:
    """Return the structured ParsedJD for the Senior AI Engineer role.

    Returns:
        A dict with structured JD requirements.
    """
    return {
        "title": "Senior AI Engineer",
        "company": "Redrob AI",
        "experience_range": {"min": 5, "max": 9, "ideal": 7},
        "required_domains": [
            "retrieval",
            "ranking",
            "recommendation",
            "search",
            "evaluation",
            "production_ml",
        ],
        "must_have_capabilities": [
            "embeddings-based retrieval systems",
            "vector databases or hybrid search",
            "python",
            "evaluation frameworks for ranking systems",
        ],
        "nice_to_have_capabilities": [
            "llm fine-tuning",
            "learning-to-rank models",
            "hr-tech or recruiting tech",
            "distributed systems",
            "open-source contributions",
        ],
        "disqualifiers": [
            "pure research without production deployment",
            "only recent llm api wrapper experience",
            "no production code in 18 months",
            "only consulting company career",
            "primarily cv/speech/robotics without nlp",
        ],
        "behavior_preferences": {
            "preferred_work_mode": "hybrid",
            "preferred_locations": [
                "pune", "noida", "hyderabad", "mumbai",
                "delhi", "ncr", "bangalore", "bengaluru",
            ],
            "preferred_country": "india",
            "max_notice_days": 30,
            "notice_buyout_days": 30,
        },
        "company_preferences": {
            "prefers_product_companies": True,
            "penalizes_only_consulting": True,
            "values_startup_experience": True,
        },
        "role_keywords": [
            "ai engineer", "ml engineer", "machine learning engineer",
            "data scientist", "research engineer", "search engineer",
            "nlp engineer", "retrieval engineer", "ranking engineer",
            "platform engineer", "backend engineer", "software engineer",
        ],
        "anti_role_keywords": [
            "marketing manager", "hr manager", "sales executive",
            "content writer", "graphic designer", "accountant",
            "civil engineer", "mechanical engineer", "operations manager",
            "customer support", "business analyst", "project manager",
            "devops engineer", "frontend developer", "ui designer",
        ],
    }
