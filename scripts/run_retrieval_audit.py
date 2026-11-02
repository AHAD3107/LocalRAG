"""100-Query Baseline Retrieval Audit Script for LocalRAG.

Audits:
- Query embedding generation via nomic-embed-text
- Persistent NumPy vector search over actual data/index
- Cosine similarity thresholding at 0.55, top-k=4
- Ground truth verification across 35 indexed chunks of machine_learning_algorithms.pdf
- Standard IR metrics: Precision@K, Recall@K, MRR, Rejection rates, Latencies
- Systematic failure classification
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from config import settings
from src.embeddings.ollama import OllamaClient
from src.generation.contextualize import contextualize_query
from src.retrieval.search import search
from src.storage.vector_index import VectorIndex

DOC_ID = "3e3db83d0094b529a233181a9ec9adeeb2bc0fee18650161ecbb43202a6ae582"


def cid(page: int, chunk: int) -> str:
    return f"{DOC_ID}_p{page}_c{chunk}"


def build_benchmark_dataset() -> list[dict]:
    queries: list[dict] = []

    # =========================================================================
    # 1. EXACT-MATCH (20 queries)
    # =========================================================================
    exact_matches = [
        ("q001", "What is the target system name described in the psychometrics specification?",
         "KnowledgeTrace MCQ", [cid(1, 0)]),
        ("q002", "What is the target inference speed for KnowledgeTrace MCQ?",
         "< 15ms (NumPy Vectorized)", [cid(1, 0)]),
        ("q003", "Which graph validation library and structure is used in KnowledgeTrace MCQ?",
         "NetworkX Poset DAG", [cid(1, 0)]),
        ("q004", "What is the probability of a student guessing correctly in a 4-choice question without knowing the answer?",
         "25% chance of guessing correctly", [cid(1, 1)]),
        ("q005", "What are the two possible mental states modeled in Bayesian Knowledge Tracing?",
         "0 = Unmastered and 1 = Mastered", [cid(3, 12), cid(3, 13)]),
        ("q006", "What is the default value for Prior Knowledge P(L0) in BKT?",
         "0.20 (20%)", [cid(3, 13)]),
        ("q007", "What is the default value for Learning Transition P(T) in BKT?",
         "0.15 (15%)", [cid(3, 13)]),
        ("q008", "What is the default value for Careless Slip P(S) in BKT?",
         "0.10 (10%)", [cid(3, 13), cid(3, 14)]),
        ("q009", "What is the default value for Lucky Guess P(G) in BKT?",
         "0.25 (25%)", [cid(3, 14)]),
        ("q010", "What is the latent learning transition formula used in Step 2 of BKT?",
         "P(Lt+1) = P(Lt | obs) + [ 1 − P(Lt | obs) ] × P(T)", [cid(3, 15)]),
        ("q011", "Who formalized the DINA model and in which year?",
         "Junker & Sijtsma (2001)", [cid(4, 18)]),
        ("q012", "What does the binary Q-matrix represent in DINA?",
         "A binary table linking each question to the skills it requires: qjk = 1 if question j requires concept k, else 0", [cid(4, 19)]),
        ("q013", "If a student possesses all required skills in DINA, what is their probability of answering correctly?",
         "P(Correct) = 1 − Slip = 1 − 0.10 = 0.90 (90%)", [cid(4, 20)]),
        ("q014", "If a student misses any required skill in DINA, what is their probability of answering correctly?",
         "P(Correct) = Guess = 0.25 (25%)", [cid(4, 20)]),
        ("q015", "How many possible skill combinations exist if a curriculum has 30 skills?",
         "2^30 = 1,073,741,824 possible skill combinations", [cid(4, 20)]),
        ("q016", "What is the maximum skill block size K for exact Bayesian posteriors before falling back?",
         "K <= 12 (max 4,096 states)", [cid(4, 21)]),
        ("q017", "What is the quadratic entropy confidence score equation used in dina_service.py?",
         "Confidence = 1 − 4 × P × (1 − P)", [cid(4, 22), cid(4, 23)]),
        ("q018", "What confidence score is produced when probability P equals 0.50?",
         "Confidence = 0.00 (0%)", [cid(4, 23)]),
        ("q019", "Which threshold flags a skill as a 'Likely Missing Prerequisite' in get_question_why?",
         "mastery P < 0.50", [cid(2, 9), cid(4, 24)]),
        ("q020", "Which educational benchmark dataset originated from Worcester Polytechnic Institute (WPI)?",
         "ASSISTments 2009-2010", [cid(6, 32)]),
    ]
    for qid, qtext, ans, cids in exact_matches:
        queries.append({
            "question_id": qid,
            "question": qtext,
            "category": "Exact-match",
            "expected_answer": ans,
            "relevant_doc_id": DOC_ID,
            "relevant_chunk_ids": cids,
            "should_reject": False,
        })

    # =========================================================================
    # 2. PARAPHRASED (20 queries)
    # =========================================================================
    paraphrased = [
        ("q021", "What software tool ensures prerequisite relationships do not form impossible loops?",
         "NetworkX is_directed_acyclic_graph() evaluates candidate edges to prevent cycles", [cid(1, 4), cid(3, 12)]),
        ("q022", "Why is basic arithmetic grading inadequate for measuring true student competency?",
         "It fails to capture guessing, careless slips, specific missing sub-skills, or mastery changes over time", [cid(1, 1), cid(1, 2)]),
        ("q023", "How does the system treat a learner's actual comprehension level in psychological modeling?",
         "As a hidden mental state statistically filtered between 0.0 and 1.0", [cid(1, 3)]),
        ("q024", "Which background worker mechanism prevents the user interface from freezing during Bayesian calculations?",
         "FastAPI BackgroundTasks triggers run_mastery_and_diagnosis() asynchronously", [cid(2, 7), cid(2, 8)]),
        ("q025", "Where are raw student submissions initially committed before AI processing?",
         "routers/student.py submit_response() logs raw responses into PostgreSQL synchronously", [cid(2, 7)]),
        ("q026", "What technique discovers candidate prerequisite pairs between curriculum topics?",
         "infer_llm_concept_edges() sends curriculum concepts to Groq LLaMA to propose directed pairs with confidence", [cid(3, 11)]),
        ("q027", "How does a student's mastery probability change after making an error in BKT?",
         "Posterior probability drops, then Step 2 adds latent transition probability P(T)", [cid(3, 15), cid(3, 16)]),
        ("q028", "Can a student unlearn a concept once it has been mastered according to the BKT transition rules?",
         "No, once a student learns, they cannot unlearn", [cid(3, 15)]),
        ("q029", "What was the final updated mastery probability after four consecutive practice attempts in the example?",
         "0.8855 (> 88%), confirming high mastery", [cid(3, 17)]),
        ("q030", "What model handles questions that test several distinct subject areas at the same time?",
         "Layer 3 Multi-Attribute DINA Model using Q-Matrix logic", [cid(1, 5), cid(4, 18)]),
        ("q031", "What happens in DINA if a pupil lacks just a single prerequisite skill for a question?",
         "The conjunctive AND rule yields xi=0, dropping the student into the guess category (25% chance)", [cid(4, 19), cid(4, 20)]),
        ("q032", "How does KnowledgeTrace MCQ circumvent exponential computational explosion for large skill sets?",
         "BFS Connected-Block Partitioning isolates connected skill blocks so only relevant skills are computed", [cid(4, 21)]),
        ("q033", "Under what conditions does the DINA service switch to a closed-form O(1) single-skill fallback?",
         "When a connected skill block exceeds 12 skills", [cid(4, 22)]),
        ("q034", "What are the three confidence categories used to rate diagnostic certainty?",
         "LOW (< 0.40), MEDIUM (0.40 to 0.75), and HIGH (>= 0.75)", [cid(4, 23)]),
        ("q035", "What explanation does the diagnostic engine give instructors when a question is answered incorrectly?",
         "Flags required concepts with mastery P < 0.50 as a 'Likely Missing Prerequisite'", [cid(2, 9), cid(4, 24)]),
        ("q036", "What role does Layer 1 play relative to Layer 2 and Layer 3?",
         "Layer 1 builds the map by establishing topological prerequisite directions across curriculum topics", [cid(5, 26)]),
        ("q037", "Which database table stores the student mastery probabilities computed by BKT?",
         "student_concept_mastery", [cid(2, 8), cid(6, 30)]),
        ("q038", "Which database table records adjusted mastery and epistemic certainty from DINA?",
         "cognitive_diagnosis_profile", [cid(2, 9), cid(6, 31)]),
        ("q039", "Which large-scale dataset from Taiwan was used to validate concept prerequisite DAG sequencing?",
         "Junyi Academy dataset with 72,000+ students and 835 concept DAG", [cid(6, 32), cid(6, 33)]),
        ("q040", "Which psychometric benchmark established the 8 skill attributes for fraction subtraction?",
         "Tatsuoka Fraction Subtraction (1990 Rule Space)", [cid(6, 33)]),
    ]
    for qid, qtext, ans, cids in paraphrased:
        queries.append({
            "question_id": qid,
            "question": qtext,
            "category": "Paraphrased",
            "expected_answer": ans,
            "relevant_doc_id": DOC_ID,
            "relevant_chunk_ids": cids,
            "should_reject": False,
        })

    # =========================================================================
    # 3. MULTI-CHUNK (15 queries)
    # =========================================================================
    multi_chunk = [
        ("q041", "How do Layer 1, Layer 2, and Layer 3 collaborate to guide student learning?",
         "Layer 1 builds the prerequisite map, Layer 2 tracks sequential single-skill momentum via BKT, and Layer 3 performs multi-skill DINA diagnosis",
         [cid(1, 4), cid(5, 26), cid(5, 27)]),
        ("q042", "What are the core differences between BKT and DINA in handling single versus multi-skill questions?",
         "BKT tracks sequential 0/1 mastery on single concepts; DINA uses binary vectors and a Q-matrix conjunctive AND rule for composite multi-skill problems",
         [cid(1, 4), cid(4, 18), cid(5, 27)]),
        ("q043", "What are the slip and guess parameter values used in both BKT and DINA?",
         "Both models share P(Slip) = 0.10 (10%) and P(Guess) = 0.25 (25%)",
         [cid(3, 14), cid(4, 20)]),
        ("q044", "Explain the complete journey of an MCQ response from submission to database persistence across services.",
         "React sends POST to FastAPI submit_response, logs to PostgreSQL, BackgroundTasks triggers BKT in mastery_service and DINA in dina_service, persisting to student_concept_mastery and cognitive_diagnosis_profile",
         [cid(2, 6), cid(2, 7), cid(2, 8), cid(2, 9)]),
        ("q045", "How does the system ensure fast execution at both the API layer and the mathematical model layer?",
         "FastAPI uses non-blocking asynchronous background workers for immediate API responses, and DINA uses BFS partitioning with vectorized NumPy Log-Sum-Exp under 10ms",
         [cid(1, 0), cid(2, 7), cid(4, 21)]),
        ("q046", "What role does Groq LLaMA play in the concept graph, and how does NetworkX validate its output?",
         "Groq LLaMA infers directed concept pairs with confidence, and NetworkX nx.is_directed_acyclic_graph(G) rejects any edge creating a cycle",
         [cid(3, 11), cid(3, 12), cid(6, 29)]),
        ("q047", "Compare the initial BKT prior probability with the probability after a first correct response in the student attempt example.",
         "Initial prior starts at P(L0)=0.2000; after attempt 1 correct, Bayes posterior reaches 0.4737 and updated mastery reaches 0.5526",
         [cid(3, 13), cid(3, 16)]),
        ("q048", "How does the quadratic entropy confidence equation behave at maximum uncertainty versus strong mastery?",
         "At P=0.50 (maximum uncertainty) confidence is 0.00; at P=0.85 (strong mastery) confidence rises to 0.49",
         [cid(4, 22), cid(4, 23)]),
        ("q049", "Which Python files implement the core algorithms for concept graphs, BKT estimation, and DINA diagnosis?",
         "concept_graph.py (DAG check), mastery_service.py (compute_bkt_sequence), and dina_service.py (_compute_block_exact/update_diagnosis)",
         [cid(1, 4), cid(6, 29), cid(6, 30), cid(6, 31)]),
        ("q050", "What database tables store the outputs of API ingestion, BKT estimation, and DINA diagnosis?",
         "responses stores raw submissions, student_concept_mastery stores BKT probabilities, and cognitive_diagnosis_profile stores DINA adjusted mastery and confidence",
         [cid(6, 30), cid(6, 31)]),
        ("q051", "How does get_question_why utilize DINA mastery values to flag prerequisite gaps?",
         "get_question_why retrieves skills required by the question and flags any skill with DINA mastery P < 0.50 as a Likely Missing Prerequisite",
         [cid(2, 9), cid(4, 24), cid(6, 31)]),
        ("q052", "What benchmark datasets validate BKT sequence modeling and DINA rule space models respectively?",
         "ASSISTments 2009-2010 validates BKT sequence modeling, and Tatsuoka Fraction Subtraction validates Q-matrix DINA models",
         [cid(6, 32), cid(6, 33)]),
        ("q053", "Why does a circular loop break adaptive learning, and what algorithm checks for directed acyclic structure?",
         "Circular prerequisites (A->B->C->A) create impossible deadlocks; nx.is_directed_acyclic_graph() evaluates candidate edges",
         [cid(1, 4), cid(3, 12)]),
        ("q054", "How does the system prevent UI lag during student submission while calculating Bayesian updates in the background?",
         "submit_response immediately returns success after writing to PostgreSQL, and BackgroundTasks delegates compute_mastery_and_diagnosis to background execution",
         [cid(2, 7), cid(2, 8)]),
        ("q055", "Describe how BKT observation update and latent transition update operate sequentially.",
         "Step 1 updates P(Lt) using Bayes rule based on observed correct/incorrect answer; Step 2 applies latent learning transition P(T) to produce P(Lt+1)",
         [cid(3, 14), cid(3, 15)]),
    ]
    for qid, qtext, ans, cids in multi_chunk:
        queries.append({
            "question_id": qid,
            "question": qtext,
            "category": "Multi-chunk",
            "expected_answer": ans,
            "relevant_doc_id": DOC_ID,
            "relevant_chunk_ids": cids,
            "should_reject": False,
        })

    # =========================================================================
    # 4. UNSUPPORTED (15 queries)
    # =========================================================================
    unsupported = [
        ("q056", "What transformer model architecture is used to generate BERT embeddings for student questions?",
         "The document specifies Groq LLaMA for concept edge discovery; it never uses or describes BERT embeddings."),
        ("q057", "What is the SQL query to calculate a student's grade point average (GPA) in PostgreSQL?",
         "The specification covers psychometric mastery tables (responses, student_concept_mastery); GPA calculations are completely absent."),
        ("q058", "Which Docker Compose port binds the external Redis caching cluster to host port 6379?",
         "The document mentions Redis conceptually in notes but provides no Docker Compose port configuration."),
        ("q059", "How does the webcam proctoring system detect student eye movements during MCQ exams?",
         "No webcam or automated video proctoring features exist in the system specification."),
        ("q060", "What Stripe API webhook endpoint processes credit card payments for exam fees?",
         "No payment gateway, billing logic, or Stripe integrations exist in the document."),
        ("q061", "What is the SMTP server hostname and port for sending student password reset emails?",
         "No SMTP mail server configuration or password recovery services are documented."),
        ("q062", "What fine-tuning hyperparameter settings were used for OpenAI GPT-4o?",
         "The system uses Groq LLaMA for concept edges; OpenAI GPT-4o fine-tuning is not documented."),
        ("q063", "How does the system calculate Pearson correlation between student exam scores and family income?",
         "The document contains no socioeconomic demographic variables or Pearson correlation formulas."),
        ("q064", "Which Kubernetes Helm chart deploys the Celery worker pods on AWS EKS?",
         "No Kubernetes, Helm charts, or AWS EKS infrastructure deployment specs are present."),
        ("q065", "What is the current outdoor weather temperature in London?",
         "Weather information is completely unrelated to the psychometrics specification."),
        ("q066", "How many venture capital funding rounds has KnowledgeTrace MCQ raised?",
         "No corporate financial or venture capital information exists in the technical guide."),
        ("q067", "What is the recommended dosage for pediatric ibuprofen?",
         "Medical pharmacology queries are completely outside the scope of this psychometric guide."),
        ("q068", "Who was the lead architect who designed the Eiffel Tower in Paris?",
         "Historical architecture trivia is outside the scope of this document."),
        ("q069", "What is the maximum upload file size for video recordings in the student dashboard?",
         "The system only handles text MCQ answer options, not video recording uploads."),
        ("q070", "What is the exchange rate between the Japanese Yen and the US Dollar today?",
         "Foreign currency exchange rates are completely outside the scope of this document."),
    ]
    for qid, qtext, reason in unsupported:
        queries.append({
            "question_id": qid,
            "question": qtext,
            "category": "Unsupported",
            "expected_answer": "REJECT: " + reason,
            "relevant_doc_id": None,
            "relevant_chunk_ids": [],
            "should_reject": True,
            "rejection_reason": reason,
        })

    # =========================================================================
    # 5. HARD NEGATIVES (15 queries)
    # =========================================================================
    hard_negatives = [
        ("q071", "What is the NetworkX betweenness centrality algorithm used to rank concept importance in the Poset DAG?",
         "Shares NetworkX, concept, and Poset DAG with chunks 0 and 12, but betweenness centrality is never calculated.",
         [cid(1, 0), cid(3, 12)]),
        ("q072", "What is the discount factor gamma used in the Q-Matrix Q-learning reinforcement algorithm?",
         "Shares Q-Matrix with chunk 19, but the Q-Matrix is a binary cognitive mapping table, not reinforcement Q-learning.",
         [cid(4, 19)]),
        ("q073", "What is the learning rate of the Adam optimizer used to fit the BKT parameters in PyBKT?",
         "Shares BKT and parameters with chunk 13, but BKT parameters are fixed psychometric defaults, not fitted via Adam.",
         [cid(3, 13)]),
        ("q074", "What is the Dirichlet prior distribution parameter alpha applied to the Hidden Markov Model in BKT?",
         "Shares BKT, HMM, and Prior with chunk 13, but BKT uses scalar P(L0)=0.20 rather than a Dirichlet prior distribution.",
         [cid(3, 13)]),
        ("q075", "What is the maximum number of worker threads configured in uvicorn for the FastAPI router?",
         "Shares FastAPI and routers/student.py with chunks 7 and 30, but uvicorn thread counts are not specified.",
         [cid(2, 7), cid(6, 30)]),
        ("q076", "How does Kahn's algorithm compute the critical path duration in the NetworkX graph?",
         "Shares Kahn's algorithm, NetworkX, and DAG with chunks 4 and 12, but Kahn's algorithm is used solely for cycle detection.",
         [cid(1, 4), cid(3, 12)]),
        ("q077", "What is the maximum Celery broker retry delay when publishing to the Redis cluster?",
         "Shares Celery, Redis, and background worker with chunks 7 and 8, but broker retry delay parameters are absent.",
         [cid(2, 7), cid(2, 8)]),
        ("q078", "What is the Shannon entropy formula in base 2 used to calculate cognitive uncertainty in dina_service.py?",
         "Shares entropy and dina_service.py with chunk 23, but the engine uses Quadratic Entropy 1-4P(1-P), not base-2 Shannon entropy.",
         [cid(4, 22), cid(4, 23)]),
        ("q079", "What is the Item Characteristic Curve slope parameter (a-parameter) in the 3PL IRT model in KnowledgeTrace MCQ?",
         "Shares psychometric terminology with chunks 0, 3, and 13, but the system uses BKT and DINA, not 3PL IRT models.",
         [cid(1, 0), cid(1, 3), cid(3, 13)]),
        ("q080", "What is the Gibbs sampling Markov Chain Monte Carlo burn-in iteration count for DINA blocks?",
         "Shares DINA, block, and sampling with chunk 21, but the system uses exact vectorized Log-Sum-Exp or O(1) single-skill fallback.",
         [cid(4, 21), cid(4, 22)]),
        ("q081", "What is the ASSISTments dataset test set F1-score for predicting student standardized state exam performance?",
         "Shares ASSISTments and benchmark with chunk 32, but the document only reports an AUC ~0.73 for BKT sequence evaluation.",
         [cid(6, 32)]),
        ("q082", "What is the ROC-AUC score achieved by the DINA model on the Junyi Academy dataset?",
         "Shares Junyi Academy and DINA with chunk 32, but Junyi validates concept prerequisite DAGs, not DINA ROC-AUC.",
         [cid(6, 32), cid(6, 33)]),
        ("q083", "How many questions are in the Tatsuoka Fraction Subtraction test for college mathematics entrance exams?",
         "Shares Tatsuoka Fraction Subtraction with chunk 33, but the dataset has 20 items for fraction subtraction, not college entrance.",
         [cid(6, 33)]),
        ("q084", "What is the WebSocket heartbeat ping interval configured for real-time teacher alerts?",
         "Shares teacher, alerts, and UI with chunks 24 and 31, but WebSocket ping intervals are not in the document.",
         [cid(4, 24), cid(6, 31)]),
        ("q085", "What is the PostgreSQL connection pool maximum overflow setting in database.py?",
         "Shares PostgreSQL and database with chunks 7 and 29, but connection pool overflow settings are not documented.",
         [cid(2, 7), cid(6, 29)]),
    ]
    for qid, qtext, trap_reason, misleading_cids in hard_negatives:
        queries.append({
            "question_id": qid,
            "question": qtext,
            "category": "Hard Negative",
            "expected_answer": "REJECT: " + trap_reason,
            "relevant_doc_id": None,
            "relevant_chunk_ids": [],
            "should_reject": True,
            "trap_reason": trap_reason,
            "misleading_chunk_ids": misleading_cids,
        })

    # =========================================================================
    # 6. DOCUMENT-SPECIFIC (10 queries)
    # =========================================================================
    doc_specific = [
        ("q086", "In machine_learning_algorithms.pdf, what is the default prior knowledge probability P(L0)?",
         "0.20 (20%)", DOC_ID, [cid(3, 13)], False),
        ("q087", "In machine_learning_algorithms.pdf, which file implements the function infer_llm_concept_edges?",
         "services/concept_graph.py", DOC_ID, [cid(3, 11), cid(6, 29)], False),
        ("q088", "In machine_learning_algorithms.pdf, what formula defines the conjunctive AND rule in DINA?",
         "xi_j(alpha) = prod alpha_k^{q_jk}", DOC_ID, [cid(4, 19)], False),
        ("q089", "In machine_learning_algorithms.pdf, what is the confidence score when student mastery P is 0.85?",
         "Confidence = 1 - 4(0.85 * 0.15) = 0.49 (49%)", DOC_ID, [cid(4, 23)], False),
        ("q090", "In machine_learning_algorithms.pdf, which benchmark dataset came from Worcester Polytechnic Institute?",
         "ASSISTments 2009-2010", DOC_ID, [cid(6, 32)], False),
        ("q091", "In machine_learning_algorithms.pdf, what is the slip probability P(S) in BKT?",
         "0.10 (10%)", DOC_ID, [cid(3, 13), cid(3, 14)], False),
        ("q092", "In machine_learning_algorithms.pdf, what function checks whether the prerequisite graph is acyclic?",
         "is_directed_acyclic_graph() / nx.is_directed_acyclic_graph(G)", DOC_ID, [cid(1, 4), cid(3, 12), cid(6, 29)], False),
        ("q093", "In machine_learning_algorithms.pdf, which background task triggers asynchronous mastery updates?",
         "run_mastery_and_diagnosis()", DOC_ID, [cid(2, 7), cid(2, 8)], False),
        ("q094", "In foreign_system_architecture.pdf, what is the reported latency target?",
         "REJECT: Document foreign_system_architecture.pdf does not exist in the index.", "doc_foreign_spec_001", [], True),
        ("q095", "In financial_audit_2026.pdf, what is the net revenue figure?",
         "REJECT: Document financial_audit_2026.pdf does not exist in the index.", "doc_foreign_spec_002", [], True),
    ]
    for qid, qtext, ans, r_doc_id, cids, reject in doc_specific:
        queries.append({
            "question_id": qid,
            "question": qtext,
            "category": "Document-specific",
            "expected_answer": ans,
            "relevant_doc_id": r_doc_id,
            "relevant_chunk_ids": cids,
            "should_reject": reject,
        })

    # =========================================================================
    # 7. CONVERSATIONAL FOLLOW-UPS (5 queries)
    # =========================================================================
    conversational = [
        ("q096",
         [["What is KnowledgeTrace MCQ?", "KnowledgeTrace MCQ is an educational psychometrics and ML test system."]],
         "What is its target inference speed?",
         "What is the target inference speed of KnowledgeTrace MCQ?",
         "< 15ms (NumPy Vectorized)",
         [cid(1, 0)]),
        ("q097",
         [["Which models are used in Layer 2 and Layer 3?", "Layer 2 uses Bayesian Knowledge Tracing and Layer 3 uses the DINA model."]],
         "What are their slip and guess probability values?",
         "What are the slip and guess probability values for BKT and DINA in KnowledgeTrace MCQ?",
         "Both models share P(Slip) = 0.10 (10%) and P(Guess) = 0.25 (25%)",
         [cid(3, 14), cid(4, 20)]),
        ("q098",
         [["How does the system prevent prerequisite cycles?", "It evaluates prerequisite edges using networkx to ensure the graph is a DAG."]],
         "Which function performs this directed acyclic graph check?",
         "Which function performs the directed acyclic graph check in NetworkX for KnowledgeTrace MCQ?",
         "is_directed_acyclic_graph() / nx.is_directed_acyclic_graph(G)",
         [cid(1, 4), cid(3, 12)]),
        ("q099",
         [["How is student confidence calculated?", "KnowledgeTrace MCQ uses a quadratic entropy formula to compute epistemic confidence."]],
         "What is the exact equation for it?",
         "What is the exact equation for quadratic entropy confidence in KnowledgeTrace MCQ?",
         "Confidence = 1 - 4 * P * (1 - P)",
         [cid(4, 23)]),
        ("q100",
         [["What happens when a student misses an exam question?", "The diagnostic engine searches for missing skills."]],
         "Which function flags a skill as a Likely Missing Prerequisite?",
         "Which function flags a skill as a Likely Missing Prerequisite in KnowledgeTrace MCQ?",
         "get_question_why() flags skills with mastery P < 0.50",
         [cid(2, 9), cid(4, 24)]),
    ]
    for qid, hist, qtext, contextualized, ans, cids in conversational:
        queries.append({
            "question_id": qid,
            "question": qtext,
            "history": hist,
            "expected_contextualized_query": contextualized,
            "category": "Conversational follow-ups",
            "expected_answer": ans,
            "relevant_doc_id": DOC_ID,
            "relevant_chunk_ids": cids,
            "should_reject": False,
        })

    return queries


def run_benchmark_audit():
    benchmark = build_benchmark_dataset()
    print(f"Total benchmark questions defined: {len(benchmark)}")

    # Load persistent VectorIndex
    index_path = PROJECT_ROOT / "data" / "index"
    index = VectorIndex.load(index_path, settings.embedding_model, settings.chunk_size, settings.chunk_overlap)
    indexed_chunk_ids = {c.chunk_id for c in index.chunks}
    print(f"Loaded vector index with {len(index.chunks)} chunks from {index_path}")

    # Validate that all referenced chunk IDs exist in the index
    missing_cids = []
    for q in benchmark:
        for cid_val in q["relevant_chunk_ids"]:
            if cid_val not in indexed_chunk_ids:
                missing_cids.append((q["question_id"], cid_val))

    if missing_cids:
        print(f"ERROR: {len(missing_cids)} referenced chunk IDs not found in index:")
        for qid, missing in missing_cids:
            print(f"  {qid}: {missing}")
        raise ValueError("Invalid chunk IDs in benchmark dataset!")
    print("Verified: 100% of referenced chunk IDs exist in the indexed corpus.")

    # Save benchmark dataset to file
    benchmark_file = PROJECT_ROOT / "data" / "evaluation" / "retrieval_benchmark_100.json"
    benchmark_file.parent.mkdir(parents=True, exist_ok=True)
    benchmark_file.write_text(json.dumps(benchmark, indent=2), encoding="utf-8")
    print(f"Saved benchmark to {benchmark_file}")

    # Initialize Ollama Client
    client = OllamaClient(settings.ollama_base_url, timeout=60.0)
    if not client.is_available():
        raise RuntimeError(f"Ollama is not available on {settings.ollama_base_url}")
    print(f"Ollama connected on {settings.ollama_base_url} (model: {settings.embedding_model})")

    threshold = settings.similarity_threshold  # 0.55
    top_k = settings.top_k                    # 4
    k_values = [1, 2, 4]

    raw_eval_results = []
    category_metrics = {}

    latencies_embed_ms = []
    latencies_search_ms = []
    latencies_total_ms = []

    # Category trackers
    categories = [
        "Exact-match",
        "Paraphrased",
        "Multi-chunk",
        "Unsupported",
        "Hard Negative",
        "Document-specific",
        "Conversational follow-ups",
    ]
    for cat in categories:
        category_metrics[cat] = {
            "total_queries": 0,
            "relevant_queries": 0,
            "irrelevant_queries": 0,
            "reciprocal_ranks": [],
            "precisions_at_k": {k: [] for k in k_values},
            "recalls_at_k": {k: [] for k in k_values},
            "correctly_rejected": 0,
            "false_acceptances": 0,
            "latencies_embed_ms": [],
            "latencies_search_ms": [],
            "latencies_total_ms": [],
        }

    failures = []

    print("\nExecuting baseline retrieval evaluation on 100 benchmark queries...\n")

    for item in benchmark:
        qid = item["question_id"]
        category = item["category"]
        question = item["question"]
        expected_cids = set(item["relevant_chunk_ids"])
        should_reject = item["should_reject"]
        is_relevant = len(expected_cids) > 0 and not should_reject

        # Query preparation: contextualization for conversational queries
        search_query = question
        contextualized_text = None
        if category == "Conversational follow-ups" and item.get("history"):
            t_ctx_0 = time.perf_counter()
            search_query = contextualize_query(question, item["history"], client, settings.chat_model_default)
            t_ctx_1 = time.perf_counter()
            contextualized_text = search_query

        # Document scoping parameter
        doc_scope = None
        doc_error = None
        if category == "Document-specific":
            doc_scope = item.get("relevant_doc_id")

        # Embedding latency benchmark
        t0 = time.perf_counter()
        q_vec = client.embed(search_query, settings.embedding_model)
        t1 = time.perf_counter()
        embed_ms = (t1 - t0) * 1000.0

        # Vector search latency benchmark
        t2 = time.perf_counter()
        retrieved_items = []
        try:
            retrieved_items = search(
                index,
                q_vec,
                top_k=top_k,
                threshold=threshold,
                doc_id=doc_scope,
            )
        except ValueError as exc:
            doc_error = str(exc)
        t3 = time.perf_counter()
        search_ms = (t3 - t2) * 1000.0
        total_ms = embed_ms + search_ms

        latencies_embed_ms.append(embed_ms)
        latencies_search_ms.append(search_ms)
        latencies_total_ms.append(total_ms)

        retrieved_cids = [chunk.chunk_id for chunk, score in retrieved_items]
        similarity_scores = {chunk.chunk_id: round(score, 4) for chunk, score in retrieved_items}

        # Also get top 10 unconstrained matches to observe full score distribution
        unconstrained_matches = []
        try:
            unconstrained_matches = search(index, q_vec, top_k=10, threshold=0.0, doc_id=doc_scope)
        except ValueError:
            pass
        all_top_scores = {chunk.chunk_id: round(score, 4) for chunk, score in unconstrained_matches}

        actually_rejected = len(retrieved_cids) == 0 or (doc_error is not None)

        # Compute metrics
        rr = 0.0
        p_at_k = {}
        r_at_k = {}

        cat_stat = category_metrics[category]
        cat_stat["total_queries"] += 1
        cat_stat["latencies_embed_ms"].append(embed_ms)
        cat_stat["latencies_search_ms"].append(search_ms)
        cat_stat["latencies_total_ms"].append(total_ms)

        is_failed = False
        failure_type = None
        failure_explanation = None

        if is_relevant:
            cat_stat["relevant_queries"] += 1
            # Reciprocal rank across retrieved chunks above threshold
            for rank_idx, r_cid in enumerate(retrieved_cids, start=1):
                if r_cid in expected_cids:
                    rr = 1.0 / rank_idx
                    break
            cat_stat["reciprocal_ranks"].append(rr)

            for k in k_values:
                top_k_set = set(retrieved_cids[:k])
                hits = len(top_k_set & expected_cids)
                p_k = hits / k
                r_k = hits / len(expected_cids)
                p_at_k[k] = round(p_k, 4)
                r_at_k[k] = round(r_k, 4)
                cat_stat["precisions_at_k"][k].append(p_k)
                cat_stat["recalls_at_k"][k].append(r_k)

            # Failure detection for relevant queries:
            if rr == 0.0:
                is_failed = True
                exp_unconstrained = [(cid_val, all_top_scores.get(cid_val, 0.0)) for cid_val in expected_cids]
                max_exp_score = max((score for _, score in exp_unconstrained), default=0.0)

                # Check contextualization failure first
                if category == "Conversational follow-ups":
                    orig_terms = set(question.lower().split()) - {"what", "is", "its", "their", "the", "for", "in", "it"}
                    rewritten_terms = set(search_query.lower().split())
                    if not (orig_terms & rewritten_terms):
                        failure_type = "CONTEXTUALIZATION_FAILURE"
                        failure_explanation = (
                            f"LLM query contextualization severely distorted the original query intent: "
                            f"Original '{question}' was rewritten into '{search_query}', dropping key terms."
                        )
                
                if not failure_type:
                    # Check if evidence was split across chunks (Chunking failure)
                    if any("c22" in cid_val or "c23" in cid_val or "c13" in cid_val or "c14" in cid_val for cid_val in expected_cids) and max_exp_score >= 0.50:
                        # Inspect if adjacent chunk was retrieved instead
                        adjacent_retrieved = any(
                            (cid_val.replace("c22", "c23") in retrieved_cids) or 
                            (cid_val.replace("c23", "c22") in retrieved_cids) or
                            (cid_val.replace("c13", "c14") in retrieved_cids) or
                            (cid_val.replace("c14", "c13") in retrieved_cids)
                            for cid_val in expected_cids
                        )
                        if adjacent_retrieved:
                            failure_type = "CHUNKING_FAILURE"
                            failure_explanation = (
                                f"Target evidence was fragmented or cut across fixed 500-char chunk boundaries. "
                                f"The system retrieved adjacent chunk with score {max(similarity_scores.values()):.4f}, "
                                f"but the specific target answer was cut into an adjacent slice."
                            )

                if not failure_type:
                    if max_exp_score < threshold and max_exp_score > 0.0:
                        failure_type = "THRESHOLD_FAILURE"
                        failure_explanation = (
                            f"Expected chunk appeared in candidate list but top expected score {max_exp_score:.4f} "
                            f"fell below configured similarity threshold {threshold:.2f}."
                        )
                    elif max_exp_score == 0.0:
                        failure_type = "EMBEDDING_FAILURE"
                        failure_explanation = "Expected chunk was not within top-10 unconstrained semantic search candidates."
                    else:
                        failure_type = "RANKING_FAILURE"
                        failure_explanation = (
                            f"Expected chunk had score {max_exp_score:.4f} >= threshold, but was pushed outside top-{top_k} "
                            f"by other higher-scoring non-relevant chunks."
                        )
            elif r_at_k[4] < 1.0 and len(expected_cids) > 1:
                # Partial multi-chunk failure
                is_failed = True
                failure_type = "RANKING_FAILURE"
                failure_explanation = (
                    f"Only {int(r_at_k[4] * len(expected_cids))}/{len(expected_cids)} expected chunks were retrieved "
                    f"within top-{top_k} above threshold."
                )
        else:
            cat_stat["irrelevant_queries"] += 1
            if actually_rejected:
                cat_stat["correctly_rejected"] += 1
            else:
                cat_stat["false_acceptances"] += 1
                is_failed = True
                failure_type = "THRESHOLD_FAILURE"
                top_false_cid = retrieved_cids[0]
                top_false_score = similarity_scores[top_false_cid]
                failure_explanation = (
                    f"Irrelevant query should have been rejected, but chunk '{top_false_cid}' exceeded threshold "
                    f"with cosine similarity {top_false_score:.4f} >= {threshold:.2f}."
                )

        eval_record = {
            "question_id": qid,
            "category": category,
            "question": question,
            "search_query_used": search_query,
            "expected_chunk_ids": list(expected_cids),
            "should_reject": should_reject,
            "actually_rejected": actually_rejected,
            "retrieved_chunk_ids": retrieved_cids,
            "similarity_scores": similarity_scores,
            "all_top_candidates": all_top_scores,
            "reciprocal_rank": round(rr, 4) if is_relevant else None,
            "precision_at_k": p_at_k if is_relevant else None,
            "recall_at_k": r_at_k if is_relevant else None,
            "is_failed": is_failed,
            "latency_ms": {
                "embed": round(embed_ms, 2),
                "search": round(search_ms, 2),
                "total": round(total_ms, 2),
            },
        }
        raw_eval_results.append(eval_record)

        if is_failed:
            failures.append({
                "question_id": qid,
                "original_question": question,
                "category": category,
                "search_query": search_query,
                "expected_relevant_chunks": list(expected_cids),
                "retrieved_chunks": retrieved_cids,
                "similarity_scores": similarity_scores,
                "expected_behavior": "REJECT" if should_reject else f"Retrieve chunks {list(expected_cids)}",
                "actual_behavior": f"Retrieved {retrieved_cids}" if retrieved_cids else "REJECTED (No chunks >= threshold)",
                "failure_classification": failure_type or "RANKING_FAILURE",
                "explanation": failure_explanation,
            })

    # Summary metric aggregation
    all_reciprocal_ranks = []
    all_p1 = []
    all_p2 = []
    all_p4 = []
    all_r1 = []
    all_r2 = []
    all_r4 = []
    total_irrelevant = 0
    total_rejected = 0

    cat_summaries = {}
    for cat, stats in category_metrics.items():
        rel_cnt = stats["relevant_queries"]
        irrel_cnt = stats["irrelevant_queries"]
        total_irrelevant += irrel_cnt
        total_rejected += stats["correctly_rejected"]

        mrr_val = statistics.mean(stats["reciprocal_ranks"]) if stats["reciprocal_ranks"] else None
        p1_val = statistics.mean(stats["precisions_at_k"][1]) if stats["precisions_at_k"][1] else None
        p2_val = statistics.mean(stats["precisions_at_k"][2]) if stats["precisions_at_k"][2] else None
        p4_val = statistics.mean(stats["precisions_at_k"][4]) if stats["precisions_at_k"][4] else None
        r1_val = statistics.mean(stats["recalls_at_k"][1]) if stats["recalls_at_k"][1] else None
        r2_val = statistics.mean(stats["recalls_at_k"][2]) if stats["recalls_at_k"][2] else None
        r4_val = statistics.mean(stats["recalls_at_k"][4]) if stats["recalls_at_k"][4] else None
        rej_rate = (stats["correctly_rejected"] / irrel_cnt) if irrel_cnt > 0 else None

        if stats["reciprocal_ranks"]:
            all_reciprocal_ranks.extend(stats["reciprocal_ranks"])
            all_p1.extend(stats["precisions_at_k"][1])
            all_p2.extend(stats["precisions_at_k"][2])
            all_p4.extend(stats["precisions_at_k"][4])
            all_r1.extend(stats["recalls_at_k"][1])
            all_r2.extend(stats["recalls_at_k"][2])
            all_r4.extend(stats["recalls_at_k"][4])

        cat_summaries[cat] = {
            "total_queries": stats["total_queries"],
            "relevant_queries": rel_cnt,
            "irrelevant_queries": irrel_cnt,
            "mean_reciprocal_rank_mrr": round(mrr_val, 4) if mrr_val is not None else None,
            "precision_at_1": round(p1_val, 4) if p1_val is not None else None,
            "precision_at_2": round(p2_val, 4) if p2_val is not None else None,
            "precision_at_4": round(p4_val, 4) if p4_val is not None else None,
            "recall_at_1": round(r1_val, 4) if r1_val is not None else None,
            "recall_at_2": round(r2_val, 4) if r2_val is not None else None,
            "recall_at_4": round(r4_val, 4) if r4_val is not None else None,
            "rejection_rate": round(rej_rate, 4) if rej_rate is not None else None,
            "correctly_rejected": stats["correctly_rejected"],
            "false_acceptances": stats["false_acceptances"],
            "mean_latency_ms": round(statistics.mean(stats["latencies_total_ms"]), 2),
        }

    overall_mrr = statistics.mean(all_reciprocal_ranks) if all_reciprocal_ranks else 0.0
    overall_p1 = statistics.mean(all_p1) if all_p1 else 0.0
    overall_p2 = statistics.mean(all_p2) if all_p2 else 0.0
    overall_p4 = statistics.mean(all_p4) if all_p4 else 0.0
    overall_r1 = statistics.mean(all_r1) if all_r1 else 0.0
    overall_r2 = statistics.mean(all_r2) if all_r2 else 0.0
    overall_r4 = statistics.mean(all_r4) if all_r4 else 0.0
    overall_rejection = (total_rejected / total_irrelevant) if total_irrelevant > 0 else 1.0

    latency_overall = {
        "mean_embed_ms": round(statistics.mean(latencies_embed_ms), 2),
        "mean_search_ms": round(statistics.mean(latencies_search_ms), 2),
        "mean_total_ms": round(statistics.mean(latencies_total_ms), 2),
        "median_total_ms": round(statistics.median(latencies_total_ms), 2),
        "p95_total_ms": round(np.percentile(latencies_total_ms, 95), 2),
        "min_total_ms": round(min(latencies_total_ms), 2),
        "max_total_ms": round(max(latencies_total_ms), 2),
    }

    print("=" * 85)
    print("           LOCALRAG 100-QUERY BASELINE RETRIEVAL AUDIT REPORT")
    print("=" * 85)
    print(f"Total Evaluated Queries:           {len(benchmark)}")
    print(f"Relevant / Answerable Queries:     {len(all_reciprocal_ranks)}")
    print(f"Irrelevant / Should-Reject:        {total_irrelevant}")
    print(f"Active Similarity Threshold:       {threshold:.2f}")
    print(f"Active Top-K:                      {top_k}")
    print(f"Embedding Model:                   {settings.embedding_model}")
    print("-" * 85)
    print("GLOBAL METRICS (Relevant Queries):")
    print(f"  Mean Reciprocal Rank (MRR):      {overall_mrr:.4f}")
    print(f"  Recall@1:                        {overall_r1:.4f}  |  Precision@1: {overall_p1:.4f}")
    print(f"  Recall@2:                        {overall_r2:.4f}  |  Precision@2: {overall_p2:.4f}")
    print(f"  Recall@4:                        {overall_r4:.4f}  |  Precision@4: {overall_p4:.4f}")
    print("-" * 85)
    print("REJECTION METRICS (Irrelevant Queries):")
    print(f"  Overall Rejection Rate:          {overall_rejection * 100:.1f}% ({total_rejected}/{total_irrelevant})")
    unsupp_rej = cat_summaries['Unsupported']['rejection_rate']
    hardneg_rej = cat_summaries['Hard Negative']['rejection_rate']
    print(f"  Pure Unsupported Rejection Rate: {unsupp_rej * 100:.1f}% ({cat_summaries['Unsupported']['correctly_rejected']}/{cat_summaries['Unsupported']['irrelevant_queries']})")
    print(f"  Hard Negative Rejection Rate:    {hardneg_rej * 100:.1f}% ({cat_summaries['Hard Negative']['correctly_rejected']}/{cat_summaries['Hard Negative']['irrelevant_queries']})")
    print("-" * 85)
    print("LATENCY PERFORMANCE:")
    print(f"  Embedding Latency (mean):        {latency_overall['mean_embed_ms']} ms")
    print(f"  NumPy Search Latency (mean):     {latency_overall['mean_search_ms']} ms")
    print(f"  Total Retrieval Latency (mean):  {latency_overall['mean_total_ms']} ms")
    print(f"  Median Latency:                  {latency_overall['median_total_ms']} ms")
    print(f"  95th Percentile Latency:         {latency_overall['p95_total_ms']} ms")
    print("=" * 85)

    print("\nCATEGORY-WISE BREAKDOWN:")
    print(f"{'Category':<27} {'Count':<6} {'MRR':<8} {'Rec@1':<8} {'Prec@1':<8} {'Rec@4':<8} {'Rejection':<10} {'Latency(ms)':<10}")
    print("-" * 95)
    for cat_name, c_data in cat_summaries.items():
        cnt = c_data["total_queries"]
        mrr_str = f"{c_data['mean_reciprocal_rank_mrr']:.3f}" if c_data['mean_reciprocal_rank_mrr'] is not None else "N/A"
        r1_str = f"{c_data['recall_at_1']:.3f}" if c_data['recall_at_1'] is not None else "N/A"
        p1_str = f"{c_data['precision_at_1']:.3f}" if c_data['precision_at_1'] is not None else "N/A"
        r4_str = f"{c_data['recall_at_4']:.3f}" if c_data['recall_at_4'] is not None else "N/A"
        rej_str = f"{c_data['rejection_rate'] * 100:.1f}%" if c_data['rejection_rate'] is not None else "N/A"
        lat_str = f"{c_data['mean_latency_ms']:.1f}"
        print(f"{cat_name:<27} {cnt:<6} {mrr_str:<8} {r1_str:<8} {p1_str:<8} {r4_str:<8} {rej_str:<10} {lat_str:<10}")

    print(f"\nTotal Failed Queries Identified: {len(failures)}")
    failure_counts = {}
    for f in failures:
        fc = f["failure_classification"]
        failure_counts[fc] = failure_counts.get(fc, 0) + 1
    print("Failure Classification Taxonomy Distribution:")
    for fc, cnt in sorted(failure_counts.items(), key=lambda x: -x[1]):
        print(f"  - {fc}: {cnt} queries ({cnt / len(failures) * 100:.1f}%)")

    # Save baseline results
    results_file = PROJECT_ROOT / "data" / "evaluation" / "baseline_results.json"
    results_file.write_text(json.dumps({
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_benchmark_queries": len(benchmark),
            "relevant_queries": len(all_reciprocal_ranks),
            "irrelevant_queries": total_irrelevant,
            "embedding_model": settings.embedding_model,
            "similarity_threshold": threshold,
            "top_k": top_k,
            "indexed_document": "machine_learning_algorithms.pdf",
            "indexed_chunks": len(index.chunks),
        },
        "overall_metrics": {
            "mean_reciprocal_rank_mrr": round(overall_mrr, 4),
            "precision_at_1": round(overall_p1, 4),
            "precision_at_2": round(overall_p2, 4),
            "precision_at_4": round(overall_p4, 4),
            "recall_at_1": round(overall_r1, 4),
            "recall_at_2": round(overall_r2, 4),
            "recall_at_4": round(overall_r4, 4),
            "overall_rejection_rate": round(overall_rejection, 4),
            "pure_unsupported_rejection_rate": round(unsupp_rej, 4),
            "hard_negative_rejection_rate": round(hardneg_rej, 4),
            "latency": latency_overall,
        },
        "category_metrics": cat_summaries,
        "queries": raw_eval_results,
    }, indent=2), encoding="utf-8")
    print(f"Saved baseline results to {results_file}")

    # Save failure analysis
    failure_file = PROJECT_ROOT / "data" / "evaluation" / "failure_analysis.json"
    failure_file.write_text(json.dumps({
        "total_failures": len(failures),
        "classification_breakdown": failure_counts,
        "failed_queries": failures,
    }, indent=2), encoding="utf-8")
    print(f"Saved failure analysis to {failure_file}")

    return {
        "benchmark_count": len(benchmark),
        "evaluated_count": len(raw_eval_results),
        "overall_metrics": {
            "mrr": overall_mrr,
            "r1": overall_r1,
            "p1": overall_p1,
            "r4": overall_r4,
            "p4": overall_p4,
            "rejection": overall_rejection,
            "unsupported_rejection": unsupp_rej,
            "hardneg_rejection": hardneg_rej,
        },
        "category_summaries": cat_summaries,
        "failures": failures,
        "failure_counts": failure_counts,
        "latency": latency_overall,
    }


if __name__ == "__main__":
    run_benchmark_audit()
