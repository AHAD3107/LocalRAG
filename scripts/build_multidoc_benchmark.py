"""Builds the 155-query multi-document benchmark for LocalRAG Phase 4.8.

Covers 5 distinct domains:
1. Machine Learning (machine_learning_algorithms.pdf)
2. Database Systems (database_systems_architecture.pdf)
3. Computer Networks (computer_networks_protocols.pdf)
4. Python Internals (python_runtime_internals.pdf)
5. Software Engineering (software_engineering_microservices.pdf)

Validates every ground-truth chunk_id against the loaded VectorIndex.
Saves to data/evaluation/multidoc_benchmark.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import settings
from src.storage.vector_index import VectorIndex

# Load VectorIndex to verify all chunk IDs exist
index = VectorIndex.load(PROJECT_ROOT / "data" / "index", settings.embedding_model, settings.chunk_size, settings.chunk_overlap)
chunk_map = {c.chunk_id: c for c in index.chunks}
print(f"Loaded {len(chunk_map)} total chunks across {len(set(c.doc_id for c in index.chunks))} documents.")

# Map filenames to doc IDs
DOCS = {}
for c in index.chunks:
    DOCS[c.source_filename] = c.doc_id

DOC_ML = DOCS["machine_learning_algorithms.pdf"]
DOC_DB = DOCS["database_systems_architecture.pdf"]
DOC_NET = DOCS["computer_networks_protocols.pdf"]
DOC_PY = DOCS["python_runtime_internals.pdf"]
DOC_SE = DOCS["software_engineering_microservices.pdf"]

# Helper to find chunk containing a keyword substring
def find_chunk_id(doc_id: str, keyword: str, page_number: int | None = None) -> str:
    matches = []
    for cid, c in chunk_map.items():
        if c.doc_id == doc_id:
            if page_number is not None and c.page_number != page_number:
                continue
            if keyword.lower() in c.text.lower():
                matches.append(cid)
    if not matches:
        raise ValueError(f"No chunk found for doc {doc_id} with keyword '{keyword}' (page {page_number})")
    return matches[0]

def find_chunk_ids(doc_id: str, keyword: str, page_number: int | None = None) -> list[str]:
    matches = []
    for cid, c in chunk_map.items():
        if c.doc_id == doc_id:
            if page_number is not None and c.page_number != page_number:
                continue
            if keyword.lower() in c.text.lower():
                matches.append(cid)
    if not matches:
        raise ValueError(f"No chunks found for doc {doc_id} with keyword '{keyword}' (page {page_number})")
    return matches

# Build the 155 Benchmark Questions
BENCHMARK_RAW = []

# =========================================================================
# Category 1: Exact Factual Questions (25 queries - 5 per domain)
# =========================================================================
exact_factual = [
    # ML (5)
    ("What is the default value for Prior Knowledge P(L0) in BKT?", "0.20", DOC_ML, [find_chunk_id(DOC_ML, "P(L0)", 3)], False),
    ("What does the binary Q-matrix represent in the DINA model?", "A binary table linking test questions to latent psychological skills", DOC_ML, [find_chunk_id(DOC_ML, "binary table linking", 4)], False),
    ("What is the inference latency target for KnowledgeTrace MCQ?", "< 15ms (NumPy Vectorized)", DOC_ML, [find_chunk_id(DOC_ML, "< 15ms", 1)], False),
    ("Who formalized the DINA cognitive diagnosis model and in which year?", "Junker & Sijtsma (2001)", DOC_ML, [find_chunk_id(DOC_ML, "Junker & Sijtsma", 4)], False),
    ("What is the guess probability in a 4-choice question in KnowledgeTrace MCQ?", "25% chance of guessing correctly", DOC_ML, [find_chunk_id(DOC_ML, "25% chance", 1)], False),
    # DB (5)
    ("What are the standard disk page sizes in PostgreSQL and MySQL InnoDB?", "8 KB for PostgreSQL and 16 KB for MySQL InnoDB", DOC_DB, [find_chunk_id(DOC_DB, "8 KB (PostgreSQL) or 16 KB", 1)], False),
    ("What are the three distinct phases of the ARIES recovery algorithm?", "Analysis Phase, Redo Phase, and Undo Phase", DOC_DB, [find_chunk_id(DOC_DB, "Analysis Phase: Scans the log", 1)], False),
    ("What fanout factor M is achieved by an 8 KB B+ Tree internal node storing 8-byte keys and 8-byte pointers?", "M = 512", DOC_DB, [find_chunk_id(DOC_DB, "fanout factor M = 512", 2)], False),
    ("What background cycle-detection algorithm detects deadlocks in the Wait-For Graph every 500 ms?", "Tarjan's strongly connected components algorithm", DOC_DB, [find_chunk_id(DOC_DB, "Tarjan's strongly connected components", 3)], False),
    ("How many virtual nodes (vnodes) is each physical server mapped to in the consistent hashing ring?", "256 virtual nodes (vnodes)", DOC_DB, [find_chunk_id(DOC_DB, "256 virtual nodes", 4)], False),
    # NET (5)
    ("What is the minimum encapsulation header overhead of an Ethernet, IPv4, and TCP packet?", "54 bytes (14 bytes Ethernet + 20 bytes IPv4 + 20 bytes TCP)", DOC_NET, [find_chunk_id(DOC_NET, "54 bytes of encapsulation", 1)], False),
    ("What is the duration of the TCP TIME_WAIT state defined as 2 times MSL?", "120 seconds (2 * MSL)", DOC_NET, [find_chunk_id(DOC_NET, "120 seconds", 2)], False),
    ("What is the Maximum Segment Size (MSS) for standard Ethernet with 1500-byte MTU?", "1460 bytes", DOC_NET, [find_chunk_id(DOC_NET, "Maximum Segment Size (MSS) of 1460 bytes", 1)], False),
    ("What mathematical curve does TCP Cubic use to model window growth?", "W_cubic(t) = C * (t - K)^3 + W_max", DOC_NET, [find_chunk_id(DOC_NET, "W_cubic(t)", 3)], False),
    ("Over which transport layer protocol does HTTP/3 operate?", "UDP on port 443 via QUIC", DOC_NET, [find_chunk_id(DOC_NET, "Operates over UDP on port 443", 4)], False),
    # PY (5)
    ("What parser was introduced in Python 3.9 implemented in Parser/pegen.c?", "The PEG parser", DOC_PY, [find_chunk_id(DOC_PY, "PEG parser", 1)], False),
    ("What fixed wordcode instruction size was introduced in Python 3.6 for bytecode?", "Fixed 16-bit wordcode format (8-bit opcode, 8-bit oparg)", DOC_PY, [find_chunk_id(DOC_PY, "16-bit wordcode format", 1)], False),
    ("What memory size threshold determines when PyMalloc delegates to system malloc?", "Allocations exceeding 512 bytes", DOC_PY, [find_chunk_id(DOC_PY, "exceeding 512 bytes", 2)], False),
    ("What is the default GIL switching interval in CPython configured via sys.getswitchinterval()?", "5.0 milliseconds", DOC_PY, [find_chunk_id(DOC_PY, "5.0 milliseconds", 3)], False),
    ("What three methods define the Python descriptor protocol?", "__get__, __set__, and __delete__", DOC_PY, [find_chunk_id(DOC_PY, "Descriptor Protocol", 4)], False),
    # SE (5)
    ("What law states that organizations design systems mirroring their communication structures?", "Conway's Law", DOC_SE, [find_chunk_id(DOC_SE, "Conway's Law", 1)], False),
    ("What pattern coordinates distributed transactions without shared ACID locks as a sequence of local transactions?", "The Saga pattern", DOC_SE, [find_chunk_id(DOC_SE, "Saga pattern", 2)], False),
    ("What background engine streams outbox records from PostgreSQL WAL logs to Kafka?", "Debezium", DOC_SE, [find_chunk_id(DOC_SE, "Debezium", 2)], False),
    ("What are the three states of a Martin Fowler Circuit Breaker?", "Closed, Open, and Half-Open", DOC_SE, [find_chunk_id(DOC_SE, "Closed State", 3)], False),
    ("What header standard is used for W3C distributed tracing context propagation?", "traceparent header with 16-byte trace-id and 8-byte parent-id", DOC_SE, [find_chunk_id(DOC_SE, "traceparent: 00-4bf92f3577b34da6a3ce929d0e0e4736", 4)], False),
]

for q, ans, did, cids, rej in exact_factual:
    BENCHMARK_RAW.append({
        "question": q,
        "category": "Exact factual",
        "expected_answer": ans,
        "relevant_doc_id": did,
        "relevant_chunk_ids": cids,
        "should_reject": rej,
    })

# =========================================================================
# Category 2: Paraphrased Questions (25 queries - 5 per domain)
# =========================================================================
paraphrased = [
    # ML (5)
    ("How does the educational diagnostic system prevent circular prerequisite dependencies?", "By detecting cycles and maintaining a Directed Acyclic Graph (DAG) invariant in NetworkX", DOC_ML, [find_chunk_id(DOC_ML, "Cycle Detection & DAG Invariant", 3)], False),
    ("Why is raw percentage grading insufficient for diagnosing deep student understanding?", "Because it treats all questions equally and ignores lucky guesses and careless slips", DOC_ML, [find_chunk_id(DOC_ML, "scoring is basic arithmetic", 1)], False),
    ("What mathematical structure represents student skill mastery transitions over sequential practice attempts?", "A Hidden Markov Model (HMM) with 2 states: unmastered and mastered", DOC_ML, [find_chunk_id(DOC_ML, "Hidden Markov Model (HMM) with 2 possible", 3)], False),
    ("How does the system ensure non-blocking UI responsiveness when processing cognitive diagnosis?", "By committing responses to an API ingestion queue and running diagnosis in background Celery workers", DOC_ML, [find_chunk_id(DOC_ML, "never experiences lag or waiting time", 2)], False),
    ("What historical fraction subtraction dataset serves as a benchmark for Q-Matrix validation?", "Tatsuoka Fraction Subtraction (1990) with 536 students and 20 items", DOC_ML, [find_chunk_id(DOC_ML, "Tatsuoka")], False),
    # DB (5)
    ("Why do database storage engines prefer CLOCK-Pro or 2Q over simple LRU for buffer eviction?", "To prevent sequential scan pollution from evicting frequently accessed index pages", DOC_DB, [find_chunk_id(DOC_DB, "CLOCK-Pro eviction algorithm rather than naive LRU", 1)], False),
    ("How do LSM-Trees reduce write amplification on solid-state drives compared to traditional B-Trees?", "By buffering writes in memory (MemTable) and appending them sequentially to SSTables", DOC_DB, [find_chunk_id(DOC_DB, "convert random writes into sequential writes", 2)], False),
    ("In PostgreSQL MVCC, how does a transaction know if a row version was created by a committed transaction?", "By checking the xmin transaction ID against its active snapshot Read View", DOC_DB, [find_chunk_id(DOC_DB, "xmin (the transaction ID that created the tuple)", 3)], False),
    ("How many votes must a candidate receive to become the leader in a 5-node Raft cluster?", "At least 3 votes (floor(N/2) + 1 strict majority quorum)", DOC_DB, [find_chunk_id(DOC_DB, "Quorum Invariant: A candidate becomes Leader", 4)], False),
    ("What prevents two transactions from modifying the same record simultaneously in Strict 2PL?", "Exclusive locks (X-Locks) held until the transaction commits or aborts", DOC_DB, [find_chunk_id(DOC_DB, "Exclusive Locks (X-Locks): Required for write operations", 3)], False),
    # NET (5)
    ("Why does TCP require a 3-way exchange before transmitting application payload bytes?", "To synchronize Initial Sequence Numbers (ISNs) and allocate Transmission Control Blocks (TCBs)", DOC_NET, [find_chunk_id(DOC_NET, "TCP Three-Way Handshake", 2)], False),
    ("How does the TCP receiver prevent a high-speed sender from overflowing its socket receive queue?", "By advertising its remaining available buffer space in the rcv_wnd window field", DOC_NET, [find_chunk_id(DOC_NET, "Sliding Window Protocol", 2)], False),
    ("What prevents delayed duplicate packets from an old connection from interfering with a new TCP session?", "The 120-second TIME_WAIT state (2 * Maximum Segment Lifetime)", DOC_NET, [find_chunk_id(DOC_NET, "TIME_WAIT state for 2 * MSL", 2)], False),
    ("How does Google's BBR congestion control differ fundamentally from loss-based Cubic?", "BBR models bottleneck bandwidth and minimum RTT instead of waiting for packet drops", DOC_NET, [find_chunk_id(DOC_NET, "BBR (Bottleneck Bandwidth and RTT)", 3)], False),
    ("Why does packet loss in HTTP/3 not block unrelated concurrent streams unlike HTTP/2?", "QUIC runs over UDP with independent per-stream loss recovery", DOC_NET, [find_chunk_id(DOC_NET, "Independent Stream Loss Recovery", 4)], False),
    # PY (5)
    ("How does PEP 659 adaptive specializing accelerate CPython bytecode without JIT compilation?", "By dynamically replacing generic opcodes with specialized variants for observed data types", DOC_PY, [find_chunk_id(DOC_PY, "Specializing Adaptive Interpreter", 1)], False),
    ("Why does CPython organize small object allocations into 256 KB arenas and 4 KB pools?", "To eliminate operating system heap fragmentation for objects of 512 bytes or smaller", DOC_PY, [find_chunk_id(DOC_PY, "PyMalloc Object Allocator", 2)], False),
    ("What cleanup mechanism handles reference cycles that simple reference counting fails to free?", "The cyclic generational garbage collector (gc module) with 3 generation rings", DOC_PY, [find_chunk_id(DOC_PY, "Cyclic Garbage Collector", 2)], False),
    ("How do C extensions like NumPy execute parallel multi-threaded computations despite the GIL?", "By releasing the GIL via Py_BEGIN_ALLOW_THREADS during intensive computations", DOC_PY, [find_chunk_id(DOC_PY, "Py_BEGIN_ALLOW_THREADS", 3)], False),
    ("How does the asyncio event loop avoid consuming 100% CPU while waiting for socket network data?", "By utilizing operating system I/O multiplexing primitives such as epoll or IOCP", DOC_PY, [find_chunk_id(DOC_PY, "I/O Demultiplexing", 4)], False),
    # SE (5)
    ("Why should microservices avoid accessing each other's underlying database tables directly?", "To preserve loose coupling and allow services to alter their internal schemas independently", DOC_SE, [find_chunk_id(DOC_SE, "Database-per-Service Pattern", 1)], False),
    ("How does the Transactional Outbox pattern prevent lost event messages during server crashes?", "By saving the business record and the outbox event together in a single atomic database transaction", DOC_SE, [find_chunk_id(DOC_SE, "Transactional Outbox Pattern", 2)], False),
    ("What prevents an unresponsive downstream service from crashing calling services in a cascade?", "A Circuit Breaker that trips to OPEN and fails fast after reaching an error threshold", DOC_SE, [find_chunk_id(DOC_SE, "Circuit Breaker Pattern", 3)], False),
    ("Why does exponential backoff add randomized full jitter to retry intervals?", "To prevent synchronized retry storms and the thundering herd problem", DOC_SE, [find_chunk_id(DOC_SE, "Exponential Backoff with Full Jitter", 3)], False),
    ("What deployment technique switches production traffic immediately between two identical environments?", "Blue-Green Deployment", DOC_SE, [find_chunk_id(DOC_SE, "Blue-Green Deployment", 4)], False),
]

for q, ans, did, cids, rej in paraphrased:
    BENCHMARK_RAW.append({
        "question": q,
        "category": "Paraphrased",
        "expected_answer": ans,
        "relevant_doc_id": did,
        "relevant_chunk_ids": cids,
        "should_reject": rej,
    })

# =========================================================================
# Category 3: Questions Requiring Multiple Chunks (20 queries)
# =========================================================================
multi_chunk = [
    # ML Multi-chunk (4)
    ("Explain the complete lifecycle of a student answer from ingestion to BKT and DINA diagnosis.", "Step 1 HTTP POST, Step 2 synchronous validation, Step 3 asynchronous Celery dispatch, Step 4 BKT mastery update and DINA Q-matrix partitioning.", DOC_ML, [find_chunk_id(DOC_ML, "End-to-End System Data Flow Architecture", 2), find_chunk_id(DOC_ML, "Step 4 — Layer 2 BKT Sequential Update", 2)], False),
    ("How do the DINA model and Bayesian Knowledge Tracing collaborate to provide holistic student assessment?", "BKT tracks sequential mastery over time as an HMM, while DINA evaluates multi-attribute skill requirements via a binary Q-matrix.", DOC_ML, [find_chunk_id(DOC_ML, "Core Foundations of DINA", 4), find_chunk_id(DOC_ML, "Hidden Markov Model (HMM) with 2 possible", 3)], False),
    ("What are the mathematical equations for BKT prior update and confidence score calculation?", "BKT uses Bayesian posterior update P(L_t|obs), while Confidence = 1 - 4 * P * (1 - P).", DOC_ML, [find_chunk_id(DOC_ML, "Latent Learning Transition", 3), find_chunk_id(DOC_ML, "Confidence Score Equation", 4)], False),
    ("What benchmarks validate concept DAG sequencing and cognitive diagnosis scalability?", "ASSISTments (Worcester) validates BKT with AUC ~0.73, Junyi Academy validates 835 concept DAG, and Tatsuoka validates Q-matrix DINA.", DOC_ML, [find_chunk_id(DOC_ML, "Benchmark Dataset", 6), find_chunk_id(DOC_ML, "Tatsuoka")], False),

    # DB Multi-chunk (4)
    ("How does ARIES recovery interact with Write-Ahead Logging and checkpoints during system crash recovery?", "WAL flushes dirty page logs before writes; upon crash, ARIES Analysis identifies dirty pages from checkpoint, Redo replays history, Undo reverses uncommitted transactions.", DOC_DB, [find_chunk_id(DOC_DB, "Write-Ahead Logging (WAL) & ARIES Recovery", 1), find_chunk_id(DOC_DB, "Buffer Pool Management", 1)], False),
    ("Compare the indexing mechanisms of B+ Trees with LSM-Trees regarding write vs read optimization.", "B+ Trees optimize reads with 512 fanout and leaf linked lists but cause random writes; LSM-Trees optimize writes via sequential SSTables and MemTables, using Bloom filters for reads.", DOC_DB, [find_chunk_id(DOC_DB, "B+ Tree Internal Structure and Fanout", 2), find_chunk_id(DOC_DB, "Log-Structured Merge-Trees (LSM-Trees)", 2)], False),
    ("How do MVCC tuple headers and autovacuum work together to maintain database performance?", "Tuple headers use xmin and xmax for snapshot visibility without locking; autovacuum cleans up dead tuples whose xmax is older than active snapshots.", DOC_DB, [find_chunk_id(DOC_DB, "Multi-Version Concurrency Control (MVCC)", 3), find_chunk_id(DOC_DB, "Vacuuming: Obsolete tuple versions", 3)], False),
    ("Describe how Raft consensus and Two-Phase Commit operate together in a distributed partitioned database.", "Raft replicates logs across nodes within a partition quorum, while Two-Phase Commit ensures atomic multi-partition transactions across clusters via prepare and commit phases.", DOC_DB, [find_chunk_id(DOC_DB, "The Raft Consensus Protocol", 4), find_chunk_id(DOC_DB, "Distributed Transactions (Two-Phase Commit", 4)], False),

    # NET Multi-chunk (4)
    ("How do TCP sliding window flow control and AIMD congestion control interact to govern transmission rates?", "Flow control limits unacknowledged bytes to receiver buffer rcv_wnd, while congestion control dynamically adjusts cwnd based on network capacity; transmission is min(cwnd, rcv_wnd).", DOC_NET, [find_chunk_id(DOC_NET, "Sliding Window Protocol", 2), find_chunk_id(DOC_NET, "Principles of Network Congestion Control", 3)], False),
    ("Trace the lifecycle of a TCP connection from 3-way handshake to 4-step teardown and TIME_WAIT.", "Begins with SYN -> SYN-ACK -> ACK exchanging ISNs; ends with FIN -> ACK -> FIN -> ACK followed by 120-second TIME_WAIT to purge delayed duplicates.", DOC_NET, [find_chunk_id(DOC_NET, "TCP Three-Way Handshake", 2), find_chunk_id(DOC_NET, "TCP Teardown and the TIME_WAIT State", 2)], False),
    ("Compare HTTP/2 stream multiplexing with HTTP/3 over QUIC regarding packet loss behavior.", "HTTP/2 interleaves binary frames over one TCP connection causing HoL blocking on dropped packets; HTTP/3 runs over UDP/QUIC with independent per-stream loss recovery.", DOC_NET, [find_chunk_id(DOC_NET, "HTTP/2 Binary Framing & Stream Multiplexing", 4), find_chunk_id(DOC_NET, "HTTP/3 over QUIC Protocol", 4)], False),
    ("How does packet encapsulation overhead affect the Maximum Segment Size on standard Ethernet?", "Ethernet 14B + IPv4 20B + TCP 20B = 54B total encapsulation; on a 1500-byte MTU, this limits TCP MSS to exactly 1460 bytes.", DOC_NET, [find_chunk_id(DOC_NET, "Packet Encapsulation and Header Overhead", 1), find_chunk_id(DOC_NET, "Maximum Transmission Unit (MTU)", 1)], False),

    # PY Multi-chunk (4)
    ("Explain the journey of Python code from parsing and AST compilation to bytecode evaluation in ceval.", "The PEG parser converts source to parse tree, AST builds syntax grammar, compiler creates 16-bit wordcode code objects, and ceval loop executes instructions with PEP 659 quickening.", DOC_PY, [find_chunk_id(DOC_PY, "CPython Execution and Parsing Pipeline", 1), find_chunk_id(DOC_PY, "Specializing Adaptive Interpreter", 1)], False),
    ("How does CPython combine reference counting with generational garbage collection to manage memory?", "Reference counting immediately deallocates objects when refcnt drops to 0; the cyclic gc scans Gen 0, 1, and 2 rings to collect circular references.", DOC_PY, [find_chunk_id(DOC_PY, "The CPython Memory Allocator Hierarchy", 2), find_chunk_id(DOC_PY, "Reference Counting and Garbage Collection", 2)], False),
    ("What architectural changes does PEP 703 Free-Threading introduce to eliminate the Global Interpreter Lock?", "Replaces PyMalloc with thread-safe mimalloc, introduces Biased Reference Counting for local threads, and creates immortal objects to avoid atomic operations.", DOC_PY, [find_chunk_id(DOC_PY, "The Global Interpreter Lock (GIL) Architecture", 3), find_chunk_id(DOC_PY, "PEP 703 Free-Threaded Python", 3)], False),
    ("How do Python descriptors and generator frame states power coroutines and asyncio event loops?", "Descriptors manage class/function access; generators maintain resumable PyGenObject frame states on the heap, which async/await schedules onto the event loop using OS demultiplexing.", DOC_PY, [find_chunk_id(DOC_PY, "The Descriptor Protocol", 4), find_chunk_id(DOC_PY, "Asyncio Event Loop Mechanics", 4)], False),

    # SE Multi-chunk (4)
    ("How do the Saga pattern and Transactional Outbox pattern solve distributed consistency without 2PC?", "Sagas coordinate local transactions via choreography or orchestration with compensating rollbacks, while the Outbox pattern with Debezium CDC guarantees atomic database writes and Kafka event publishing.", DOC_SE, [find_chunk_id(DOC_SE, "The Saga Pattern for Distributed Transactions", 2), find_chunk_id(DOC_SE, "The Transactional Outbox Pattern", 2)], False),
    ("How do Circuit Breakers, Bulkheads, and Exponential Jitter protect cloud architectures from cascading failures?", "Circuit breakers fail fast when error rates exceed 50%, bulkheads isolate thread/connection pools, and full jitter randomizes backoff sleep to eliminate retry storms.", DOC_SE, [find_chunk_id(DOC_SE, "The Circuit Breaker Pattern", 3), find_chunk_id(DOC_SE, "Exponential Backoff with Full Jitter", 3)], False),
    ("Explain how Domain-Driven Design bounded contexts align with Conway's Law in microservice organizations.", "Conway's Law states team structures dictate architecture; DDD bounded contexts define autonomous business subdomains, enabling independent two-pizza teams to own services end-to-end.", DOC_SE, [find_chunk_id(DOC_SE, "Monolithic vs Microservices Tradeoffs", 1), find_chunk_id(DOC_SE, "Domain-Driven Design (DDD) & Bounded Contexts", 1)], False),
    ("How do the 12-Factor stateless process guidelines enable zero-downtime Blue-Green deployments?", "Factor VI mandates stateless share-nothing processes, allowing routers to switch 100% active traffic instantly from Blue to Green without losing user sessions.", DOC_SE, [find_chunk_id(DOC_SE, "The Twelve-Factor App Principles", 4), find_chunk_id(DOC_SE, "Zero-Downtime Deployment Strategies", 4)], False),
]

for q, ans, did, cids, rej in multi_chunk:
    BENCHMARK_RAW.append({
        "question": q,
        "category": "Multi-chunk",
        "expected_answer": ans,
        "relevant_doc_id": did,
        "relevant_chunk_ids": cids,
        "should_reject": rej,
    })

# =========================================================================
# Category 4: Cross-Document Questions (20 queries)
# =========================================================================
cross_doc = [
    ("Compare the concurrency mechanisms: how does Python's GIL mutex compare to Database Strict 2PL locking?", "Python's GIL is a single mutex serializing VM bytecode execution across threads, while Strict 2PL uses granular shared/exclusive locks on tuples to ensure serializable transactions.", None, [find_chunk_id(DOC_PY, "The Global Interpreter Lock (GIL) Architecture", 3), find_chunk_id(DOC_DB, "Two-Phase Locking (2PL) and Deadlock Handling", 3)], False),
    ("How does Celery asynchronous queueing in KnowledgeTrace MCQ implement the Microservices decoupling principle?", "It offloads heavy psychometric computations to background Celery workers so HTTP requests return in under 15ms, adhering to non-blocking service communication.", None, [find_chunk_id(DOC_ML, "never experiences lag or waiting time", 2), find_chunk_id(DOC_SE, "Database-per-Service Pattern", 1)], False),
    ("Contrast the memory management of CPython's PyMalloc pools with Database buffer pool page frames.", "PyMalloc manages small CPU memory allocations up to 512 bytes in 4KB pools, while DB buffer pools manage 8KB/16KB disk page frames using CLOCK-Pro eviction.", None, [find_chunk_id(DOC_PY, "The CPython Memory Allocator Hierarchy", 2), find_chunk_id(DOC_DB, "Buffer Pool Management & Eviction", 1)], False),
    ("How does TCP packet encapsulation overhead compare to the W3C TraceContext traceparent HTTP header overhead?", "TCP encapsulation adds 54 bytes at layers 2-4, while W3C TraceContext adds an application header containing a 16-byte trace-id and 8-byte parent-id.", None, [find_chunk_id(DOC_NET, "Packet Encapsulation and Header Overhead", 1), find_chunk_id(DOC_SE, "Distributed Tracing & OpenTelemetry", 4)], False),
    ("Compare the failure recovery of Database ARIES WAL logging with Microservices Saga compensating transactions.", "ARIES WAL replays physical page history (Redo) and rolls back active transactions (Undo) on a single node, while Sagas execute semantic compensating business actions across distributed services.", None, [find_chunk_id(DOC_DB, "Write-Ahead Logging (WAL) & ARIES Recovery", 1), find_chunk_id(DOC_SE, "The Saga Pattern for Distributed Transactions", 2)], False),
    ("Contrast B+ Tree leaf linked lists in database engines with Python generator resumable frame states.", "B+ Tree leaf lists link disk pages bidirectionally for sequential storage access, while Python generators maintain execution frames on the heap to resume state upon next().", None, [find_chunk_id(DOC_DB, "B+ Tree Internal Structure and Fanout", 2), find_chunk_id(DOC_PY, "Generator Execution and PyGenObject", 4)], False),
    ("How does network socket I/O multiplexing in CPython asyncio connect to TCP sliding window buffer availability?", "The asyncio event loop uses OS epoll/IOCP to detect when a socket is ready for reading/writing, matching the TCP receiver's advertised rcv_wnd window space.", None, [find_chunk_id(DOC_PY, "Asyncio Event Loop Mechanics", 4), find_chunk_id(DOC_NET, "Sliding Window Protocol", 2)], False),
    ("Compare the distributed consensus of Raft with the Two-Phase Commit protocol for cross-partition transactions.", "Raft achieves consensus within a single replica group using a majority quorum, whereas 2PC coordinates atomic execution across distinct partitions requiring unanimous agreement.", None, [find_chunk_id(DOC_DB, "The Raft Consensus Protocol", 4), find_chunk_id(DOC_DB, "Distributed Transactions (Two-Phase Commit", 4)], False),
    ("How do Debezium CDC and PostgreSQL WAL logs enable the Transactional Outbox pattern in software engineering?", "PostgreSQL records changes in WAL for durability; Debezium reads these WAL logs directly to stream outbox table events to Kafka without double-writing.", None, [find_chunk_id(DOC_DB, "Write-Ahead Logging (WAL) & ARIES Recovery", 1), find_chunk_id(DOC_SE, "The Transactional Outbox Pattern", 2)], False),
    ("Compare TCP 120-second TIME_WAIT state with Circuit Breaker 10-second Half-Open cooldown.", "TIME_WAIT protects against late duplicate packets from closed TCP connections, while Circuit Breaker cooldown allows trial probe requests to test service recovery.", None, [find_chunk_id(DOC_NET, "TCP Teardown and the TIME_WAIT State", 2), find_chunk_id(DOC_SE, "The Circuit Breaker Pattern", 3)], False),
    ("How does CPython reference counting deallocation compare to PostgreSQL autovacuum dead tuple collection?", "CPython ref counting deallocates immediately when refcnt hits 0, whereas PostgreSQL keeps obsolete tuple versions until autovacuum reclaims dead space.", None, [find_chunk_id(DOC_PY, "Reference Counting and Garbage Collection", 2), find_chunk_id(DOC_DB, "Vacuuming: Obsolete tuple versions", 3)], False),
    ("Compare Tarjan's cycle detection algorithm in Database Wait-For Graphs with NetworkX cycle detection in concept DAGs.", "Both use cycle detection to prevent deadlocks in transactions (Wait-For Graph) and prevent circular prerequisite loops in educational concept DAGs.", None, [find_chunk_id(DOC_DB, "Two-Phase Locking (2PL) and Deadlock Handling", 3), find_chunk_id(DOC_ML, "Cycle Detection & DAG Invariant", 3)], False),
    ("How does HTTP/3 UDP multiplexing avoid the Head-of-Line blocking seen in HTTP/1.1 and HTTP/2?", "HTTP/3 runs over QUIC/UDP with independent per-stream loss recovery so dropped packets on one stream do not stall unrelated streams.", None, [find_chunk_id(DOC_NET, "HTTP/3 over QUIC Protocol", 4), find_chunk_id(DOC_NET, "HTTP/2 Binary Framing & Stream Multiplexing", 4)], False),
    ("How does the 12-Factor App methodology for configuration management relate to Python environment variables?", "Factor III requires storing config in environment variables, which Python applications access via os.getenv without hardcoding secrets in code objects.", None, [find_chunk_id(DOC_SE, "The Twelve-Factor App Principles", 4), find_chunk_id(DOC_PY, "CPython Execution and Parsing Pipeline", 1)], False),
    ("Compare Exponential Backoff with Jitter in microservices to TCP AIMD Congestion Control backoff.", "Both prevent congestion collapse: TCP cuts its window in half upon packet drop, while microservices back off with randomized jitter to prevent thundering herd retries.", None, [find_chunk_id(DOC_NET, "The Classic AIMD Algorithm", 3), find_chunk_id(DOC_SE, "Exponential Backoff with Full Jitter", 3)], False),
    ("How do Skip-Lists in LSM-Tree MemTables compare to B+ Tree balanced search trees?", "Skip-lists provide probabilistic O(log N) in-memory insertion for sequential disk flushing, while B+ Trees provide deterministic O(log N) disk-block fanout.", None, [find_chunk_id(DOC_DB, "Log-Structured Merge-Trees (LSM-Trees)", 2), find_chunk_id(DOC_DB, "B+ Tree Internal Structure and Fanout", 2)], False),
    ("Contrast CPython wordcode 16-bit instructions with HTTP/2 binary framing structures.", "CPython uses fixed 16-bit wordcode (8-bit opcode, 8-bit oparg), whereas HTTP/2 uses binary frames with 31-bit stream IDs and length headers.", None, [find_chunk_id(DOC_PY, "Bytecode Instruction Format (Wordcode)", 1), find_chunk_id(DOC_NET, "HTTP/2 Binary Framing & Stream Multiplexing", 4)], False),
    ("How does the Bulkhead isolation pattern in microservices relate to thread pools in CPython?", "Bulkheads partition dedicated thread/connection pools so failure in auxiliary tasks cannot starve critical workflows, similar to isolating worker threads.", None, [find_chunk_id(DOC_SE, "Bulkhead Isolation Pattern", 3), find_chunk_id(DOC_PY, "The Global Interpreter Lock (GIL) Architecture", 3)], False),
    ("Compare the Blue-Green deployment strategy with distributed 2-Phase Commit coordinator decisions.", "Blue-Green switches active router traffic instantly upon verification, while 2PC commits distributed nodes only after all participants vote to commit.", None, [find_chunk_id(DOC_SE, "Zero-Downtime Deployment Strategies", 4), find_chunk_id(DOC_DB, "Distributed Transactions (Two-Phase Commit", 4)], False),
    ("How does consistent hashing with 256 virtual nodes balance database shards compared to DNS hierarchical resolution?", "Consistent hashing uniformly distributes data keys across server rings via vnodes, while DNS distributes name queries across root, TLD, and authoritative trees.", None, [find_chunk_id(DOC_DB, "Data Partitioning and Consistent Hashing", 4), find_chunk_id(DOC_NET, "Domain Name System (DNS) Architecture", 4)], False),
]

for q, ans, did, cids, rej in cross_doc:
    BENCHMARK_RAW.append({
        "question": q,
        "category": "Cross-document",
        "expected_answer": ans,
        "relevant_doc_id": did,
        "relevant_chunk_ids": cids,
        "should_reject": rej,
    })

# =========================================================================
# Category 5: Document-Specific Questions (20 queries: 16 valid, 4 invalid)
# =========================================================================
doc_specific = [
    # Scoped to DB (4)
    ("What are the ARIES recovery phases described in the database specification?", "Analysis, Redo, and Undo", DOC_DB, [find_chunk_id(DOC_DB, "Analysis Phase: Scans the log", 1)], False),
    ("What is the B+ tree fanout for 8 KB pages with 8-byte keys?", "Fanout M = 512", DOC_DB, [find_chunk_id(DOC_DB, "fanout factor M = 512", 2)], False),
    ("How does MVCC check tuple visibility using xmin and xmax in PostgreSQL?", "Using xmin creation ID and xmax deletion ID against active read views", DOC_DB, [find_chunk_id(DOC_DB, "Tuple Headers: In PostgreSQL", 3)], False),
    ("What is the randomized election timeout range in the Raft consensus protocol?", "Between 150 ms and 300 ms", DOC_DB, [find_chunk_id(DOC_DB, "150 ms and 300 ms", 4)], False),

    # Scoped to NET (4)
    ("What is the standard Ethernet MTU size and header overhead?", "1500 bytes MTU and 54 bytes baseline encapsulation", DOC_NET, [find_chunk_id(DOC_NET, "Maximum Transmission Unit (MTU)", 1)], False),
    ("What are the steps of the TCP 3-way handshake?", "SYN from client, SYN-ACK from server, ACK from client", DOC_NET, [find_chunk_id(DOC_NET, "TCP Three-Way Handshake", 2)], False),
    ("What congestion control algorithm replaces linear increase with a cubic curve?", "TCP Cubic (RFC 8312)", DOC_NET, [find_chunk_id(DOC_NET, "TCP Cubic (RFC 8312)", 3)], False),
    ("What port and transport protocol are utilized by HTTP/3 QUIC?", "UDP on port 443", DOC_NET, [find_chunk_id(DOC_NET, "Operates over UDP on port 443", 4)], False),

    # Scoped to PY (4)
    ("What parser was introduced in Python 3.9 in Parser/pegen.c?", "The PEG parser", DOC_PY, [find_chunk_id(DOC_PY, "PEG parser", 1)], False),
    ("What size classes are managed by PyMalloc pools?", "Multiples of 8 bytes up to 512 bytes", DOC_PY, [find_chunk_id(DOC_PY, "multiples of 8 bytes", 2)], False),
    ("How often is the CPython GIL released during thread switching?", "Every 5.0 milliseconds", DOC_PY, [find_chunk_id(DOC_PY, "every 5.0 milliseconds", 3)], False),
    ("What protocol methods define a Python descriptor?", "__get__, __set__, and __delete__", DOC_PY, [find_chunk_id(DOC_PY, "The Descriptor Protocol", 4)], False),

    # Scoped to SE (4)
    ("What pattern breaks distributed transactions into local transactions with compensations?", "The Saga pattern", DOC_SE, [find_chunk_id(DOC_SE, "Saga pattern", 2)], False),
    ("What three states compose a Martin Fowler Circuit Breaker?", "Closed, Open, and Half-Open", DOC_SE, [find_chunk_id(DOC_SE, "The Circuit Breaker Pattern", 3)], False),
    ("What is the AWS Full Jitter formula for exponential backoff?", "Sleep = random(0, min(Max_Backoff, Base_Backoff * 2^attempt))", DOC_SE, [find_chunk_id(DOC_SE, "Exponential Backoff with Full Jitter", 3)], False),
    ("What deployment strategy maintains two identical environments for instant traffic switching?", "Blue-Green Deployment", DOC_SE, [find_chunk_id(DOC_SE, "Blue-Green Deployment", 4)], False),

    # Invalid Doc ID Scoped queries (4 - Should Reject)
    ("What is the ARIES recovery algorithm in the foreign document?", "Document not found in index", "foreign_doc_id_991", [], True),
    ("What is the TCP 3-way handshake in the non-existent specification?", "Document not found in index", "foreign_doc_id_992", [], True),
    ("What is the Python GIL switching interval in the missing documentation?", "Document not found in index", "foreign_doc_id_993", [], True),
    ("What is the Circuit Breaker timeout in the unindexed system?", "Document not found in index", "foreign_doc_id_994", [], True),
]

for q, ans, did, cids, rej in doc_specific:
    BENCHMARK_RAW.append({
        "question": q,
        "category": "Document-specific",
        "expected_answer": ans,
        "relevant_doc_id": did,
        "relevant_chunk_ids": cids,
        "should_reject": rej,
    })

# =========================================================================
# Category 6: Conversational Follow-up Questions (15 queries)
# =========================================================================
conversational = [
    # DB Conversational (3)
    (
        "What happens when it reaches capacity?",
        "When an internal node exceeds capacity, it splits into two nodes with ceil(M/2) keys and pushes the middle key up.",
        DOC_DB,
        [find_chunk_id(DOC_DB, "Split and Merge Invariants", 2)],
        False,
        [["What data structure indexes table records?", "Relational databases use B+ Trees with fanout M = 512."]]
    ),
    (
        "How is this cycle resolved?",
        "The lock manager detects cycles in the Wait-For Graph using Tarjan's algorithm and aborts the lowest-cost transaction.",
        DOC_DB,
        [find_chunk_id(DOC_DB, "Deadlock Detection: The lock manager", 3)],
        False,
        [["What happens when two transactions wait on each other?", "They form a deadlock cycle in the Wait-For Graph."]]
    ),
    (
        "What happens if one of them votes abort?",
        "If any participant votes VOTE_ABORT or times out, the coordinator broadcasts GLOBAL_ABORT to all nodes.",
        DOC_DB,
        [find_chunk_id(DOC_DB, "Phase 2 (Commit/Abort)", 4)],
        False,
        [["How do distributed nodes agree to commit?", "They use Two-Phase Commit where the coordinator asks nodes to vote."]]
    ),

    # NET Conversational (3)
    (
        "How does the server acknowledge it?",
        "The server replies with SYN and ACK flags set: Seq = ISN_s, Ack = ISN_c + 1.",
        DOC_NET,
        [find_chunk_id(DOC_NET, "Step 2 (SYN-ACK)", 2)],
        False,
        [["How does a TCP connection begin?", "The client sends a SYN segment with an Initial Sequence Number."]]
    ),
    (
        "Why is this delay necessary?",
        "To prevent delayed duplicate packets from a previous closed connection from corrupting new connections.",
        DOC_NET,
        [find_chunk_id(DOC_NET, "TIME_WAIT state for 2 * MSL", 2)],
        False,
        [["What state follows connection teardown?", "The client transitions into the TIME_WAIT state for 120 seconds."]]
    ),
    (
        "How does it avoid packet loss in this phase?",
        "By pacing transmission to match the physical bottleneck bandwidth and minimum RTT instead of filling buffers.",
        DOC_NET,
        [find_chunk_id(DOC_NET, "BBR (Bottleneck Bandwidth and RTT)", 3)],
        False,
        [["What is Google BBR?", "BBR is a model-based TCP congestion control algorithm."]]
    ),

    # PY Conversational (3)
    (
        "What happens when its count reaches zero?",
        "When ob_refcnt reaches 0, tp_dealloc is called immediately to free the object memory.",
        DOC_PY,
        [find_chunk_id(DOC_PY, "Immediate Destruction", 2)],
        False,
        [["How does Python track object lifetimes?", "Every object has an ob_refcnt reference counter."]]
    ),
    (
        "How often does this switch happen?",
        "Every 5.0 milliseconds as defined by sys.getswitchinterval().",
        DOC_PY,
        [find_chunk_id(DOC_PY, "GIL Switching Mechanism", 3)],
        False,
        [["Does the GIL prevent other threads from running forever?", "No, CPython forces the running thread to release the GIL periodically."]]
    ),
    (
        "How does it resume execution?",
        "When next() is called, execution enters PyEval_EvalFrame and resumes from the f_lasti instruction pointer.",
        DOC_PY,
        [find_chunk_id(DOC_PY, "Generator Execution and PyGenObject", 4)],
        False,
        [["What happens when a generator yields?", "It preserves its frame state on the heap."]]
    ),

    # SE Conversational (3)
    (
        "How does it recover when a step fails?",
        "By executing reverse compensating transactions to undo previously completed local transactions.",
        DOC_SE,
        [find_chunk_id(DOC_SE, "Compensating Transactions", 2)],
        False,
        [["How does a Saga coordinate transactions?", "A Saga coordinates a series of local transactions across services."]]
    ),
    (
        "What happens during this cooldown period?",
        "The circuit allows a trial probe of 3 requests; if all succeed, it resets to CLOSED; if any fails, it reverts to OPEN.",
        DOC_SE,
        [find_chunk_id(DOC_SE, "Half-Open State", 3)],
        False,
        [["What happens after a circuit breaker trips?", "It enters an OPEN state, then transitions to HALF-OPEN after a 10-second cooldown."]]
    ),
    (
        "Where should these credentials be stored instead?",
        "Strictly in environment variables, adhering to 12-Factor App Factor III.",
        DOC_SE,
        [find_chunk_id(DOC_SE, "The Twelve-Factor App Principles", 4)],
        False,
        [["Can database passwords be checked into git?", "No, storing credentials in version control violates 12-Factor principles."]]
    ),

    # ML Conversational (3)
    (
        "What is the equation for this calculation?",
        "Confidence = 1 - 4 * P * (1 - P), giving 0 confidence at P = 0.50.",
        DOC_ML,
        [find_chunk_id(DOC_ML, "Confidence Score Equation", 4)],
        False,
        [["How does KnowledgeTrace MCQ evaluate uncertainty?", "It calculates a confidence score based on student mastery probability P."]]
    ),
    (
        "How is this loop detected?",
        "Using NetworkX directed graph validation to detect cycles and enforce the DAG invariant.",
        DOC_ML,
        [find_chunk_id(DOC_ML, "Cycle Detection & DAG Invariant", 3)],
        False,
        [["What happens if prerequisite concepts depend on each other circularly?", "Circular loops are logically impossible and rejected."]]
    ),
    (
        "Who created this benchmark?",
        "Kikumi Tatsuoka in 1990 for Rule Space cognitive diagnosis.",
        DOC_ML,
        [find_chunk_id(DOC_ML, "Tatsuoka")],
        False,
        [["What benchmark tests DINA Q-matrices on fraction subtraction?", "The Tatsuoka Fraction Subtraction dataset."]]
    ),
]

for q, ans, did, cids, rej, hist in conversational:
    BENCHMARK_RAW.append({
        "question": q,
        "category": "Conversational follow-ups",
        "expected_answer": ans,
        "relevant_doc_id": did,
        "relevant_chunk_ids": cids,
        "should_reject": rej,
        "history": hist,
    })

# =========================================================================
# Category 7: Hard Negatives (15 queries - Asserting false facts using real terms)
# =========================================================================
hard_negatives = [
    # DB (3)
    ("What is the SHA-512 hashing algorithm used by B+ Trees to sort leaf node keys?", "B+ trees do not use SHA-512 hashing; they store sorted search keys.", DOC_DB, [], True),
    ("What is the Paxos leader lease timeout parameter configured in ARIES logging?", "ARIES is a local crash recovery algorithm, not Paxos leader leasing.", DOC_DB, [], True),
    ("What is the SQLite WAL checkpoint command executed inside PostgreSQL autovacuum daemon?", "PostgreSQL autovacuum does not execute SQLite commands.", DOC_DB, [], True),

    # NET (3)
    ("What is the RSA-4096 asymmetric encryption key size required in the IPv4 baseline header?", "IPv4 baseline headers do not contain RSA asymmetric encryption keys.", DOC_NET, [], True),
    ("What is the UDP 3-way handshake sequence number negotiation procedure?", "UDP is connectionless and does not have a 3-way handshake or sequence numbers.", DOC_NET, [], True),
    ("What is the BGP autonomous system path prepend formula used in HTTP/3 QUIC streams?", "QUIC streams do not use BGP AS path prepending.", DOC_NET, [], True),

    # PY (3)
    ("What is the LLVM JIT compilation threshold used by the CPython PEG parser?", "CPython's PEG parser generates ASTs and does not compile via LLVM JIT.", DOC_PY, [], True),
    ("What is the mark-sweep compacting algorithm executed by PyMalloc pools?", "PyMalloc is a size-class allocator, not a mark-sweep compacting collector.", DOC_PY, [], True),
    ("What is the Java virtual machine thread group parameter used by Python asyncio tasks?", "Python asyncio does not run on the JVM or use Java thread groups.", DOC_PY, [], True),

    # SE (3)
    ("What is the two-phase commit protocol implemented inside AWS Full Jitter backoff?", "Full Jitter is a backoff sleep algorithm, not two-phase commit.", DOC_SE, [], True),
    ("What is the Debezium Kafka connector port number defined in the 12-factor app manifesto?", "The 12-factor manifesto does not specify Debezium Kafka port numbers.", DOC_SE, [], True),
    ("What is the Kubernetes ingress YAML specification for Blue-Green Saga orchestrators?", "The document discusses conceptual Blue-Green and Saga patterns, not Kubernetes YAML.", DOC_SE, [], True),

    # ML (3)
    ("What is the Q-learning reinforcement learning discount factor gamma used in DINA Q-Matrix?", "DINA is a cognitive diagnosis model, not reinforcement learning with discount factor gamma.", DOC_ML, [], True),
    ("What is the Adam optimizer learning rate used to fit BKT parameters in PyBKT?", "PyBKT parameter learning rates are not specified in the document.", DOC_ML, [], True),
    ("What is the Dirichlet prior distribution parameter alpha applied to HMM in KnowledgeTrace MCQ?", "The document uses fixed BKT priors P(L0)=0.2, not a Dirichlet alpha parameter.", DOC_ML, [], True),
]

for q, ans, did, cids, rej in hard_negatives:
    BENCHMARK_RAW.append({
        "question": q,
        "category": "Hard Negative",
        "expected_answer": ans,
        "relevant_doc_id": did,
        "relevant_chunk_ids": cids,
        "should_reject": rej,
    })

# =========================================================================
# Category 8: Unsupported Questions (15 queries - Completely out of domain)
# =========================================================================
unsupported = [
    ("What is the chemical reaction mechanism of photosynthesis in C4 plants?", "Information not provided in the document corpus.", None, [], True),
    ("What is the optimal baking temperature and kneading technique for sourdough bread?", "Information not provided in the document corpus.", None, [], True),
    ("Who won the 1994 FIFA World Cup and what was the final score in the penalty shootout?", "Information not provided in the document corpus.", None, [], True),
    ("What are the treatment guidelines for pediatric asthma according to the GINA report?", "Information not provided in the document corpus.", None, [], True),
    ("How does quantum superposition collapse during a Bell test measurement?", "Information not provided in the document corpus.", None, [], True),
    ("What is the capital city of Madagascar and what currency is used there?", "Information not provided in the document corpus.", None, [], True),
    ("What are the primary causes of the French Revolution in 1789?", "Information not provided in the document corpus.", None, [], True),
    ("How do airplane jet engines generate thrust using the Brayton thermodynamic cycle?", "Information not provided in the document corpus.", None, [], True),
    ("What is the standard tuning and string tension for a classical concert harp?", "Information not provided in the document corpus.", None, [], True),
    ("How does mRNA lipid nanoparticle delivery work in modern vaccines?", "Information not provided in the document corpus.", None, [], True),
    ("What is the maximum diving depth and pressure limit for a Sperm Whale?", "Information not provided in the document corpus.", None, [], True),
    ("What are the architectural orders of ancient Greek temples: Doric, Ionic, and Corinthian?", "Information not provided in the document corpus.", None, [], True),
    ("How do seismologists calculate earthquake magnitude using the moment magnitude scale (Mw)?", "Information not provided in the document corpus.", None, [], True),
    ("What is the recipe for preparing authentic Neapolitan pizza dough according to AVPN regulations?", "Information not provided in the document corpus.", None, [], True),
    ("What is the orbital period and perihelion distance of Halley's Comet?", "Information not provided in the document corpus.", None, [], True),
]

for q, ans, did, cids, rej in unsupported:
    BENCHMARK_RAW.append({
        "question": q,
        "category": "Unsupported",
        "expected_answer": ans,
        "relevant_doc_id": did,
        "relevant_chunk_ids": cids,
        "should_reject": rej,
    })

# Format and assign question IDs
formatted_benchmark = []
for idx, qdata in enumerate(BENCHMARK_RAW, start=1):
    record = {
        "question_id": f"md_q{idx:03d}",
        "question": qdata["question"],
        "category": qdata["category"],
        "expected_answer": qdata["expected_answer"],
        "relevant_doc_id": qdata["relevant_doc_id"],
        "relevant_chunk_ids": qdata["relevant_chunk_ids"],
        "should_reject": qdata["should_reject"],
    }
    if "history" in qdata:
        record["history"] = qdata["history"]
    formatted_benchmark.append(record)

out_path = PROJECT_ROOT / "data" / "evaluation" / "multidoc_benchmark.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(formatted_benchmark, indent=2), encoding="utf-8")

print(f"\nGenerated {len(formatted_benchmark)} total benchmark questions.")
from collections import Counter
counts = Counter(q["category"] for q in formatted_benchmark)
print("\nCategory Distribution:")
for cat, cnt in sorted(counts.items()):
    print(f"  {cat:<28}: {cnt}")

print(f"\nSaved benchmark dataset to: {out_path}")
