"""Builds and ingests a multi-domain 5-document corpus for LocalRAG Phase 4.8.

Generates 4 authentic technical PDFs across distinct computer science domains:
1. Database Systems Architecture (database_systems_architecture.pdf)
2. Computer Networks Protocols (computer_networks_protocols.pdf)
3. Python Runtime Internals (python_runtime_internals.pdf)
4. Software Engineering Microservices (software_engineering_microservices.pdf)

Coexists with the existing Document 1 (machine_learning_algorithms.pdf).
Extracts text, creates page-bounded chunks, computes real nomic-embed-text embeddings,
and stores them in data/index.
"""
from __future__ import annotations

import hashlib
import json
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import fitz  # PyMuPDF

from config import settings
from src.embeddings.ollama import OllamaClient
from src.ingestion.models import Chunk, Document
from src.ingestion.pdf_extractor import extract_pdf_result
from src.ingestion.text_splitter import chunk_document
from src.storage.vector_index import VectorIndex

logger = logging.getLogger(__name__)


DOCUMENTS_CONTENT = {
    "database_systems_architecture.pdf": [
        # Page 1: Storage engine & WAL
        """DATABASE SYSTEMS ARCHITECTURE & STORAGE ENGINE SPECIFICATION
Page 1: Storage Engine Foundations, Buffer Management, and Write-Ahead Logging

1. Overview and Core Engine Structure
Modern relational database management systems (RDBMS) separate the SQL parser and query execution engine from the underlying storage manager. The storage manager is responsible for page layout on disk, buffer pool caching in main memory, and transaction durability. Standard disk pages are sized at 8 KB (PostgreSQL) or 16 KB (MySQL InnoDB) to align with hardware block boundaries.

2. Buffer Pool Management & Eviction
The buffer pool maintains an in-memory cache of disk frames organized as a page hash table. To avoid thrashing under sequential table scans, engines employ the 2Q or CLOCK-Pro eviction algorithm rather than naive LRU. The LRU-K algorithm tracks the time of the last K references (typically K=2) to differentiate between temporary sequential scans and frequently accessed hot index pages.

3. Write-Ahead Logging (WAL) & ARIES Recovery
Durability (the D in ACID) is guaranteed through the Write-Ahead Logging protocol following the ARIES recovery framework. The fundamental invariant states that log records representing changes must be flushed to non-volatile disk before the corresponding dirty data page is written to disk (WAL protocol).
The ARIES recovery algorithm operates in three distinct phases:
• Analysis Phase: Scans the log forward from the most recent checkpoint to reconstruct the Dirty Page Table (DPT) and Transaction Table (TT), determining the state of the database at the time of system crash.
• Redo Phase: Replays all logged changes forward from the smallest RecLSN in the DPT to restore the database state to the exact instant of failure (repeating history).
• Undo Phase: Scans backward from the crash point, reversing the modifications of all active transactions that never committed before the crash by writing Compensation Log Records (CLRs).""",

        # Page 2: B+ Trees & LSM-Trees
        """DATABASE SYSTEMS ARCHITECTURE & STORAGE ENGINE SPECIFICATION
Page 2: Index Architectures: B+ Tree Fanout, Leaf Node Linkages, and LSM-Trees

1. B+ Tree Internal Structure and Fanout
The B+ Tree is a self-balancing search tree where all data records or record pointers reside exclusively in leaf nodes, while internal nodes store only routing search keys and child page pointers.
• Node Fanout: For an 8 KB page storing 8-byte integer keys and 8-byte page pointers, an internal node achieves a fanout factor M = 512. A 3-level B+ Tree can address over 134 million leaf records (512^3 = 134,217,728) with at most 3 random disk I/Os.
• Leaf Linked Lists: Leaf pages are connected via bidirectional pointers (prev_page and next_page), allowing sequential range scans to proceed without traversing parent internal nodes.
• Split and Merge Invariants: When an insertion causes a node to exceed capacity (M keys), it splits into two nodes each holding ceil(M/2) keys, pushing the middle key up to the parent.

2. Log-Structured Merge-Trees (LSM-Trees)
In high-throughput write workloads, random disk writes in B+ Trees cause severe write amplification. LSM-Trees (used in RocksDB and Cassandra) convert random writes into sequential writes:
• MemTable: In-memory write buffer backed by a skip-list or red-black tree.
• Commit Log: Sequential write-ahead log for crash recovery.
• SSTables (Sorted String Tables): Immutable disk files organized in hierarchical levels (Level 0 to Level L). When Level N reaches capacity, a background compaction worker merges SSTables into Level N+1.
• Bloom Filters: Fast probabilistic bit vectors placed before each SSTable to test key set membership with a 1% false positive rate using 10 bits per key, avoiding unnecessary disk seeks for nonexistent keys.""",

        # Page 3: Concurrency Control & MVCC
        """DATABASE SYSTEMS ARCHITECTURE & STORAGE ENGINE SPECIFICATION
Page 3: Concurrency Control, Two-Phase Locking, and Multi-Version Concurrency (MVCC)

1. Two-Phase Locking (2PL) and Deadlock Handling
Serializability is guaranteed through Strict Two-Phase Locking (Strict 2PL). A transaction acquires locks during its Growing Phase and cannot release any lock until commit or abort (Shrinking Phase):
• Shared Locks (S-Locks): Required for read operations; multiple transactions may hold shared locks on the same tuple simultaneously.
• Exclusive Locks (X-Locks): Required for write operations; conflicts with both S-Locks and X-Locks.
• Deadlock Detection: The lock manager runs a background cycle-detection algorithm on a directed Wait-For Graph (WFG) every 500 ms using Tarjan's strongly connected components algorithm. When a deadlock cycle (T1 -> T2 -> T1) is detected, the transaction with the lowest cost or shortest runtime is chosen as victim and aborted with a serialization error.

2. Multi-Version Concurrency Control (MVCC)
To ensure that readers do not block writers and writers do not block readers, modern databases use MVCC.
• Tuple Headers: In PostgreSQL, every tuple header contains xmin (the transaction ID that created the tuple) and xmax (the transaction ID that deleted or updated the tuple, initially set to 0).
• Read Views: When Transaction T with ID 105 begins a snapshot read, it creates a snapshot containing the active transaction list [102, 104]. A tuple is visible to T if its xmin is committed and not in the active list.
• Vacuuming: Obsolete tuple versions whose xmax is older than all running transactions are dead tuples reclaimed by the background autovacuum daemon to prevent table bloat.""",

        # Page 4: Distributed Databases & Consensus
        """DATABASE SYSTEMS ARCHITECTURE & STORAGE ENGINE SPECIFICATION
Page 4: Distributed Databases: Partitioning, Raft Consensus, and Two-Phase Commit

1. Data Partitioning and Consistent Hashing
Large datasets are horizontally partitioned across multiple physical nodes. Consistent hashing arranges nodes in an abstract 2^32 - 1 ring using the MurmurHash3 algorithm. To guarantee uniform data distribution and avoid hotspots, each physical server is mapped to 256 virtual nodes (vnodes) across the ring.

2. The Raft Consensus Protocol
Distributed state machine replication relies on the Raft consensus algorithm (Ongaro & Ousterhout, 2014):
• Leader Election: Nodes start as Followers. If a follower receives no heartbeat from a Leader within a randomized election timeout between 150 ms and 300 ms, it transitions to Candidate and broadcasts RequestVote RPCs.
• Quorum Invariant: A candidate becomes Leader only after receiving votes from a strict majority quorum: Quorum = floor(N/2) + 1. For a 5-node cluster, 3 votes are required.
• Log Replication: The Leader accepts client commands, appends entries to its local log, and broadcasts AppendEntries RPCs. An entry is committed once replicated across a majority of nodes.

3. Distributed Transactions (Two-Phase Commit - 2PC)
When a transaction spans multiple partitions, atomicity across nodes requires Two-Phase Commit:
• Phase 1 (Prepare): The Coordinator sends a PREPARE message to all Participant nodes. Each participant executes changes locally, flushes WAL records, and replies VOTE_COMMIT or VOTE_ABORT.
• Phase 2 (Commit/Abort): If all participants voted VOTE_COMMIT, the coordinator logs a GLOBAL_COMMIT record and broadcasts COMMIT. If any node voted abort or timed out, GLOBAL_ABORT is broadcast."""
    ],

    "computer_networks_protocols.pdf": [
        # Page 1: OSI vs TCP/IP & Encapsulation
        """COMPUTER NETWORKS & INTERNETWORKING PROTOCOLS SPECIFICATION
Page 1: Architecture Models, Packet Encapsulation, and Layer 2/3 Fundamentals

1. Layered Network Architecture
The Internet architecture maps the theoretical 7-layer OSI model into the practical 4-layer TCP/IP suite:
• Application Layer (Layer 7): HTTP/HTTPS, DNS, SSH, SMTP, and gRPC.
• Transport Layer (Layer 4): End-to-end process communication via TCP (connection-oriented, reliable) and UDP (connectionless, datagram).
• Internet/Network Layer (Layer 3): Logical host addressing and global packet routing via IPv4 and IPv6.
• Link Layer (Layer 2): Hardware framing and node-to-node physical transmission via Ethernet (IEEE 802.3) and Wi-Fi (IEEE 802.11).

2. Packet Encapsulation and Header Overhead
As user application data moves down the protocol stack, each layer prepends a protocol header:
• Ethernet Header: 14 bytes (6-byte destination MAC, 6-byte source MAC, 2-byte EtherType) plus a 4-byte Frame Check Sequence (FCS) trailer.
• IPv4 Header: 20 bytes minimum without options (includes Version, IHL, Total Length, TTL, Protocol 6 for TCP, Source IP, and Destination IP).
• TCP Header: 20 bytes minimum without options (includes Source Port, Destination Port, 32-bit Sequence Number, 32-bit Acknowledgment Number, Data Offset, Flags, and Window Size).
• Total Baseline Header: 14 + 20 + 20 = 54 bytes of encapsulation overhead.
• Maximum Transmission Unit (MTU): Standard Ethernet enforces an MTU of 1500 bytes. Subtracting the 40-byte IP+TCP headers yields a Maximum Segment Size (MSS) of 1460 bytes. Data exceeding MSS undergoes IP fragmentation or TCP segmentation.""",

        # Page 2: TCP Handshake & Flow Control
        """COMPUTER NETWORKS & INTERNETWORKING PROTOCOLS SPECIFICATION
Page 2: TCP Connection Lifecycle, Sliding Windows, and Reliable Delivery

1. TCP Three-Way Handshake Connection Establishment
TCP establishes a reliable full-duplex byte stream via a three-way handshake:
• Step 1 (SYN): The client selects a randomized Initial Sequence Number (ISN_c) and sends a TCP segment with the SYN flag set: Seq = ISN_c, ACK = 0.
• Step 2 (SYN-ACK): The server allocates connection transmission control blocks (TCBs), generates its own sequence number ISN_s, and replies with SYN and ACK flags set: Seq = ISN_s, Ack = ISN_c + 1.
• Step 3 (ACK): The client confirms connection establishment: Seq = ISN_c + 1, Ack = ISN_s + 1. At this instant, the connection enters the ESTABLISHED state.

2. TCP Teardown and the TIME_WAIT State
Connection termination uses a 4-step handshake with FIN flags:
• Client sends FIN (enters FIN_WAIT_1); Server replies ACK (enters CLOSE_WAIT).
• Server finishes transmission and sends FIN (enters LAST_ACK).
• Client acknowledges FIN and transitions to the TIME_WAIT state for 2 * MSL (Maximum Segment Lifetime), standardly defined as 120 seconds. This prevents delayed duplicate segments from a closed connection from corrupting subsequent connections on the same port tuple.

3. Flow Control and the Sliding Window Protocol
TCP flow control prevents a fast sender from overwhelming a slow receiver. The receiver advertises its available buffer space in the 16-bit Window Size header field (rcv_wnd). The Window Scale option (RFC 1323) introduces a scale factor up to 14 bits, allowing window sizes up to 1 Gigabyte (65,535 * 2^14 bytes).""",

        # Page 3: Congestion Control Algorithms
        """COMPUTER NETWORKS & INTERNETWORKING PROTOCOLS SPECIFICATION
Page 3: TCP Congestion Control: AIMD, Cubic Curve, and BBR Model Pacing

1. Principles of Network Congestion Control
While flow control protects the receiver, congestion control protects the shared network transit path from packet bufferbloat and collapse. TCP maintains a Congestion Window (cwnd), and the sender cannot have more than min(cwnd, rcv_wnd) unacknowledged bytes in flight.

2. The Classic AIMD Algorithm
• Slow Start: Starts with cwnd = initial window (typically 10 MSS). For each received ACK, cwnd increases exponentially (cwnd = cwnd + 1 MSS per ACK) until reaching the slow start threshold (ssthresh).
• Congestion Avoidance: Once cwnd >= ssthresh, the algorithm switches to Additive Increase, incrementing cwnd by 1 MSS per Round-Trip Time (RTT).
• Multiplicative Decrease: Upon packet loss detected via 3 duplicate ACKs (Triple Duplicate ACK), ssthresh is cut in half: ssthresh = cwnd / 2.

3. Cubic and BBR Modern Algorithms
• TCP Cubic (RFC 8312): Standard in Linux and Windows kernels. Replaces linear increase with a cubic function of elapsed time t since the last congestion event: W_cubic(t) = C * (t - K)^3 + W_max, where C is a scaling constant (0.4) and K = (W_max * beta / C)^(1/3). This achieves rapid recovery in high-bandwidth long-distance networks.
• BBR (Bottleneck Bandwidth and RTT): Developed by Google. Unlike loss-based algorithms that wait for packet drops, BBR builds an explicit physical model of the network bottleneck bandwidth and minimum round-trip propagation time (RTprop), pacing transmission to prevent queue formation at intermediate routers.""",

        # Page 4: DNS, HTTP/2, and HTTP/3 QUIC
        """COMPUTER NETWORKS & INTERNETWORKING PROTOCOLS SPECIFICATION
Page 4: Application Protocols: DNS Hierarchy, HTTP/2 Multiplexing, and HTTP/3 QUIC

1. Domain Name System (DNS) Architecture
DNS resolves human-readable domain names into IP addresses through a distributed tree hierarchy:
• Root DNS Servers: 13 named root authority systems (A through M) distributed worldwide via Anycast IP routing.
• Top-Level Domain (TLD) Servers: Manage generic (.com, .org) and country-code (.uk, .de) extensions.
• Authoritative Name Servers: Hold the official resource records (A for IPv4, AAAA for IPv6, CNAME for alias, MX for mail, TXT for domain verification).
• Time-To-Live (TTL): Specifies the duration in seconds that recursive resolvers may cache records before initiating fresh DNS queries.

2. HTTP/2 Binary Framing & Stream Multiplexing (RFC 7540)
HTTP/1.1 suffered from Head-of-Line (HoL) blocking because requests on a TCP connection were strictly serialized. HTTP/2 replaced plain text with a binary framing layer:
• Independent Streams: Multiple bidirectional streams (each tagged with an odd or even 31-bit stream ID) interleave simultaneously over a single underlying TCP connection.
• HPACK Compression: Compresses repeated HTTP request headers using a static table of 61 common headers and a sliding dynamic table.

3. HTTP/3 over QUIC Protocol (RFC 9000)
While HTTP/2 eliminated HTTP-level HoL blocking, a single dropped TCP packet still stalled all interleaved streams at the TCP kernel buffer. HTTP/3 resolves this by running over QUIC (Quick UDP Internet Connections):
• Transport: Operates over UDP on port 443.
• Independent Stream Loss Recovery: A dropped packet on Stream 3 does not delay delivery of data on Stream 7.
• 0-RTT Connection Resumption: Combines cryptographic TLS 1.3 handshake with transport parameters in the very first round trip, allowing returning clients to send encrypted request payloads immediately."""
    ],

    "python_runtime_internals.pdf": [
        # Page 1: CPython compilation & bytecode
        """PYTHON RUNTIME INTERNALS & CPYTHON VIRTUAL MACHINE SPECIFICATION
Page 1: Compilation Pipeline, Bytecode Architecture, and Tier-2 Quickening

1. CPython Execution and Parsing Pipeline
CPython is the reference implementation of Python written in C. When a Python source file (.py) executes, it passes through four discrete phases:
• Lexical Analysis & Parsing: The PEG parser (introduced in Python 3.9, implemented in Parser/pegen.c) converts source tokens into a Concrete Parse Tree.
• Abstract Syntax Tree (AST): The parse tree transforms into an AST representation containing grammar nodes (ast.Module, ast.FunctionDef, ast.BinOp).
• Symbol Table Generation: Analyzes variable scopes (local, non-local, global, cell variables for closures).
• Bytecode Compilation: Compiles the AST into an immutable code object (types.CodeType) containing the co_code byte array, co_consts tuple, and co_names identifier list.

2. Bytecode Instruction Format (Wordcode)
Starting with Python 3.6, CPython instructions use a fixed 16-bit wordcode format: 8 bits for the opcode (0 to 255) and 8 bits for the oparg. If an argument exceeds 255, the compiler prepends one or more EXTENDED_ARG instructions.
Common opcodes include:
• LOAD_FAST: Loads a local variable from the frame array in O(1) time.
• LOAD_GLOBAL: Searches module globals followed by the builtins dictionary.
• BINARY_OP: Executes binary arithmetic dispatching to slot functions (nb_add).
• CALL: Invokes a callable object, pushing a new PyFrameObject onto the call stack.

3. Specializing Adaptive Interpreter (PEP 659 Quickening)
Introduced in Python 3.11 as part of the Faster CPython initiative. The runtime monitors bytecode execution. If an instruction is executed repeatedly with identical operand types, it dynamically mutates into a specialized instruction (e.g., LOAD_ATTR transforms into LOAD_ATTR_MODULE or LOAD_ATTR_INSTANCE_VALUE), eliminating dictionary lookups and achieving a 10% to 25% speedup without ahead-of-time compilation.""",

        # Page 2: Memory & Garbage Collection
        """PYTHON RUNTIME INTERNALS & CPYTHON VIRTUAL MACHINE SPECIFICATION
Page 2: Memory Allocator Hierarchy (PyMalloc) and Generational Garbage Collection

1. The CPython Memory Allocator Hierarchy
To avoid operating system malloc fragmentation for millions of small objects, CPython employs a 4-layer memory hierarchy:
• Layer 0 (System Allocator): Calls standard glibc malloc/free for allocations exceeding 512 bytes.
• Layer 1 (Python Core Allocator): PyMem_RawMalloc for internal raw buffers.
• Layer 2 (PyMalloc Object Allocator): Optimized small-object allocator for sizes <= 512 bytes:
  - Arenas: 256 KB aligned memory segments allocated from the OS.
  - Pools: 4 KB subdivisions within an arena; each pool is dedicated to a single fixed size-class (multiples of 8 bytes from 8 to 512 bytes).
  - Blocks: The fundamental allocation units carved out of pools.

2. Reference Counting and Garbage Collection (gc module)
Every Python object (PyObject) begins with a standard header: ob_refcnt (reference count) and ob_type (pointer to type descriptor PyTypeObject).
• Immediate Destruction: When ob_refcnt reaches 0, the object's tp_dealloc method is invoked immediately, freeing memory without latency.
• Cyclic Garbage Collector: Reference counting cannot collect circular references (Object A points to Object B, and Object B points to Object A). CPython runs a tri-color generational cycle detector (gc module) across 3 generation rings:
  - Generation 0: Holds newly allocated container objects. Scanned when allocations minus deallocations exceed threshold 0 (default 700).
  - Generation 1: Intermediate objects that survived a Gen 0 scan. Scanned when Gen 0 collections exceed threshold 1 (default 10).
  - Generation 2: Long-lived objects (modules, classes). Scanned when Gen 1 collections exceed threshold 2 (default 10).""",

        # Page 3: Concurrency & GIL
        """PYTHON RUNTIME INTERNALS & CPYTHON VIRTUAL MACHINE SPECIFICATION
Page 3: Concurrency, the Global Interpreter Lock (GIL), and PEP 703 Free-Threading

1. The Global Interpreter Lock (GIL) Architecture
The GIL is a mutual exclusion mutex (PyThread_type_lock) implemented in Python/ceval_gil.h that prevents multiple native OS threads from executing CPython bytecode simultaneously.
• Motivation: CPython's memory manager is non-thread-safe. Without the GIL, concurrent modifications to ob_refcnt and PyMalloc memory pools would trigger catastrophic race conditions.
• GIL Switching Mechanism: To enable cooperative multi-threading, CPython forces the running thread to release the GIL every 5.0 milliseconds (configurable via sys.getswitchinterval()).
• I/O Non-Blocking Behavior: Long-running I/O system calls (socket recv, file read) and external C extensions (NumPy vector matrix multiplication, PyMuPDF rendering) explicitly release the GIL via Py_BEGIN_ALLOW_THREADS and reacquire it via Py_END_ALLOW_THREADS, allowing true multi-core concurrency during I/O and numeric processing.

2. PEP 703 Free-Threaded Python (No-GIL Python)
Python 3.13 introduces experimental support for running without the Global Interpreter Lock (PEP 703):
• Mimalloc Allocator: Replaces PyMalloc with the thread-safe Microsoft mimalloc memory allocator.
• Biased Reference Counting (BRC): Differentiates between the thread that created an object (local thread) and foreign threads. Local increments avoid expensive atomic CPU instructions.
• Immortal Objects: Frequently accessed singletons (None, True, False, small integers -5 to 256) have their reference count flag set to immortal, eliminating all atomic contention.""",

        # Page 4: Descriptors, Generators & Asyncio
        """PYTHON RUNTIME INTERNALS & CPYTHON VIRTUAL MACHINE SPECIFICATION
Page 4: Descriptors, Generator State Machines, and Asyncio Event Loop Mechanics

1. The Descriptor Protocol
Descriptors are the foundational mechanism powering properties, methods, classmethods, and staticmethods. A descriptor is any object defining at least one protocol method:
• __get__(self, instance, owner): Retrieves the attribute value.
• __set__(self, instance, value): Mutates the attribute.
• __delete__(self, instance): Deletes the attribute.
• Data Descriptors: Define both __get__ and __set__. Take precedence over instance __dict__ lookups.
• Non-Data Descriptors: Define only __get__ (e.g. regular Python functions). Instance __dict__ overrides non-data descriptors.

2. Generator Execution and PyGenObject
Generators are resumable stack frames. Calling a generator function does not execute code; it allocates a PyGenObject wrapping a PyFrameObject:
• State Machine: When next() is called, execution enters PyEval_EvalFrame, advancing the instruction pointer f_lasti until hitting YIELD_VALUE. The yielded object is returned, and frame state (local variable array and value stack) is preserved in heap memory.

3. Asyncio Event Loop Mechanics
Asynchronous concurrency in Python operates via cooperative multitasking over an event loop:
• Coroutines: Built on top of generator infrastructure using async/await syntax (PEP 492). A coroutine function returns a PyCoroObject.
• Task Objects: Wraps a coroutine as an asyncio.Task, scheduling it on the running event loop.
• I/O Demultiplexing: The event loop utilizes OS-specific notification primitives (epoll on Linux, kqueue on macOS, I/O Completion Ports on Windows) to monitor file descriptors without busy-polling, resuming waiting coroutines when sockets become readable."""
    ],

    "software_engineering_microservices.pdf": [
        # Page 1: Microservices vs Monoliths & Bounded Contexts
        """SOFTWARE ENGINEERING & CLOUD-NATIVE ARCHITECTURE SPECIFICATION
Page 1: Architectural Decomposition: Modular Monoliths vs Microservices

1. Monolithic vs Microservices Tradeoffs
Modern software architecture balances operational complexity against organizational scalability:
• Modular Monolith: Single deployment unit where domain modules maintain strict code boundaries and public interfaces. Minimizes deployment overhead, simplifies ACID database transactions, and avoids network latency serialization costs.
• Microservices Architecture: Decentralized distributed systems where each business domain runs as an independent deployable service with isolated data storage.
• Conway's Law: Organizations design systems that mirror their communication structures. Independent two-pizza teams operate autonomously by owning microservices end-to-end.

2. Domain-Driven Design (DDD) & Bounded Contexts
Microservices boundaries must be delineated using Domain-Driven Design principles:
• Bounded Context: An explicit boundary within which a domain model applies. For example, in an e-commerce platform, "Customer" in the Identity Context represents authentication credentials, while "Customer" in the Billing Context represents tax ID and payment tokens.
• Database-per-Service Pattern: Each service owns its dedicated private database schema (e.g., Order Service owns PostgreSQL; Recommendation Service owns Redis). External services cannot query the database directly and must communicate via REST, gRPC, or asynchronous message events.""",

        # Page 2: Saga Pattern & Outbox
        """SOFTWARE ENGINEERING & CLOUD-NATIVE ARCHITECTURE SPECIFICATION
Page 2: Distributed Data Consistency: Saga Pattern and Transactional Outbox

1. The Saga Pattern for Distributed Transactions
Because microservices reject shared databases, distributed transactions cannot use ACID locks across services. The Saga pattern coordinates long-running business workflows as a sequence of local transactions:
• Choreography-Based Saga: Services publish domain events to an event bus (Apache Kafka, RabbitMQ). Other services listen and react autonomously. Suitable for simple 2 to 3 step workflows.
• Orchestration-Based Saga: A central Orchestrator service (e.g. Temporal, AWS Step Functions) manages state machines, invoking each service via command messages.
• Compensating Transactions: If Step 3 (Payment Service) fails, the orchestrator executes reverse compensating transactions: cancels order in Step 2 and releases inventory reservation in Step 1.

2. The Transactional Outbox Pattern
A common distributed systems pitfall is dual-write inconsistency: saving an entity to a SQL database and publishing an event to Kafka in separate operations. If the application crashes between the two, data becomes corrupt.
• Solution (Transactional Outbox): The application saves the business entity AND inserts an event record into a local outbox table inside a single local ACID transaction.
• Change Data Capture (CDC): A background relay engine (Debezium reading PostgreSQL WAL logs) streams outbox records to Kafka with guaranteed at-least-once delivery, ensuring zero message loss.""",

        # Page 3: Fault Tolerance & Circuit Breakers
        """SOFTWARE ENGINEERING & CLOUD-NATIVE ARCHITECTURE SPECIFICATION
Page 3: Distributed Resilience: Circuit Breakers, Bulkheads, and Exponential Jitter

1. The Circuit Breaker Pattern (Martin Fowler)
In a distributed network, cascading failures occur when Service A repeatedly calls an unavailable Service B, exhausting thread pools and sockets. The Circuit Breaker wraps network calls in a state machine:
• Closed State: Requests pass through normally. A sliding window tracks success and error rates.
• Open State: If error rate exceeds the failure threshold (e.g. 50% failures over 20 requests), the circuit trips to OPEN. All subsequent requests fail immediately without making network calls, preventing server overload.
• Half-Open State: After a cooldown timeout (e.g. 10 seconds), the circuit transitions to HALF-OPEN, allowing a trial probe of 3 requests. If all probe requests succeed, the circuit resets to CLOSED; if any fails, it reverts to OPEN.

2. Bulkhead Isolation Pattern
Isolates critical resource pools (connection pools, thread executors) so that failure in an auxiliary component (e.g. email notification generation) cannot exhaust worker threads dedicated to mission-critical workflows (e.g. checkout payment processing).

3. Exponential Backoff with Full Jitter
When retrying transient network errors, deterministic backoff triggers retry storms. The standard AWS Full Jitter algorithm randomizes backoff sleep intervals:
Sleep = random(0, min(Max_Backoff, Base_Backoff * 2^attempt))
This spreads retry traffic uniformly across time, preventing thundering herd congestion at recovery.""",

        # Page 4: Observability, 12-Factor, & Zero-Downtime
        """SOFTWARE ENGINEERING & CLOUD-NATIVE ARCHITECTURE SPECIFICATION
Page 4: Cloud-Native Delivery: The 12-Factor Methodology, OpenTelemetry, and Deployments

1. The Twelve-Factor App Principles (Heroku)
Cloud-native applications adhere to the 12-factor standard:
• Factor III (Config): Store all configuration (database URLs, API credentials) strictly in environment variables, never checked into version control.
• Factor VI (Stateless Processes): Execute app as one or more stateless processes; share-nothing architecture with state persisted to backing datastores.
• Factor IX (Disposability): Fast startup (< 5 seconds) and graceful shutdown upon SIGTERM, completing in-flight HTTP requests before exit.

2. Distributed Tracing & OpenTelemetry (W3C TraceContext)
In distributed microservice graphs, debugging latency requires tracing requests across service boundaries:
• Trace ID: A 16-byte globally unique identifier identifying the entire end-to-end transaction.
• Span ID: An 8-byte identifier representing an individual unit of work within a service.
• Context Propagation: Transmitted across HTTP headers using the W3C traceparent standard:
  traceparent: 00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01

3. Zero-Downtime Deployment Strategies
• Blue-Green Deployment: Runs two identical production environments. Blue serves 100% active traffic while Green receives the new release. Following smoke testing, the load balancer router switches traffic instantly to Green.
• Canary Releases: Routes a small fraction of real user traffic (e.g. 5%) to the new version while monitoring error rates, p99 latency, and CPU metrics before promoting to 100%."""
    ]
}


def build_and_ingest_multidoc_corpus():
    docs_dir = settings.documents_dir
    index_dir = settings.index_dir
    docs_dir.mkdir(parents=True, exist_ok=True)
    index_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("BUILDING 4 NEW DOMAIN-SPECIFIC PDF DOCUMENTS FOR PHASE 4.8")
    print("=" * 80)

    # 1. Generate the PDF files
    generated_files = []
    for filename, pages_text in DOCUMENTS_CONTENT.items():
        pdf_path = docs_dir / filename
        doc = fitz.open()
        for page_idx, text_content in enumerate(pages_text):
            page = doc.new_page(width=612, height=792)  # Standard Letter size
            # Insert text in a clean formatted rect
            rect = fitz.Rect(50, 50, 562, 742)
            page.insert_textbox(rect, text_content, fontsize=9.5, fontname="helv", align=0)
        doc.save(pdf_path)
        doc.close()
        generated_files.append(pdf_path)
        print(f"Generated PDF: {pdf_path.name} ({len(pages_text)} pages, {pdf_path.stat().st_size} bytes)")

    # 2. Ingest into VectorIndex
    print("\n" + "=" * 80)
    print("INGESTING AND VECTORIZING MULTI-DOCUMENT CORPUS VIA OLLAMA")
    print("=" * 80)

    client = OllamaClient(settings.ollama_base_url, timeout=60.0)
    if not client.is_available():
        raise RuntimeError(f"Ollama is not reachable on {settings.ollama_base_url}")

    # Load existing index (which already has Document 1)
    index = VectorIndex.load(index_dir, settings.embedding_model, settings.chunk_size, settings.chunk_overlap)
    print(f"Loaded existing index with {len(index.chunks)} chunks.")

    doc_registry = {}
    # Record existing document 1
    for c in index.chunks:
        doc_registry[c.doc_id] = {
            "filename": c.source_filename,
            "doc_id": c.doc_id,
        }

    for pdf_path in generated_files:
        filename = pdf_path.name
        content = pdf_path.read_bytes()
        doc_fingerprint = hashlib.sha256(content).hexdigest()

        # Check if already in index
        if any(c.doc_id == doc_fingerprint for c in index.chunks):
            print(f"Document {filename} (ID {doc_fingerprint[:12]}...) is already indexed. Skipping.")
            continue

        result = extract_pdf_result(pdf_path)
        doc_id = result.doc_id or doc_fingerprint
        print(f"\nProcessing {filename} (Doc ID: {doc_id[:16]}...):")
        print(f"  Extracted pages: {len(result.pages)}")

        doc_obj = Document(doc_id=doc_id, source_filename=filename, pages=result.pages)
        chunks = chunk_document(doc_obj, settings.chunk_size, settings.chunk_overlap)
        print(f"  Generated chunks: {len(chunks)}")

        # Verify page numbers
        for chk in chunks:
            assert chk.page_number >= 1, f"Invalid page number in {chk.chunk_id}"
            assert chk.doc_id == doc_id, f"Doc ID mismatch in {chk.chunk_id}"

        # Generate embeddings in batch via Ollama
        print(f"  Embedding {len(chunks)} chunks via Ollama nomic-embed-text...")
        t0 = time.perf_counter()
        texts = [c.text for c in chunks]
        vectors = client.embed_batch(texts, settings.embedding_model)
        t1 = time.perf_counter()
        print(f"  Embedded in {(t1 - t0)*1000:.1f} ms ({(t1 - t0)*1000 / len(chunks):.1f} ms/chunk)")

        # Add to index
        index.add(chunks, vectors)
        print(f"  Added {len(chunks)} chunks to VectorIndex.")

    # Save expanded index
    index.save()
    print(f"\nSaved updated VectorIndex to {index_dir}. Total indexed chunks: {len(index.chunks)}")

    # 3. Print Summary of All 5 Documents
    from collections import defaultdict
    doc_chunks = defaultdict(list)
    for c in index.chunks:
        doc_chunks[c.doc_id].append(c)

    print("\n" + "=" * 90)
    print(f"{'DOCUMENT FILENAME':<38} | {'DOC ID (SHA-256)':<16} | {'PAGES':<6} | {'CHUNKS':<7}")
    print("-" * 90)
    for did, chks in doc_chunks.items():
        fname = chks[0].source_filename
        max_page = max(c.page_number for c in chks)
        print(f"{fname:<38} | {did[:16]}... | {max_page:<6} | {len(chks):<7}")
    print("=" * 90)


if __name__ == "__main__":
    build_and_ingest_multidoc_corpus()
