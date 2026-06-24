"""
benchmark_queries.py

A 50-query benchmark spanning the four query types (FACTUAL, REASONING,
CREATIVE, ANALYTICAL), matching the project's research-paper plan
("Run a 50-query benchmark... Metrics: answer quality score, tokens
used, escalation frequency. Compare: your system vs always-zero-shot
baseline vs always-CoT baseline.").

Each entry is (query_text, expected_query_type) so the benchmark
runner can also report classifier accuracy against this label set if
desired. Feel free to edit/extend this list — it's a reasonable
starting set covering a spread of difficulty within each type.
"""

from classifier.query_classifier import QueryType

BENCHMARK_QUERIES = [
    # ---- FACTUAL (12) ----
    ("What year was Python created?", QueryType.FACTUAL),
    ("What is the capital of France?", QueryType.FACTUAL),
    ("Who wrote the novel 1984?", QueryType.FACTUAL),
    ("What is the boiling point of water at sea level?", QueryType.FACTUAL),
    ("How many continents are there on Earth?", QueryType.FACTUAL),
    ("Define photosynthesis.", QueryType.FACTUAL),
    ("What is the chemical symbol for gold?", QueryType.FACTUAL),
    ("When did World War II end?", QueryType.FACTUAL),
    ("What is the capital of Japan?", QueryType.FACTUAL),
    ("How many bones are in the human body?", QueryType.FACTUAL),
    ("What is the speed of light in a vacuum?", QueryType.FACTUAL),
    ("Who is credited with inventing the telephone?", QueryType.FACTUAL),
    # ---- REASONING (13) ----
    ("Prove that the sum of two odd numbers is always even.", QueryType.REASONING),
    ("Prove that the square root of 2 is irrational.", QueryType.REASONING),
    ("Derive the quadratic formula from ax^2+bx+c=0.", QueryType.REASONING),
    ("Show that the product of two even numbers is even.", QueryType.REASONING),
    ("Why does increasing temperature increase reaction rate?", QueryType.REASONING),
    ("Solve for x: 2x + 5 = 17, and explain each step.", QueryType.REASONING),
    ("Calculate the area under the curve y=x^2 from 0 to 3 and show your work.", QueryType.REASONING),
    ("Why is the sky blue? Explain the underlying physics step by step.", QueryType.REASONING),
    ("Prove that there are infinitely many prime numbers.", QueryType.REASONING),
    ("Derive Newton's second law from the definition of momentum.", QueryType.REASONING),
    ("Show step by step why a triangle's angles sum to 180 degrees.", QueryType.REASONING),
    ("Solve the system of equations x+y=10 and x-y=4, step by step.", QueryType.REASONING),
    ("Why does ice float on water? Explain logically using density.", QueryType.REASONING),
    # ---- CREATIVE (12) ----
    ("Write a Tamil folk song about monsoon season.", QueryType.CREATIVE),
    ("Write a short poem about the ocean at sunset.", QueryType.CREATIVE),
    ("Compose a haiku about autumn leaves.", QueryType.CREATIVE),
    ("Write a short story about a robot learning to paint.", QueryType.CREATIVE),
    ("Imagine a city built entirely underwater and describe a day there.", QueryType.CREATIVE),
    ("Write a joke about software developers.", QueryType.CREATIVE),
    ("Brainstorm three creative names for a coffee shop.", QueryType.CREATIVE),
    ("Write a lullaby about the stars.", QueryType.CREATIVE),
    ("Compose a short script for a two-character comedy sketch.", QueryType.CREATIVE),
    ("Write a poem in the style of a monsoon evening in Chennai.", QueryType.CREATIVE),
    ("Imagine you are a tree and describe your life across four seasons.", QueryType.CREATIVE),
    ("Write a creative product description for a futuristic backpack.", QueryType.CREATIVE),
    # ---- ANALYTICAL (13) ----
    ("Compare LSTM vs Transformer architectures for NLP tasks.", QueryType.ANALYTICAL),
    ("Analyze the pros and cons of microservices versus a monolith.", QueryType.ANALYTICAL),
    ("Contrast supervised and unsupervised learning approaches.", QueryType.ANALYTICAL),
    ("Compare SQL and NoSQL databases for a high-write workload.", QueryType.ANALYTICAL),
    ("Evaluate the trade-offs between REST and GraphQL APIs.", QueryType.ANALYTICAL),
    ("Compare renewable energy sources versus fossil fuels for grid stability.", QueryType.ANALYTICAL),
    ("Analyze the differences between TCP and UDP for real-time applications.", QueryType.ANALYTICAL),
    ("Contrast agile and waterfall software development methodologies.", QueryType.ANALYTICAL),
    ("Compare Python and Rust for systems programming, considering safety and speed.", QueryType.ANALYTICAL),
    ("Evaluate the trade-offs of caching at the application layer versus the database layer.", QueryType.ANALYTICAL),
    ("Compare zero-shot and few-shot prompting for low-resource NLP tasks.", QueryType.ANALYTICAL),
    ("Analyze the pros and cons of remote work versus in-office work.", QueryType.ANALYTICAL),
    ("Contrast classical machine learning and deep learning for tabular data.", QueryType.ANALYTICAL),
]

assert len(BENCHMARK_QUERIES) == 50, f"Expected 50 benchmark queries, got {len(BENCHMARK_QUERIES)}"
