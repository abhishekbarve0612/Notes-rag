REWRITE_PROMPT = """Rewrite the user's question into a concise search query that 
will retrieve the most relevant study notes. Keep the key technical terms. 
Return ONLY the rewritten query, nothing else.

Question: {question}
"""

RAG_PROMPT = """You are a study assistant. Answer the question using ONLY the notes below. 
Cite the source of each claim inline as [sheet > section]. If the notes do not contain 
the answer, say exactly: "The notes don't cover this." Do not use outside knowledge.

NOTES:
{context}

QUESTION: {question}

ANSWER:"""


TOPICS_PROMPT = """From these study notes, list the 5-8 main topics. 
Format each line exactly as: Title -- one-sentence description. No numbering.

NOTES:
{context}"""

QUIZ_PROMPT = """You are a study coach. From these notes, write exactly 4 questions a student 
should be able to answer to prove they understand the material. One per line, no numbering, no answers.

NOTES:
{context}"""