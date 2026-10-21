import re
import ollama

MODEL_NAME = "llama3.2:3b"


def generate_answer(question, context_docs, model=MODEL_NAME):
    context = "\n\n".join(
        re.sub(
            r"(?m)^Patient:[^\n]*",
            lambda match: re.sub(r"\d+\b", "", match.group()),
            doc,
        )
        for doc in context_docs
    )

    prompt = f"""You are a data retrieval assistant. Your only task is to find and report facts that are explicitly stated in the patient records below. You are not diagnosing, treating, or advising anyone. You are simply reading the records and reporting what they say.

Patient records:
{context}

Question: {question}

Instructions:
- Answer only using facts explicitly stated in the records above.
- Do not add warnings, disclaimers, or suggestions to consult a doctor.
- If the records do not contain the answer, respond exactly with: "The records do not contain this information."
- Keep the answer to one or two sentences.

Answer:
"""

    response = ollama.generate(
        model=model,
        prompt=prompt,
        options={"temperature": 0},
    )

    return response["response"].strip()