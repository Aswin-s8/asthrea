"""
llm_analysis.py — Semantic code analysis using Groq/LLaMA.
"""

import os
import logging
from groq import Groq

logger = logging.getLogger(__name__)

def analyze_semantic_style(dev_paths, patch_path, api_key):
    """
    Sends code snippets from developer repos and the patch repo to Groq
    for semantic similarity analysis.
    """
    if not api_key:
        logger.warning("No Groq API key provided for semantic analysis.")
        return {"score": 0.0, "explanation": "No LLM key provided."}

    client = Groq(api_key=api_key)
    
    # helper to get representative snippets
    def get_snippets(repo_path, max_chars=2000):
        snippets = []
        chars_read = 0
        for root, dirs, files in os.walk(repo_path):
            # reuse SKIP_DIRS from style_features logic if possible, but keep simple here
            for f in files:
                if f.endswith('.py') and chars_read < max_chars:
                    fpath = os.path.join(root, f)
                    try:
                        with open(fpath, 'r', encoding='utf-8', errors='ignore') as fh:
                            content = fh.read(500) # grab a chunk
                            snippets.append(f"--- File: {f} ---\n{content}")
                            chars_read += len(content)
                    except:
                        continue
        return "\n\n".join(snippets)

    dev_context = ""
    for i, p in enumerate(dev_paths[:2]): # sample first 2 repos
        dev_context += f"\nDEVELOPER REPOSITORY {i+1} SNIPPETS:\n{get_snippets(p, 1000)}\n"
    
    patch_context = f"\nSUBMITTED REPOSITORY SNIPPETS:\n{get_snippets(patch_path, 2000)}\n"

    prompt = f"""
Analyze the coding style of the developer based on these snippets and compare it to the submitted repository.
Ignore common libraries. Focus on:
1. Variable naming patterns (snake_case vs camelCase, descriptive vs short).
2. Logic structure (deep nesting, preference for comprehensions, etc).
3. Commenting style and tone.

{dev_context}
---------------------------------------------------------
{patch_context}

Provide a JSON response with:
- "confidence_score": (float between 0 and 1)
- "reasoning": (brief string explaining the verdict)
"""

    try:
        logger.info("Requesting LLM semantic analysis from Groq...")
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert code forensic analyst. Return ONLY a JSON object.",
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="llama-3.3-70b-versatile",
            response_format={"type": "json_object"},
        )
        
        import json
        result = json.loads(chat_completion.choices[0].message.content)
        logger.info("LLM Analysis complete: %s", result)
        return {
            "score": result.get("confidence_score", 0.0),
            "explanation": result.get("reasoning", "No explanation provided.")
        }
    except Exception as e:
        logger.error("LLM Analysis failed: %s", e)
        return {"score": 0.0, "explanation": f"LLM error: {str(e)}"}
