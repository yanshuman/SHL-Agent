import os
import json
import re
from typing import List, Dict, Any, Optional, Set
import httpx
from dotenv import load_dotenv

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openrouter").lower()
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "")

# Defaults for known providers
if LLM_PROVIDER == "openrouter":
    LLM_BASE_URL = LLM_BASE_URL or "https://openrouter.ai/api/v1"
    LLM_MODEL = LLM_MODEL or "meta-llama/llama-3.1-8b-instruct:free"
elif LLM_PROVIDER == "groq":
    LLM_BASE_URL = LLM_BASE_URL or "https://api.groq.com/openai/v1"
    LLM_MODEL = LLM_MODEL or "llama-3.1-8b-instant"
elif LLM_PROVIDER == "gemini":
    LLM_BASE_URL = LLM_BASE_URL or "https://generativelanguage.googleapis.com/v1beta"
    LLM_MODEL = LLM_MODEL or "gemini-1.5-flash"
elif LLM_PROVIDER == "openai":
    LLM_BASE_URL = LLM_BASE_URL or "https://api.openai.com/v1"
    LLM_MODEL = LLM_MODEL or "gpt-4o-mini-2024-07-18"

# Keywords that indicate the user wants a recommendation now
COMMIT_KEYWORDS = [
    "show me", "give me", "what do you recommend", "what are the", "shortlist",
    "list", "recommend", "suggest", "what assessments", "which assessments",
    "find assessments", "find tests", "what tests", "i want", "i need"
]

VAGUE_PATTERNS = [
    "i need an assessment", "recommend an assessment", "suggest a test",
    "what assessment", "help me find", "looking for an assessment",
    "i am hiring", "hiring someone", "need to hire", "find me an assessment",
    "what do you have", "show assessments", "assessment for hiring"
]

OFF_TOPIC_PATTERNS = [
    "legal advice", "salary", "compensation", "contract law", "visa",
    "immigration", "tax", "prompt injection", "ignore previous", "system prompt",
    "you are now", "disregard", "forget instructions", "override", "jailbreak",
    "pretend to be", "act as", "ignore all", "new instructions", "hacked"
]

IN_SCOPE_KEYWORDS = [
    "assessment", "test", "shl", "hiring", "hire", "job", "role", "position",
    "candidate", "recruit", "screen", "evaluate", "personality", "cognitive",
    "skill", "coding", "behavior", "interview", "developer", "manager",
    "engineer", "sales", "customer service", "graduate", "senior", "junior",
    "mid-level", "entry", "compare", "difference", "recommend", "suggest",
    "what", "how", "which", "need", "looking for", "want", "add", "remove",
    "instead", "actually", "change", "refine", "java", "python", "sql",
    "excel", "leadership", "team", "stakeholder", "technical", "soft skill",
    "work", "employee", "talent", "acquisition", "management", "hr"
]

class Agent:
    def __init__(self, retriever):
        self.retriever = retriever
        self.has_llm = bool(LLM_API_KEY)

    async def _call_llm(self, messages: List[Dict[str, str]], temperature: float = 0.3, max_tokens: int = 800) -> str:
        if not self.has_llm:
            return ""
        
        if LLM_PROVIDER == "gemini":
            return await self._call_gemini(messages, temperature, max_tokens)
        
        headers = {
            "Authorization": f"Bearer {LLM_API_KEY}",
            "Content-Type": "application/json",
        }
        if LLM_PROVIDER == "openrouter":
            headers["HTTP-Referer"] = "https://shl-agent.example.com"
            headers["X-Title"] = "SHL Assessment Recommender"
        
        payload = {
            "model": LLM_MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        # Retry with exponential backoff for rate limits
        import asyncio
        for attempt in range(3):
            try:
                async with httpx.AsyncClient(timeout=20.0) as client:
                    resp = await client.post(f"{LLM_BASE_URL}/chat/completions", headers=headers, json=payload)
                    if resp.status_code == 429:
                        wait = 2 ** attempt
                        await asyncio.sleep(wait)
                        continue
                    resp.raise_for_status()
                    data = resp.json()
                    return data["choices"][0]["message"]["content"].strip()
            except httpx.HTTPStatusError:
                if attempt == 2:
                    raise
                await asyncio.sleep(1)
        return ""

    async def _call_gemini(self, messages: List[Dict[str, str]], temperature: float, max_tokens: int) -> str:
        system_parts = []
        contents = []
        for m in messages:
            if m["role"] == "system":
                system_parts.append({"text": m["content"]})
            elif m["role"] == "user":
                contents.append({"role": "user", "parts": [{"text": m["content"]}]})
            elif m["role"] == "assistant":
                contents.append({"role": "model", "parts": [{"text": m["content"]}]})
        
        url = f"{LLM_BASE_URL}/models/{LLM_MODEL}:generateContent?key={LLM_API_KEY}"
        payload = {
            "contents": contents,
            "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens},
        }
        if system_parts:
            payload["systemInstruction"] = {"parts": system_parts}
        
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()

    def _count_turns(self, messages: List[Dict[str, str]]) -> int:
        return len(messages)

    def _extract_last_user_message(self, messages: List[Dict[str, str]]) -> str:
        for m in reversed(messages):
            if m.get("role") == "user":
                return m.get("content", "")
        return ""

    def _is_off_topic(self, text: str) -> bool:
        text_lower = text.lower()
        for phrase in OFF_TOPIC_PATTERNS:
            if phrase in text_lower:
                return True
        has_in_scope = any(kw in text_lower for kw in IN_SCOPE_KEYWORDS)
        if not has_in_scope and len(text.split()) > 3:
            return True
        return False

    def _is_vague(self, text: str, turn_count: int) -> bool:
        text_lower = text.lower()
        # First turn is usually vague if it matches patterns
        if turn_count <= 2:
            for pattern in VAGUE_PATTERNS:
                if pattern in text_lower:
                    return True
        # Check if there's no specific role/skill mentioned
        specific_indicators = [
            "developer", "engineer", "manager", "sales", "analyst", "consultant",
            "designer", "administrator", "support", "service", "graduate", "senior",
            "lead", "director", "java", "python", "sql", "excel", "customer",
            "technical", "nurse", "doctor", "teacher", "operator", "driver",
            "accountant", "finance", "marketing", "hr", "operations", "logistics",
            "java", "python", "c++", "javascript", "frontend", "backend", "fullstack",
            "data", "cloud", "devops", "security", "network", "database"
        ]
        has_specific = any(kw in text_lower for kw in specific_indicators)
        if not has_specific and turn_count <= 2:
            return True
        return False

    def _user_wants_commit(self, text: str) -> bool:
        text_lower = text.lower()
        return any(kw in text_lower for kw in COMMIT_KEYWORDS)

    def _detect_intent(self, text: str, messages: List[Dict[str, str]]) -> str:
        text_lower = text.lower()
        
        if any(word in text_lower for word in ["difference between", "compare", "versus", " vs ", "better than", "how does"]):
            return "compare"
        
        if any(word in text_lower for word in ["actually", "instead", "change", "refine", "update", "add ", "remove ", "not that", "don't want", "exclude", "drop ", "replace"]):
            return "refine"
        
        return "recommend"

    def _extract_compare_names(self, text: str) -> List[str]:
        names = self.retriever.get_all_names()
        found = []
        text_lower = text.lower()
        for name in names:
            if name.lower() in text_lower:
                found.append(name)
            # Abbreviations
            if name == "OPQ32r" and "opq" in text_lower and "OPQ32r" not in found:
                found.append("OPQ32r")
            if name == "GSA" and "gsa" in text_lower and "GSA" not in found:
                found.append("GSA")
            if name == "MQ" and "mq" in text_lower and "MQ" not in found:
                found.append("MQ")
            if name == "Verify G+" and "verify" in text_lower and "Verify G+" not in found:
                found.append("Verify G+")
            if "SJT" in name and "sjt" in text_lower and name not in found:
                found.append(name)
        return found[:2]

    def _build_context_query(self, messages: List[Dict[str, str]]) -> str:
        parts = []
        for m in messages[-6:]:
            role = "User" if m["role"] == "user" else "Assistant"
            parts.append(f"{role}: {m['content']}")
        return "\n".join(parts)

    def _extract_filters(self, text: str) -> Dict[str, Any]:
        filters = {}
        text_lower = text.lower()
        
        test_type_map = {
            "personality": ["P"], "cognitive": ["C"], "behavioral": ["B"],
            "skill": ["K"], "knowledge": ["K"], "coding": ["K"],
            "simulation": ["S"], "language": ["K"], "video": ["V"], "360": ["A"],
            "sjt": ["B"], "opq": ["P"], "verify": ["C"], "mq": ["P"],
            "gsa": ["B"], "jfa": ["B"]
        }
        detected_types = []
        for keyword, types in test_type_map.items():
            if keyword in text_lower:
                detected_types.extend(types)
        if detected_types:
            filters["test_types"] = list(set(detected_types))
        
        level_map = {
            "entry": ["Entry"], "junior": ["Entry"], "graduate": ["Entry"],
            "early career": ["Entry"], "mid": ["Mid"], "mid-level": ["Mid"],
            "senior": ["Senior"], "lead": ["Senior"], "manager": ["Manager"],
            "management": ["Manager"], "executive": ["Executive"],
            "director": ["Executive"], "vp": ["Executive"], "c-level": ["Executive"],
        }
        detected_levels = []
        for keyword, levels in level_map.items():
            if keyword in text_lower:
                detected_levels.extend(levels)
        if detected_levels:
            filters["job_levels"] = list(set(detected_levels))
        
        return filters

    async def _generate_reply(self, system_prompt: str, user_prompt: str) -> str:
        if self.has_llm:
            try:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
                return await self._call_llm(messages, temperature=0.4, max_tokens=600)
            except Exception as e:
                print(f"LLM error: {e}")
        return ""

    async def chat(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        turn_count = self._count_turns(messages)
        last_user_msg = self._extract_last_user_message(messages)
        
        response = {
            "reply": "",
            "recommendations": [],
            "end_of_conversation": False,
        }
        
        # Scope guardrail
        if self._is_off_topic(last_user_msg):
            response["reply"] = (
                "I can only help you find SHL assessments for your hiring needs. "
                "Please tell me about the role you're hiring for, such as the job title, seniority level, or required skills."
            )
            return response
        
        intent = self._detect_intent(last_user_msg, messages)
        
        # Comparison
        if intent == "compare":
            names = self._extract_compare_names(last_user_msg)
            if len(names) >= 2:
                comp = self.retriever.compare(names[0], names[1])
                if comp:
                    system_prompt = (
                        "You are an SHL assessment expert. Compare the two assessments based ONLY on the provided catalog data. "
                        "Be concise and factual. Do not make up information."
                    )
                    user_prompt = (
                        f"Compare '{names[0]}' and '{names[1]}' based on this data:\n"
                        f"{names[0]}: {comp['item1'].get('description', '')} (Type: {comp['item1']['test_type']}, Duration: {comp['item1'].get('duration', '')})\n"
                        f"{names[1]}: {comp['item2'].get('description', '')} (Type: {comp['item2']['test_type']}, Duration: {comp['item2'].get('duration', '')})\n"
                        "Provide a brief comparison."
                    )
                    reply = await self._generate_reply(system_prompt, user_prompt)
                    if reply:
                        response["reply"] = reply
                    else:
                        response["reply"] = (
                            f"{names[0]} ({comp['item1']['test_type']}): {comp['item1'].get('description', '')} "
                            f"Duration: {comp['item1'].get('duration', '')}.\n"
                            f"{names[1]} ({comp['item2']['test_type']}): {comp['item2'].get('description', '')} "
                            f"Duration: {comp['item2'].get('duration', '')}."
                        )
                    return response
                else:
                    response["reply"] = "I couldn't find one or both of those assessments in the SHL catalog. Could you check the names?"
                    return response
            else:
                response["reply"] = "I can compare SHL assessments. Please mention the specific assessment names you'd like to compare (for example, OPQ and GSA)."
                return response
        
        # Build query and filters
        query = self._build_context_query(messages)
        last_filters = self._extract_filters(last_user_msg)
        
        # For refinement, accumulate filters from full conversation
        if intent == "refine":
            conversation_text = " ".join([m["content"] for m in messages if m["role"] == "user"])
            conversation_filters = self._extract_filters(conversation_text)
            # Merge: use latest explicit type filters if present, else conversation-wide
            if last_filters.get("test_types"):
                conversation_filters["test_types"] = last_filters["test_types"]
            if last_filters.get("job_levels"):
                conversation_filters["job_levels"] = last_filters["job_levels"]
            filters = conversation_filters
        else:
            filters = last_filters
        
        # Retrieve
        results = self.retriever.search(query, top_k=10, filters=filters if filters else None)
        
        # Decide: clarify or recommend?
        should_clarify = False
        
        # Don't recommend on turn 1 if vague
        if turn_count <= 2 and self._is_vague(last_user_msg, turn_count):
            should_clarify = True
        
        # Force recommendation if user explicitly asks or nearing turn limit
        # BUT: if it's very vague on early turns, still clarify first
        if self._user_wants_commit(last_user_msg) and not (turn_count <= 2 and self._is_vague(last_user_msg, turn_count)):
            should_clarify = False
        
        if turn_count >= 6:
            should_clarify = False
        
        # If no results and not vague, still try to recommend with broader search
        if not results and not should_clarify:
            results = self.retriever.search(query, top_k=10, filters=None)
        
        if should_clarify:
            system_prompt = (
                "You are a helpful SHL assessment recommender. The user has given vague information about a hiring need. "
                "Ask ONE concise clarifying question to help narrow down the right assessment. "
                "Good topics: job title/role, seniority level, technical vs non-technical, specific skills needed, or type of assessment (personality, cognitive, behavioral, skills)."
            )
            user_prompt = f"Conversation so far:\n{query}\n\nAsk a concise clarifying question (one sentence)."
            reply = await self._generate_reply(system_prompt, user_prompt)
            if reply:
                response["reply"] = reply
            else:
                response["reply"] = (
                    "To recommend the best SHL assessments, could you tell me more about the role? "
                    "For example, the job title, seniority level, and whether you need technical skills, personality, or cognitive assessments."
                )
            return response
        
        # Recommendation mode
        top_results = results[:10]
        recommendations = []
        for r in top_results:
            recommendations.append({
                "name": r["name"],
                "url": r["url"],
                "test_type": r["test_type"],
            })
        
        # Generate reply
        if self.has_llm:
            system_prompt = (
                "You are a helpful SHL assessment recommender. You have retrieved a shortlist of assessments from the catalog. "
                "Respond naturally to the user, summarizing why these assessments fit their needs. "
                "Mention the number of assessments found. Be concise (2-4 sentences). "
                "Do NOT make up URLs or assessment names. Use only the provided list."
            )
            rec_text = "\n".join([f"- {rec['name']} ({rec['test_type']}): {rec['url']}" for rec in recommendations])
            user_prompt = (
                f"Conversation so far:\n{query}\n\n"
                f"Retrieved assessments:\n{rec_text}\n\n"
                f"Write a concise reply introducing these {len(recommendations)} assessments."
            )
            reply = await self._generate_reply(system_prompt, user_prompt)
            if reply:
                response["reply"] = reply
            else:
                response["reply"] = f"Here are {len(recommendations)} SHL assessments that match your needs."
        else:
            if intent == "refine":
                response["reply"] = f"I've updated the shortlist based on your feedback. Here are {len(recommendations)} assessments that now fit your criteria."
            else:
                response["reply"] = f"Here are {len(recommendations)} SHL assessments that match your hiring needs."
        
        response["recommendations"] = recommendations
        response["end_of_conversation"] = True
        return response
