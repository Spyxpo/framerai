"""
Training data generator using locally-loaded open-source teacher LLMs.

Downloads models from HuggingFace, runs them locally, and generates
prompt-response pairs + teacher logits for knowledge distillation.
"""

import json
import logging
import os
import random
from pathlib import Path

import torch

from .teacher_models import TeacherModel

logger = logging.getLogger("framerai.distillation")

# ---------------------------------------------------------------------------
# Prompt templates for diverse training data
# ---------------------------------------------------------------------------

SYSTEM_PROMPTS = {
    "default": "You are a helpful, accurate AI assistant. Provide detailed and thoughtful responses.",
    "coder": "You are an expert programmer. Write clean, efficient, well-documented code.",
    "reasoner": "You are a logical thinker. Show your step-by-step reasoning process.",
}

DOMAIN_PROMPTS = {
    "general_knowledge": [
        "Explain {topic} in detail, covering key concepts and practical applications.",
        "What are the main differences between {topic_a} and {topic_b}?",
        "Describe the history and evolution of {topic}.",
        "What are the top 5 most important things to know about {topic}?",
        "Explain {topic} as if teaching a university course. Include examples.",
    ],
    "reasoning": [
        "Solve this step by step: {problem}",
        "Analyze the following scenario and explain your reasoning: {scenario}",
        "What would happen if {hypothesis}? Think through the implications.",
        "Compare these approaches: {approach_a} vs {approach_b}. Which is better and why?",
    ],
    "code_generation": [
        "Write a {language} function that {task}. Include error handling and comments.",
        "Implement {algorithm} in {language}.",
        "Create a {language} class for {concept} with proper methods and docstrings.",
        "Write unit tests for a {language} function that {task}.",
        "Refactor this code to be more efficient:\n```{language}\n{code_snippet}\n```",
    ],
    "instruction_following": [
        "Write a detailed tutorial on how to {process}.",
        "Create a step-by-step guide for {process}.",
        "Summarize the key points of {topic} in exactly 5 bullet points.",
        "Explain {topic} in simple terms a beginner would understand.",
    ],
    "creative": [
        "Write a short story about {creative_topic}.",
        "Write a poem about {creative_topic}.",
        "Write a dialogue between two people discussing {topic}.",
    ],
    "long_context": [
        "Write a comprehensive analysis of {topic} covering at least 10 different aspects.",
        "Create an in-depth guide about {topic} with sections, examples, and best practices.",
        "Write a detailed comparison of {topic_a} and {topic_b} across multiple dimensions.",
    ],
}

FILL_VALUES = {
    "topic": [
        "quantum computing", "machine learning", "blockchain", "neural networks",
        "distributed systems", "compiler design", "operating systems",
        "natural language processing", "reinforcement learning", "cryptography",
        "database systems", "microservices", "functional programming",
        "graph algorithms", "parallel computing", "computer vision",
        "transformers in AI", "attention mechanisms", "gradient descent",
        "convolutional neural networks", "recurrent neural networks",
        "generative adversarial networks", "transfer learning", "embeddings",
    ],
    "topic_a": ["Python", "REST APIs", "SQL databases", "monolithic architecture", "supervised learning", "TCP", "Linux"],
    "topic_b": ["Rust", "GraphQL", "NoSQL databases", "microservices", "unsupervised learning", "UDP", "FreeBSD"],
    "language": ["Python", "JavaScript", "TypeScript", "Rust", "Go", "Java", "C++"],
    "algorithm": [
        "merge sort", "binary search tree", "Dijkstra's algorithm",
        "A* pathfinding", "LRU cache", "trie", "red-black tree",
        "topological sort", "Huffman coding", "dynamic programming knapsack",
    ],
    "task": [
        "validates email addresses", "parses JSON with error recovery",
        "implements a rate limiter", "manages a thread pool",
        "handles file uploads with progress", "implements a simple HTTP server",
        "builds a CLI argument parser", "creates a connection pool",
        "computes the longest common subsequence", "implements a bloom filter",
    ],
    "concept": [
        "a linked list", "a hash map", "a binary search tree",
        "a graph with adjacency list", "a priority queue",
        "an LRU cache", "a thread-safe queue", "a state machine",
    ],
    "problem": [
        "A train travels at 60 mph for 2.5 hours, then 80 mph for 1.5 hours. What is the total distance and average speed?",
        "Given an array of integers, find the contiguous subarray with the maximum sum. Explain your approach.",
        "You have 12 balls, one is heavier. Using a balance scale at most 3 times, find the heavy ball.",
        "Design a system that handles 10,000 concurrent users with sub-100ms latency. What architecture would you use?",
    ],
    "scenario": [
        "A startup has 6 months of runway and needs to build an MVP for a collaboration tool",
        "A database migration must happen with zero downtime on a 500M row table",
        "Two microservices have a circular dependency causing cascading failures",
    ],
    "hypothesis": [
        "all programming languages were statically typed",
        "Moore's law continued indefinitely",
        "quantum computers became as common as laptops",
    ],
    "approach_a": ["microservices", "SQL", "REST API", "monorepo", "OOP"],
    "approach_b": ["monolith", "NoSQL", "GraphQL", "polyrepo", "functional programming"],
    "process": [
        "setting up a CI/CD pipeline", "deploying a ML model to production",
        "migrating to microservices", "implementing OAuth 2.0",
        "setting up Kubernetes", "building a data pipeline",
    ],
    "creative_topic": [
        "a robot discovering emotions", "the last library on Earth",
        "time travel gone wrong", "an AI that writes poetry",
    ],
    "code_snippet": [
        "def fib(n):\\n    if n<=1: return n\\n    return fib(n-1)+fib(n-2)",
        "for i in range(len(arr)):\\n    for j in range(len(arr)):\\n        if arr[i]>arr[j]: arr[i],arr[j]=arr[j],arr[i]",
    ],
}


def _fill_template(template: str) -> str:
    """Fill a prompt template with random values."""
    import re
    result = template
    for key, values in FILL_VALUES.items():
        placeholder = "{" + key + "}"
        while placeholder in result:
            result = result.replace(placeholder, random.choice(values), 1)
    result = re.sub(r"\{[^}]+\}", "[example]", result)
    return result


class DistillationDataGenerator:
    """Generates training data using a locally-loaded open-source teacher model."""

    def __init__(
        self,
        teacher: TeacherModel,
        output_dir: str = "data/distillation",
    ):
        self.teacher = teacher
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_prompts(self, domains: list = None, num_per_domain: int = 10) -> list:
        """Generate diverse prompts from templates."""
        domains = domains or list(DOMAIN_PROMPTS.keys())
        prompts = []
        for domain in domains:
            templates = DOMAIN_PROMPTS.get(domain, [])
            for _ in range(num_per_domain):
                template = random.choice(templates)
                prompt = _fill_template(template)
                prompts.append((domain, prompt))
        random.shuffle(prompts)
        return prompts

    def _format_prompt(self, prompt: str, domain: str) -> str:
        """Format prompt with chat template if the tokenizer supports it."""
        system = SYSTEM_PROMPTS.get("coder" if domain == "code_generation" else "default")
        tokenizer = self.teacher.tokenizer

        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ]
            try:
                return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                pass

        # Fallback: plain prompt
        return f"{system}\n\nUser: {prompt}\n\nAssistant:"

    def generate_dataset(
        self,
        num_samples: int = 1000,
        domains: list = None,
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        save_logits: bool = False,
        save_interval: int = 100,
    ) -> str:
        """
        Generate a full distillation dataset.

        Args:
            num_samples: Total samples to generate.
            domains: Which domains to cover.
            max_new_tokens: Max response length.
            temperature: Sampling temperature.
            save_logits: If True, also save teacher logits per sample (large files!).
            save_interval: Save partial results every N samples.

        Returns:
            Path to the saved JSONL file.
        """
        domains = domains or list(DOMAIN_PROMPTS.keys())
        num_per_domain = max(1, num_samples // len(domains))
        prompts = self.generate_prompts(domains, num_per_domain)[:num_samples]

        output_path = self.output_dir / "distillation_data.jsonl"
        logits_dir = self.output_dir / "logits" if save_logits else None
        if logits_dir:
            logits_dir.mkdir(exist_ok=True)

        logger.info(f"Generating {len(prompts)} samples with {self.teacher.name}")
        logger.info(f"Domains: {domains}")
        logger.info(f"Output: {output_path}")

        samples = []
        failed = 0

        for i, (domain, raw_prompt) in enumerate(prompts):
            try:
                formatted = self._format_prompt(raw_prompt, domain)
                response = self.teacher.generate_text(
                    formatted,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )

                sample = {
                    "prompt": raw_prompt,
                    "response": response.strip(),
                    "domain": domain,
                    "teacher_model": self.teacher.name,
                    "formatted_prompt": formatted,
                }

                # Optionally save teacher logits for true soft-label distillation
                if save_logits:
                    full_text = formatted + response
                    tok = self.teacher.tokenize(full_text, max_length=2048)
                    logits = self.teacher.get_logits(tok["input_ids"], tok["attention_mask"])
                    torch.save(logits, logits_dir / f"logits_{i:06d}.pt")
                    sample["logits_file"] = f"logits/logits_{i:06d}.pt"

                samples.append(sample)

                if (i + 1) % 10 == 0:
                    logger.info(f"  [{i + 1}/{len(prompts)}] {failed} failed")

                if save_interval and (i + 1) % save_interval == 0:
                    self._save_jsonl(samples, output_path)

            except Exception as e:
                logger.error(f"  Sample {i + 1} failed: {e}")
                failed += 1

        self._save_jsonl(samples, output_path)
        logger.info(f"Done: {len(samples)} samples saved ({failed} failed)")
        return str(output_path)

    def generate_conversation_dataset(
        self,
        num_conversations: int = 100,
        turns_per_conv: int = 4,
        max_new_tokens: int = 1024,
    ) -> str:
        """Generate multi-turn conversation data."""
        output_path = self.output_dir / "conversation_data.jsonl"
        logger.info(f"Generating {num_conversations} conversations ({turns_per_conv} turns each)")

        starters = [
            "Help me understand {topic}.",
            "I'm building a project that uses {topic}. Where should I start?",
            "Can you explain {topic} and how it relates to {topic}?",
            "I need to {task} in {language}. Can you help?",
        ]
        follow_ups = [
            "Can you elaborate on that?",
            "How would I implement that in practice?",
            "What are the pitfalls to watch out for?",
            "Give me a concrete example.",
            "How does this compare to alternatives?",
            "What would you recommend for production?",
        ]

        samples = []
        for conv_idx in range(num_conversations):
            try:
                messages = []
                user_msg = _fill_template(random.choice(starters))

                for turn in range(turns_per_conv):
                    if turn > 0:
                        user_msg = random.choice(follow_ups)
                    messages.append({"role": "user", "content": user_msg})

                    # Build context
                    tokenizer = self.teacher.tokenizer
                    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
                        try:
                            formatted = tokenizer.apply_chat_template(
                                messages, tokenize=False, add_generation_prompt=True,
                            )
                        except Exception:
                            formatted = "\n".join(f"{m['role'].title()}: {m['content']}" for m in messages) + "\nAssistant:"
                    else:
                        formatted = "\n".join(f"{m['role'].title()}: {m['content']}" for m in messages) + "\nAssistant:"

                    response = self.teacher.generate_text(formatted, max_new_tokens=max_new_tokens)
                    messages.append({"role": "assistant", "content": response.strip()})

                samples.append({
                    "conversation": messages,
                    "domain": "conversation",
                    "teacher_model": self.teacher.name,
                    "turns": len(messages) // 2,
                })

                if (conv_idx + 1) % 10 == 0:
                    logger.info(f"  [{conv_idx + 1}/{num_conversations}]")

            except Exception as e:
                logger.error(f"  Conversation {conv_idx + 1} failed: {e}")

        self._save_jsonl(samples, output_path)
        logger.info(f"Done: {len(samples)} conversations saved to {output_path}")
        return str(output_path)

    @staticmethod
    def _save_jsonl(samples: list, path):
        with open(path, "w") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

    @staticmethod
    def load_samples(path: str) -> list:
        samples = []
        with open(path) as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
        return samples
