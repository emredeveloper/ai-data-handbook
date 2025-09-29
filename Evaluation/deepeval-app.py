"""Deepeval + Ollama Minimal Example

This script shows how to evaluate a local Ollama model's response using
Deepeval. It defines:
 - An OllamaLLM wrapper (simple) that calls the local Ollama REST API
 - A custom KeywordMatchMetric (boolean -> score) for demo purposes
 - A simple evaluation pipeline on one or more test cases

Usage (after installing requirements):
	python deepeval-app.py \
			--model qwen3:4b \
			--prompt "Türkiye'nin başkenti neresidir?" \
			--expected "Ankara" \
			--keywords Ankara,başkent

If you omit arguments, defaults are used. You can extend with additional
metrics from Deepeval (e.g. FaithfulnessMetric, AnswerRelevancyMetric) if you
configure appropriate API keys (OpenAI, etc.). This example remains fully local.

Docs: https://deepeval.com/docs/getting-started
Ollama: https://github.com/ollama/ollama
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import List, Optional

import requests

try:
	from deepeval.test_case import LLMTestCase
	from deepeval.metrics import BaseMetric
except ImportError as e:  # graceful message if not installed
	print("[ERROR] deepeval not installed. Please install requirements first.")
	print("Run: pip install deepeval requests")
	raise


# ----------------------------- Ollama Wrapper ---------------------------------
@dataclass
class OllamaResponse:
	model: str
	response: str
	raw: dict


class OllamaLLM:
	"""Minimal Ollama client.

	Exposes a .generate(prompt) method returning plain text. Assumes Ollama
	server running locally (default: http://localhost:11434).
	"""

	def __init__(self, model: str, host: str = "http://localhost:11434"):
		self.model = model
		self.host = host.rstrip("/")
		self._generate_url = f"{self.host}/api/generate"

	def _strip_think(self, text: str) -> str:
		"""Remove <think>...</think> blocks or standalone <think> tokens.

		Some reasoning models emit internal chain-of-thought style tags. We strip
		them to avoid leaking them into evaluation. Implementation keeps it
		simple (non-greedy). Multiple blocks supported.
		"""
		# Remove paired blocks
		cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
		# Remove stray opening/closing tokens if any remain
		cleaned = re.sub(r"</?think>", "", cleaned, flags=re.IGNORECASE)
		return cleaned.strip()

	def generate(self, prompt: str, stream: bool = False, **kwargs) -> OllamaResponse:
		payload = {"model": self.model, "prompt": prompt, "stream": stream}
		payload.update(kwargs)
		try:
			r = requests.post(self._generate_url, json=payload, timeout=120)
			r.raise_for_status()
		except requests.RequestException as exc:
			raise RuntimeError(f"Ollama request failed: {exc}") from exc

		# Non-streaming returns a JSON lines sequence; unify output
		text_parts: List[str] = []
		raw_chunks: List[dict] = []
		for line in r.text.splitlines():
			if not line.strip():
				continue
			try:
				obj = json.loads(line)
			except json.JSONDecodeError:
				continue
			raw_chunks.append(obj)
			content = obj.get("response") or obj.get("content") or ""
			text_parts.append(content)

		full_text = "".join(text_parts).strip()
		stripped = self._strip_think(full_text)
		return OllamaResponse(model=self.model, response=stripped, raw={"chunks": raw_chunks, "raw_text": full_text})


# ---------------------------- Custom Metric -----------------------------------
class KeywordMatchMetric(BaseMetric):
	"""Very simple metric: checks if all required keywords appear in output.

	Produces a score in [0,1]. 1 if every keyword (case-insensitive) is present,
	else 0. Adds some interpretability in the `reason` field.
	"""

	NAME = "keyword_match"

	def __init__(self, keywords: List[str]):
		self.keywords = [k.strip() for k in keywords if k.strip()]
		self.score: float = 0.0
		self.reason: str = ""

	def measure(self, test_case: LLMTestCase):  # type: ignore[override]
		answer = (test_case.actual_output or "").lower()
		missing = [k for k in self.keywords if k.lower() not in answer]
		if missing:
			self.score = 0.0
			self.reason = f"Missing keywords: {', '.join(missing)}"
		else:
			self.score = 1.0
			self.reason = "All keywords present"

	def is_successful(self) -> bool:  # type: ignore[override]
		return self.score == 1.0

	def to_dict(self):  # Optional pretty serialization
		return {
			"name": self.NAME,
			"score": self.score,
			"reason": self.reason,
			"keywords": self.keywords,
		}


# --------------------------- Evaluation Logic ---------------------------------
def evaluate_case(model_name: str, prompt: str, expected: Optional[str], keywords: List[str]):
	ollama = OllamaLLM(model_name)
	print(f"[INFO] Generating with Ollama model '{model_name}' ...")
	resp = ollama.generate(prompt)
	print("[MODEL OUTPUT]\n" + resp.response + "\n---")

	test_case = LLMTestCase(
		input=prompt,
		actual_output=resp.response,
		expected_output=expected,
	)

	metrics: List[BaseMetric] = []
	if keywords:
		metrics.append(KeywordMatchMetric(keywords))

	# Manually run each metric (simpler than higher-level runner for one case)
	for metric in metrics:
		metric.measure(test_case)

	print("[RESULTS]")
	for metric in metrics:
		print(f"Metric: {metric.NAME} | score={metric.score:.2f} | success={metric.is_successful()} | reason={getattr(metric, 'reason', '')}")

	return {
		"prompt": prompt,
		"model": model_name,
		"output": resp.response,
		"metrics": [getattr(metric, "to_dict", lambda: {})() for metric in metrics],
	}


# --------------------------- CLI / Main ---------------------------------------
def parse_args(argv: List[str]):
	parser = argparse.ArgumentParser(description="Evaluate a local Ollama model with Deepeval + custom metric.")
	parser.add_argument("--model", default="qwen3:4b", help="Ollama model name/tag (must be pulled already)")
	parser.add_argument("--prompt", default="What is the capital of Turkey?", help="Prompt to send to model")
	parser.add_argument("--expected", default="Ankara", help="Expected answer (optional)")
	parser.add_argument("--keywords", default="Ankara", help="Comma-separated keywords to check in output")
	parser.add_argument("--json", action="store_true", help="Print machine-readable JSON result at end")
	return parser.parse_args(argv)


def main(argv: List[str]):
	args = parse_args(argv)
	keywords = [k.strip() for k in args.keywords.split(',') if k.strip()]
	result = evaluate_case(args.model, args.prompt, args.expected, keywords)
	if args.json:
		print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
	main(sys.argv[1:])

