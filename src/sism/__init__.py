"""SISM-Eval: Self-Image Sycophancy under persistent Memory.

A harness for measuring how much an assistant's endorsement of a user's
*self-image* is driven by what persistent memory says about that user,
rather than by the evidence present in the conversation.
"""

__version__ = "0.1.0"

CONDITIONS = ("none", "neutral", "flattering", "critical")
DOMAINS = ("ability", "moral", "decision")
