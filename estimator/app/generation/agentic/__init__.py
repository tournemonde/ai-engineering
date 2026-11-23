"""Agentic generation — Actor-Critic-Boss (S5) + hand-written agent (S12).

``boss.py`` / ``critic.py``: conversational ACB loop (may import ``conversation``).
``agent_loop.py`` / ``agent_tools.py`` / ``agent_schemas.py``: Session 12 manual
Responses API agent (may NOT import other ``generation`` siblings except via
the injectable retrieval backend that wraps ``retrieve()``).
"""
