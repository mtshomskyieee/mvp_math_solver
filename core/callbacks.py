# math_solver/core/callbacks.py
from langchain.callbacks.base import BaseCallbackHandler
from typing import Any, Dict, List, Union

class StreamlitCallbackHandler(BaseCallbackHandler):
    """Callback handler for streaming to Streamlit."""

    def __init__(self, container):
        self.container = container
        self.text = ""

    def on_llm_start(self, serialized, prompts, **kwargs):
        self.text = ""
        self.container.write("Thinking...")

    def on_llm_new_token(self, token, **kwargs):
        self.text += token
        self.container.write(self.text)

    def on_tool_start(self, serialized, input_str, **kwargs):
        self.container.write(f"Using tool: {serialized['name']} with input: {input_str}")

    def on_tool_end(self, output, **kwargs):
        self.container.write(f"Tool output: {output}")

    def on_agent_action(self, action, **kwargs):
        self.container.write(f"Agent action: {action.tool} with input: {action.tool_input}")
