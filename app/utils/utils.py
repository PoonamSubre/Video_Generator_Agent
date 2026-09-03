# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os

# Set logging
logger = logging.getLogger(__name__)
PROMPTS_PATH = "../prompts/"


def load_prompt_from_file(
    filename: str, default_instruction: str = "Default instruction."
) -> str:
    """Reads instruction text from a file relative to this script."""
    instruction = default_instruction
    try:
        # Construct path relative to the current script file (__file__)
        filepath = os.path.join(
            os.path.dirname(__file__), PROMPTS_PATH, filename
        )
        with open(filepath, encoding="utf-8") as f:
            instruction = f.read()
        logger.info(f"Successfully loaded instruction from {filename}")
    except FileNotFoundError:
        logger.warning(
            f"WARNING: Instruction file not found: {filepath}. Using default."
        )
    except Exception as e:
        logger.error(
            f"ERROR loading instruction file {filepath}: {e}. Using default."
        )
    return instruction
def create_agent_with_fallbacks(
    name: str,
    instruction: str,
    description: str,
    primary_model: str = "gemini-2.0-flash",
    output_key: str = None,
    tools: list = None,
    sub_agents: list = None,
) -> "Agent":
    """
    Creates an Agent with a primary model. 
    Note: ADK Agent handles model interaction, but this helper provides a consistent 
    interface for re-initializing with fallbacks if needed.
    """
    from google.adk.agents import Agent
    
    # In a more advanced implementation, we could wrap the agent's run method 
    # to catch quota errors and retry with a different model.
    # For now, we return a standard agent with the primary model.
    return Agent(
        name=name,
        model=primary_model,
        instruction=instruction,
        description=description,
        output_key=output_key,
        tools=tools,
        sub_agents=sub_agents
    )
