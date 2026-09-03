import logging
from google.adk.agents import Agent
from .screenplay_agent import screenplay_agent
from .story_agent import story_agent
from .storyboard_agent import storyboard_agent
from .utils.utils import load_prompt_from_file
from .video_agent import video_agent
from .post_production_agent import post_production_agent
from . import tingting_brand as brand

# Set logging
logger = logging.getLogger(__name__)

DESCRIPTION = (
    "Orchestrates the production of a short Ting Ting preschool learning "
    "episode (ages 1.5-7) based on the requested topic, utilizing "
    "specialized sub-agents for episode concept, screenplay, storyboards, "
    "video scenes and final mastering."
)

# --- Director Agent (root agent) ---

if story_agent and screenplay_agent and storyboard_agent and video_agent and post_production_agent:
    root_agent = Agent(
        name="director_agent",
        model=brand.LLM_MODEL,
        description=(DESCRIPTION),
        instruction=load_prompt_from_file("director_agent.txt"),
        sub_agents=[
            story_agent,
            screenplay_agent,
            storyboard_agent,
            video_agent,
            post_production_agent,
        ],
    )
    logger.info(f"✅ Agent '{root_agent.name}' created using model '{brand.LLM_MODEL}'.")
else:
    logger.error(
        "❌ Cannot create root agent because one or more sub-agents failed to initialize or a tool is missing."
    )
    if not story_agent:
        logger.error(" - Story Agent is missing.")
    if not screenplay_agent:
        logger.error(" - Screenplay Agent is missing.")
    if not storyboard_agent:
        logger.error(" - Storyboard Agent is missing.")
    if not video_agent:
        logger.error(" - Video Agent is missing.")
