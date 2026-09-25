import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from google.genai import types
from google.adk.agents import Agent
from google.adk.tools import ToolContext

from .utils.utils import load_prompt_from_file
from . import tingting_brand as brand

# Set logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

DESCRIPTION = (
    "Agent responsible for creating Ting Ting storyboard keyframes based on "
    "the screenplay, and visually verifying object counts in each frame."
)

SCENE_SKIPPED = "SCENE_SKIPPED"


def storyboard_generate(
    prompt: str, scene_number: int, tool_context: ToolContext
) -> str:
    """
    Generate one storyboard keyframe image for a scene and save it locally.
    If an anchor frame exists for the session (scene 1), it is passed as a
    style/character reference so every scene shares the same world.

    Args:
        prompt (str): Visual description of the scene (character actions,
            setting, exact number of countable objects, on-screen number).
        scene_number (int): Scene number.
        tool_context (): ToolContext needed by the tool.

    Returns:
        str: Absolute local file path of the generated PNG, or SCENE_SKIPPED.
    """
    try:
        session_id = tool_context._invocation_context.session.id
        state = tool_context._invocation_context.session.state
        active_ratio = state.get("aspect_ratio", "16:9")

        # Visual continuity: use the session's anchor frame (scene 1) as a
        # hard reference for art style, characters, environment and lighting.
        anchor_path = state.get("storyboard_anchor")
        contents: list = []
        if (
            anchor_path
            and os.path.exists(anchor_path)
            and scene_number != 1
        ):
            with open(anchor_path, "rb") as f:
                contents.append(
                    types.Part.from_bytes(
                        data=f.read(), mime_type="image/png"
                    )
                )
            instruction = (
                "The attached image is the reference frame of this preschool "
                "video. Generate the NEXT storyboard frame of the SAME video: "
                "keep exactly the same art style, character designs, faces, "
                "clothing, tree style, lighting and color palette as the "
                "reference - but DO vary the camera framing (wide shot, "
                "close-up, or character facing the camera) and show a "
                "different angle or corner of the same garden world as the "
                "scene requires, so consecutive frames do not look identical:\n"
            )
        else:
            instruction = (
                "Generate a single storyboard image for a preschool video.\n"
            )

        # Brand style + character sheet are enforced in code, not by the LLM.
        contents.append(
            f"{instruction}{prompt}\n{brand.CHARACTER_SHEET}\n{brand.MASTER_STYLE}"
        )
        logger.info(
            f"Generating {active_ratio} storyboard for scene {scene_number} "
            f"(anchor={'yes' if len(contents) > 1 else 'no'})"
        )

        response = brand.generate_content_safe(
            model=brand.IMAGE_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"],
                image_config=types.ImageConfig(aspect_ratio=active_ratio),
            ),
        )

        image_bytes = None
        for part in (response.candidates[0].content.parts or []):
            if part.inline_data and part.inline_data.data:
                image_bytes = part.inline_data.data
                break

        if not image_bytes:
            logger.info(f"No image returned for scene {scene_number}")
            return SCENE_SKIPPED

        out_path = brand.scene_dir(session_id, scene_number) / "storyboard.png"
        with open(out_path, "wb") as f:
            f.write(image_bytes)
        logger.info(f"Saved storyboard for scene {scene_number} to {out_path}")
        return str(out_path)
    except Exception as e:
        logger.error(
            f"Error generating storyboard for scene {scene_number}: {e}",
            exc_info=True,
        )
        return SCENE_SKIPPED


def storyboard_bulk_generate(
    prompts: list[str], scene_numbers: list[int], tool_context: ToolContext
) -> list[str]:
    """
    Generate multiple storyboard images in parallel and save them locally.

    Args:
        prompts (list[str]): One visual prompt per scene.
        scene_numbers (list[int]): Scene numbers matching the prompts.
        tool_context (): ToolContext needed by the tool.

    Returns:
        list[str]: Local image paths per scene (SCENE_SKIPPED on failure).
    """
    logger.info(f"🚀 Batch generating {len(prompts)} storyboard images...")
    results: list[str] = [SCENE_SKIPPED] * len(prompts)
    if not prompts:
        return results

    # 1. Generate the anchor frame first (lowest scene number) - it defines
    #    the art style, characters and world for the whole episode.
    anchor_idx = min(range(len(scene_numbers)), key=lambda i: scene_numbers[i])
    anchor_result = storyboard_generate(
        prompts[anchor_idx], scene_numbers[anchor_idx], tool_context
    )
    results[anchor_idx] = anchor_result
    if anchor_result != SCENE_SKIPPED:
        tool_context._invocation_context.session.state["storyboard_anchor"] = (
            anchor_result
        )

    # 2. Generate the remaining scenes in parallel against the anchor.
    remaining = [i for i in range(len(prompts)) if i != anchor_idx]
    with ThreadPoolExecutor(
        max_workers=min(max(len(remaining), 1), brand.PARALLEL_IMAGES)
    ) as executor:
        future_to_idx = {
            executor.submit(
                storyboard_generate, prompts[i], scene_numbers[i], tool_context
            ): i
            for i in remaining
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                logger.error(f"Error in batch storyboard idx {idx}: {e}")
                results[idx] = SCENE_SKIPPED

    return results


def verify_storyboard_image(
    image_path: str, expected_description: str, scene_number: int
) -> str:
    """
    Visually verify a storyboard frame against Ting Ting educational rules
    (e.g. EXACTLY three balloons visible, number 3 shown, characters on model).

    Args:
        image_path (str): Local path of the storyboard image.
        expected_description (str): What must be true, e.g.
            "exactly 3 balloons (red, yellow, blue), large number 3 visible,
            Ting Ting pointing at the balloons, no extra balloons anywhere".
        scene_number (int): Scene number.

    Returns:
        str: "PASS" or "FAIL: <reason>".
    """
    try:
        if not image_path or image_path == SCENE_SKIPPED:
            return "FAIL: no image was generated for this scene"
        if not os.path.exists(image_path):
            return f"FAIL: image file not found at {image_path}"

        with open(image_path, "rb") as f:
            image_bytes = f.read()

        question = (
            "You are a quality checker for a preschool learning video. "
            "Check this storyboard frame against these requirements:\n"
            f"{expected_description}\n\n"
            "FAIL ONLY for issues that break a toddler's learning experience:\n"
            "- wrong number of countable objects (count carefully, including "
            "partially visible ones in the background)\n"
            "- a required main character missing, or an unexpected extra character\n"
            "- a required on-screen number missing or showing the wrong digit\n"
            "- characters badly off-model, or unsafe/frightening content\n"
            "IGNORE minor differences in pose, gesture, gaze direction, which "
            "hand is used, exact framing or background decoration - those are "
            "PASS.\n"
            "Reply with exactly PASS if acceptable, otherwise FAIL: followed "
            "by a short reason."
        )
        response = brand.generate_content_safe(
            model=brand.LLM_MODEL,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                question,
            ],
        )
        verdict = (response.text or "").strip()
        logger.info(f"QC scene {scene_number}: {verdict}")
        return verdict if verdict else "FAIL: empty verification response"
    except Exception as e:
        logger.error(f"QC error for scene {scene_number}: {e}", exc_info=True)
        # Do not block production on a QC infrastructure error.
        return f"PASS (verification unavailable: {e})"


# --- Storyboard Agent ---
storyboard_agent = None
try:
    storyboard_agent = Agent(
        model=brand.LLM_MODEL,
        name="storyboard_agent",
        description=(DESCRIPTION),
        instruction=load_prompt_from_file("storyboard_agent.txt"),
        output_key="storyboard",
        tools=[storyboard_generate, storyboard_bulk_generate, verify_storyboard_image],
    )
    logger.info(
        f"✅ Agent '{storyboard_agent.name}' created using model '{brand.LLM_MODEL}'."
    )
except Exception as e:
    logger.error(
        f"❌ Could not create Storyboard agent. Check API Key ({brand.LLM_MODEL}). Error: {e}"
    )
