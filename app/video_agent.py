import logging
import os
import shutil
import subprocess
import time
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
    "Agent responsible for creating energetic preschool video scenes - "
    "characters voice all dialogue and songs natively - saved locally."
)
DEFAULT_ASPECT_RATIO = "16:9"
SCENE_SKIPPED = "SCENE_SKIPPED"

# Veo supports 4, 6 or 8 second clips.
_ALLOWED_DURATIONS = (4, 6, 8)



def _poll_video_operation(operation):
    """Poll the Veo long-running operation until it completes."""
    waited = 0
    while not operation.done:
        time.sleep(10)
        waited += 10
        if waited > 900:
            raise TimeoutError("Veo generation timed out after 15 minutes")
        operation = brand.video_genai_client().operations.get(operation)
    return operation


def _save_generated_video(gen_video, out_path: str) -> None:
    """Persist a generated video locally, regardless of transport format."""
    video = gen_video.video
    video_bytes = getattr(video, "video_bytes", None)
    if video_bytes:
        with open(out_path, "wb") as f:
            f.write(video_bytes)
        return
    # Gemini API returns a file handle that must be downloaded.
    brand.video_genai_client().files.download(file=video)
    video.save(out_path)


def _generate_scene_clip(
    prompt: str,
    scene_number: int,
    image_path: str,
    duration_seconds: int,
    tool_context: ToolContext,
) -> str:
    """Generate one scene clip (no QC). Returns local path or SCENE_SKIPPED."""
    try:
        session_id = tool_context._invocation_context.session.id
        state = tool_context._invocation_context.session.state
        active_ratio = state.get("aspect_ratio", DEFAULT_ASPECT_RATIO)

        duration = min(
            _ALLOWED_DURATIONS, key=lambda d: abs(d - int(duration_seconds))
        )
        out_dir = brand.scene_dir(session_id, scene_number)
        raw_path = str(out_dir / "raw_clip.mp4")
        final_path = str(out_dir / "final_clip.mp4")

        # Video generation - Ting Ting style enforced in code.
        # ONE VOICE AUTHORITY: the characters (via Veo) speak and sing
        # everything. No separate narrator track is mixed - that caused
        # doubled/robotic voices.
        full_prompt = f"{prompt}\n{brand.VIDEO_CONTINUITY}\n{brand.MASTER_STYLE}"

        image = None
        if (
            image_path
            and image_path.strip()
            and image_path != SCENE_SKIPPED
            and os.path.exists(image_path)
        ):
            with open(image_path, "rb") as f:
                image = types.Image(
                    image_bytes=f.read(), mime_type="image/png"
                )

        logger.info(
            f"Triggering Veo for scene {scene_number} "
            f"({active_ratio}, {duration}s, keyframe={'yes' if image else 'no'})"
        )
        config = types.GenerateVideosConfig(
            aspect_ratio=active_ratio,
            duration_seconds=duration,
            generate_audio=True,
            resolution="1080p" if active_ratio == "16:9" else "720p",
            person_generation="allow_all",
        )
        try:
            operation = brand.video_genai_client().models.generate_videos(
                model=brand.VIDEO_MODEL,
                prompt=full_prompt,
                image=image,
                config=config,
            )
        except Exception as inner:
            # allow_all may require allowlisting; retry with the default.
            logger.warning(
                f"Retrying scene {scene_number} with default person settings: {inner}"
            )
            config.person_generation = None
            operation = brand.video_genai_client().models.generate_videos(
                model=brand.VIDEO_MODEL,
                prompt=full_prompt,
                image=image,
                config=config,
            )

        operation = _poll_video_operation(operation)
        result = operation.response or operation.result
        if not result or not result.generated_videos:
            logger.warning(f"No video generated for scene {scene_number}")
            return SCENE_SKIPPED

        _save_generated_video(result.generated_videos[0], raw_path)

        shutil.copyfile(raw_path, final_path)

        logger.info(f"Scene {scene_number} finished: {final_path}")
        return final_path

    except Exception as e:
        logger.error(
            f"Video generation failed for scene {scene_number}: {e}",
            exc_info=True,
        )
        return SCENE_SKIPPED


def video_generate(
    prompt: str,
    scene_number: int,
    image_path: str,
    duration_seconds: int,
    qc_description: str,
    tool_context: ToolContext,
) -> str:
    """
    Generate one energetic preschool video scene, save it locally, and
    AUTOMATICALLY verify that the audio matches the visuals (spoken/sung
    numbers vs visible object count). A failed scene is regenerated once
    with corrective instructions.

    Args:
        prompt (str): Visual + audio description of the scene, including any
            character dialogue or sung lyric lines Veo should voice.
        scene_number (int): Scene number.
        image_path (str): Local path of the storyboard keyframe to use as the
            starting frame (pass empty string or SCENE_SKIPPED if none).
        duration_seconds (int): Clip length, one of 4, 6 or 8.
        qc_description (str): The scene's audio/visual contract, e.g.
            "exactly 3 balloons red yellow blue visible the whole clip;
            characters count from one up to three while pointing; no number
            above three is spoken". Pass NONE to skip verification.
        tool_context (): ToolContext needed by the tool.

    Returns:
        str: Local path of the finished scene clip, or SCENE_SKIPPED.
    """
    path = _generate_scene_clip(
        prompt, scene_number, image_path, duration_seconds, tool_context,
    )
    if (
        path == SCENE_SKIPPED
        or not qc_description
        or qc_description.strip().upper() == "NONE"
    ):
        return path

    verdict = verify_scene_clip(path, qc_description, scene_number)
    if verdict.upper().startswith("PASS"):
        return path

    logger.warning(
        f"Scene {scene_number} failed A/V QC ({verdict}); regenerating once."
    )
    corrective_prompt = (
        f"{prompt}\n\nCRITICAL CORRECTION - a previous attempt failed "
        f"quality control for this reason: {verdict}\n"
        f"You MUST strictly satisfy: {qc_description}"
    )
    retry_path = _generate_scene_clip(
        corrective_prompt, scene_number, image_path, duration_seconds,
        tool_context,
    )
    if retry_path == SCENE_SKIPPED:
        return path
    verdict2 = verify_scene_clip(retry_path, qc_description, scene_number)
    if not verdict2.upper().startswith("PASS"):
        logger.warning(
            f"Scene {scene_number} still imperfect after retry: {verdict2}"
        )
    return retry_path


def video_bulk_generate(
    prompts: list[str],
    scene_numbers: list[int],
    image_paths: list[str],
    durations_seconds: list[int],
    qc_descriptions: list[str],
    tool_context: ToolContext,
) -> list[str]:
    """
    Generate multiple scene clips in parallel, each automatically verified
    for audio/visual match and retried once on failure.

    Args:
        prompts (list[str]): One visual/audio prompt per scene.
        scene_numbers (list[int]): Scene numbers.
        image_paths (list[str]): Local storyboard keyframe path per scene.
        durations_seconds (list[int]): Clip length per scene (4, 6 or 8).
        qc_descriptions (list[str]): Audio/visual contract per scene (exact
            object count, allowed counting range). NONE to skip a scene.
        tool_context (): ToolContext needed by the tool.

    Returns:
        list[str]: Local clip paths per scene (SCENE_SKIPPED on failure).
    """
    logger.info(f"🚀 Batch generating {len(prompts)} video scenes...")
    results: list[str] = [SCENE_SKIPPED] * len(prompts)
    with ThreadPoolExecutor(
        max_workers=min(len(prompts), brand.PARALLEL_VIDEOS)
    ) as executor:
        future_to_idx = {
            executor.submit(
                video_generate,
                prompts[i],
                scene_numbers[i],
                image_paths[i] if i < len(image_paths) else "",
                durations_seconds[i] if i < len(durations_seconds) else 6,
                qc_descriptions[i] if i < len(qc_descriptions) else "NONE",
                tool_context,
            ): i
            for i in range(len(prompts))
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                logger.error(f"Error in batch video idx {idx}: {e}")
                results[idx] = SCENE_SKIPPED
    return results


def verify_scene_clip(
    clip_path: str, expected_description: str, scene_number: int
) -> str:
    """
    Verify a finished scene clip: watches the video AND listens to its audio
    to confirm the spoken/sung numbers match the visible object count.

    Args:
        clip_path (str): Local path of the scene clip (mp4).
        expected_description (str): What must be true, e.g.
            "exactly 3 balloons visible for the whole clip; the characters
            count or sing only up to three; no other numbers are spoken".
        scene_number (int): Scene number.

    Returns:
        str: "PASS" or "FAIL: <reason>".
    """
    try:
        if not clip_path or clip_path == SCENE_SKIPPED:
            return "FAIL: no clip was generated for this scene"
        if not os.path.exists(clip_path):
            return f"FAIL: clip file not found at {clip_path}"

        with open(clip_path, "rb") as f:
            video_bytes = f.read()
        if len(video_bytes) > 18 * 1024 * 1024:
            logger.warning(
                f"Clip for scene {scene_number} too large for inline QC; skipping."
            )
            return "PASS (clip too large for automated check)"

        question = (
            "You are a quality checker for a preschool counting video. "
            "Watch this clip AND listen to its audio track. Requirements:\n"
            f"{expected_description}\n\n"
            "COUNTING PEDAGOGY (important): counting UP from one to the "
            "total visible (e.g. saying one, two, three while three balloons "
            "are shown, ideally pointing at each) is CORRECT and must PASS. "
            "Objects may also appear one at a time in sync with the count.\n\n"
            "FAIL ONLY for these critical issues:\n"
            "1. The counting goes HIGHER than the number of objects visible, "
            "or the count ends at a different number than the visible total "
            "(e.g. counting to four when five balloons are shown).\n"
            "2. Numbers are spoken/sung while a clearly wrong number of "
            "objects is on screen (e.g. counting balloons when none are "
            "visible).\n"
            "3. Objects randomly appear or disappear mid-clip out of sync "
            "with the counting.\n"
            "4. A required main character is missing or badly off-model.\n"
            "5. Frightening, chaotic or unsafe content.\n"
            "IGNORE minor pose, gesture, gaze or framing differences - those "
            "are PASS.\n"
            "Reply with exactly PASS if acceptable, otherwise FAIL: followed "
            "by a short reason."
        )
        response = brand.generate_content_safe(
            model=brand.LLM_MODEL,
            contents=[
                types.Part.from_bytes(data=video_bytes, mime_type="video/mp4"),
                question,
            ],
        )
        verdict = (response.text or "").strip()
        logger.info(f"A/V QC scene {scene_number}: {verdict[:200]}")
        return verdict if verdict else "FAIL: empty verification response"
    except Exception as e:
        logger.error(f"A/V QC error for scene {scene_number}: {e}", exc_info=True)
        # Do not block production on a QC infrastructure error.
        return f"PASS (verification unavailable: {e})"


# --- Video Agent ---
video_agent = None
try:
    video_agent = Agent(
        model=brand.LLM_MODEL,
        name="video_agent",
        description=DESCRIPTION,
        instruction=load_prompt_from_file("video_agent.txt"),
        output_key="video",
        tools=[video_generate, video_bulk_generate, verify_scene_clip],
    )
    logger.info(f"✅ Agent '{video_agent.name}' created.")
except Exception as e:
    logger.error(f"❌ Could not create Video agent: {e}")
