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
from .utils.audio_utils import generate_speech
from . import tingting_brand as brand

# Set logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

DESCRIPTION = (
    "Agent responsible for creating gentle preschool video scenes with "
    "native music/voices and optional narrator lines, saved locally."
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


def video_generate(
    prompt: str,
    scene_number: int,
    image_path: str,
    narration_text: str,
    duration_seconds: int,
    tool_context: ToolContext,
) -> str:
    """
    Generate one gentle preschool video scene and save it locally.

    Args:
        prompt (str): Visual + audio description of the scene, including any
            character dialogue or sung lyric lines Veo should voice.
        scene_number (int): Scene number.
        image_path (str): Local path of the storyboard keyframe to use as the
            starting frame (pass empty string or SCENE_SKIPPED if none).
        narration_text (str): Optional narrator voiceover line to lay over the
            scene, or NONE if the scene has no narrator.
        duration_seconds (int): Clip length, one of 4, 6 or 8.
        tool_context (): ToolContext needed by the tool.

    Returns:
        str: Local path of the finished scene clip, or SCENE_SKIPPED.
    """
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

        # 1. Optional narrator voiceover (local WAV).
        narration_path = None
        if narration_text and narration_text.strip().upper() not in ("", "NONE"):
            narration_path = generate_speech(
                narration_text.strip(), session_id, scene_number
            )

        # 2. Video generation - Ting Ting style enforced in code.
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

        # 3. Mix narrator over Veo's native music/voices (gentle ducking).
        if narration_path and os.path.exists(narration_path):
            cmd = [
                "ffmpeg", "-y", "-i", raw_path, "-i", narration_path,
                "-filter_complex",
                "[0:a]volume=0.55[bg];"
                "[1:a]aresample=48000,volume=1.1[vo];"
                "[bg][vo]amix=inputs=2:duration=first:dropout_transition=2[a]",
                "-map", "0:v", "-map", "[a]",
                "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                final_path,
            ]
            subprocess.run(cmd, capture_output=True, check=True)
        else:
            shutil.copyfile(raw_path, final_path)

        logger.info(f"Scene {scene_number} finished: {final_path}")
        return final_path

    except Exception as e:
        logger.error(
            f"Video generation failed for scene {scene_number}: {e}",
            exc_info=True,
        )
        return SCENE_SKIPPED


def video_bulk_generate(
    prompts: list[str],
    scene_numbers: list[int],
    image_paths: list[str],
    narration_texts: list[str],
    durations_seconds: list[int],
    tool_context: ToolContext,
) -> list[str]:
    """
    Generate multiple scene clips in parallel.

    Args:
        prompts (list[str]): One visual/audio prompt per scene.
        scene_numbers (list[int]): Scene numbers.
        image_paths (list[str]): Local storyboard keyframe path per scene.
        narration_texts (list[str]): Narrator line per scene (NONE if none).
        durations_seconds (list[int]): Clip length per scene (4, 6 or 8).
        tool_context (): ToolContext needed by the tool.

    Returns:
        list[str]: Local clip paths per scene (SCENE_SKIPPED on failure).
    """
    logger.info(f"🚀 Batch generating {len(prompts)} video scenes...")
    results: list[str] = [SCENE_SKIPPED] * len(prompts)
    with ThreadPoolExecutor(max_workers=min(len(prompts), 3)) as executor:
        future_to_idx = {
            executor.submit(
                video_generate,
                prompts[i],
                scene_numbers[i],
                image_paths[i] if i < len(image_paths) else "",
                narration_texts[i] if i < len(narration_texts) else "NONE",
                durations_seconds[i] if i < len(durations_seconds) else 6,
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


# --- Video Agent ---
video_agent = None
try:
    video_agent = Agent(
        model=brand.LLM_MODEL,
        name="video_agent",
        description=DESCRIPTION,
        instruction=load_prompt_from_file("video_agent.txt"),
        output_key="video",
        tools=[video_generate, video_bulk_generate],
    )
    logger.info(f"✅ Agent '{video_agent.name}' created.")
except Exception as e:
    logger.error(f"❌ Could not create Video agent: {e}")
