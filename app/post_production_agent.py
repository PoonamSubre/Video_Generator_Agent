import logging
import os
import subprocess

from google.adk.agents import Agent
from google.adk.tools import ToolContext

from .utils.utils import load_prompt_from_file
from . import tingting_brand as brand

# Set logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

DESCRIPTION = (
    "Agent responsible for assembling the final Ting Ting episode locally: "
    "gentle crossfades, on-screen number overlays, brand intro/outro and "
    "loudness-normalized mastering."
)

_FADE = 0.5  # gentle crossfade for preschool pacing
SCENE_SKIPPED = "SCENE_SKIPPED"


def _probe_duration(path: str) -> float:
    """Return media duration in seconds using ffprobe."""
    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", path,
        ],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip())


def _escape_drawtext(text: str) -> str:
    return text.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")


def stitch_movie(
    video_paths: list[str],
    number_overlays: list[str],
    tool_context: ToolContext,
) -> str:
    """
    Stitch local scene clips into the final Ting Ting episode.

    Applies gentle crossfades, optional big friendly number overlays per
    scene, brand intro/outro (if assets/brand/intro.mp4 / outro.mp4 exist),
    and child-safe loudness normalization. Saves the master locally.

    Args:
        video_paths (list[str]): Ordered local paths of the scene clips.
        number_overlays (list[str]): One entry per clip: the number/text to
            display large on screen during that scene (e.g. "3"), or an
            empty string / NONE for no overlay.
        tool_context (): ToolContext needed by the tool.

    Returns:
        str: Local path of the final episode MP4, or empty string on failure.
    """
    session_id = tool_context._invocation_context.session.id
    state = tool_context._invocation_context.session.state
    active_ratio = state.get("aspect_ratio", "16:9")
    width, height = (1920, 1080) if active_ratio == "16:9" else (1080, 1920)

    try:
        # 1. Collect valid clips (+ overlays aligned) and brand bookends.
        clips: list[str] = []
        overlays: list[str] = []
        for i, path in enumerate(video_paths):
            if path and path != SCENE_SKIPPED and os.path.exists(path):
                clips.append(path)
                text = (
                    number_overlays[i]
                    if i < len(number_overlays)
                    else ""
                )
                overlays.append(
                    "" if not text or text.strip().upper() == "NONE" else text.strip()
                )
            else:
                logger.warning(f"Skipping missing clip at position {i}: {path}")

        if not clips:
            logger.warning("No valid video clips to stitch.")
            return ""

        if brand.BRAND_INTRO_PATH.exists():
            clips.insert(0, str(brand.BRAND_INTRO_PATH))
            overlays.insert(0, "")
            logger.info("Including brand intro asset.")
        if brand.BRAND_OUTRO_PATH.exists():
            clips.append(str(brand.BRAND_OUTRO_PATH))
            overlays.append("")
            logger.info("Including brand outro asset.")

        durations = [_probe_duration(c) for c in clips]
        # Windows drive-letter colon must be escaped inside filtergraphs.
        font = brand.OVERLAY_FONT.replace("\\", "/").replace(":", "\\:")
        font_ok = os.path.exists(brand.OVERLAY_FONT)
        if not font_ok:
            logger.warning(
                f"Overlay font not found at {brand.OVERLAY_FONT}; skipping overlays."
            )

        # 2. Normalize every input to a common format so xfade works.
        filter_parts = []
        for i in range(len(clips)):
            chain = (
                f"[{i}:v]scale={width}:{height}:force_original_aspect_ratio=decrease,"
                f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=white,"
                f"setsar=1,fps=24,format=yuv420p"
            )
            if font_ok and overlays[i]:
                text = _escape_drawtext(overlays[i])
                chain += (
                    f",drawtext=fontfile='{font}':text='{text}'"
                    f":fontsize={int(height * 0.18)}:fontcolor=white"
                    f":borderw={int(height * 0.012)}:bordercolor=0x5B3FA8"
                    f":x=w-tw-{int(width * 0.05)}:y={int(height * 0.06)}"
                )
            filter_parts.append(chain + f"[nv{i}]")
            filter_parts.append(
                f"[{i}:a]aresample=48000,aformat=channel_layouts=stereo[na{i}]"
            )

        # 3. Crossfade chains with offsets from the REAL clip durations.
        n = len(clips)
        if n > 1:
            offset = 0.0
            for i in range(n - 1):
                offset += durations[i] - _FADE
                in_v = f"xv{i}" if i > 0 else "nv0"
                filter_parts.append(
                    f"[{in_v}][nv{i + 1}]xfade=transition=fade:"
                    f"duration={_FADE}:offset={offset:.3f}[xv{i + 1}]"
                )
                in_a = f"xa{i}" if i > 0 else "na0"
                filter_parts.append(
                    f"[{in_a}][na{i + 1}]acrossfade=d={_FADE}:c1=tri:c2=tri[xa{i + 1}]"
                )
            last_v, last_a = f"xv{n - 1}", f"xa{n - 1}"
        else:
            last_v, last_a = "nv0", "na0"

        # 4. Gentle polish: slight warmth, child-safe loudness target.
        filter_parts.append(f"[{last_v}]eq=saturation=1.05[v_master]")
        filter_parts.append(
            f"[{last_a}]loudnorm=I=-16:TP=-1.5:LRA=11[a_master]"
        )
        filter_complex = ";".join(filter_parts)

        # 5. Master export (local only).
        output = str(
            brand.session_dir(session_id) / "tingting_episode_master.mp4"
        )
        cmd = ["ffmpeg", "-y"]
        for c in clips:
            cmd += ["-i", c]
        cmd += [
            "-filter_complex", filter_complex,
            "-map", "[v_master]", "-map", "[a_master]",
            "-c:v", "libx264", "-preset", "slow", "-crf", "18",
            "-pix_fmt", "yuv420p", "-profile:v", "high", "-level", "4.2",
            "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart",
            output,
        ]

        logger.info(f"Mastering final episode ({n} clips) to {output}...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"ffmpeg master export failed: {result.stderr[-2000:]}")
            return clips[0]

        return output

    except Exception as e:
        logger.error(f"Post-Production master export failed: {e}", exc_info=True)
        valid = [p for p in video_paths if p and p != SCENE_SKIPPED]
        return valid[0] if valid else ""


# --- Post-Production Agent ---
post_production_agent = None
try:
    post_production_agent = Agent(
        model=brand.LLM_MODEL,
        name="post_production_agent",
        description=DESCRIPTION,
        instruction=load_prompt_from_file("post_production_agent.txt"),
        output_key="final_episode",
        tools=[stitch_movie],
    )
    logger.info(f"✅ Agent '{post_production_agent.name}' created.")
except Exception as e:
    logger.error(f"❌ Could not create PostProduction agent: {e}")
