# Changelog

## v0.4.0 - Ting Ting preschool episode pipeline (local storage)

- Rebranded the entire pipeline from TV-commercial production to **Ting Ting**
  preschool learning episodes (ages 1.5-7): all agent prompts rewritten around
  the Ting Ting episode formula (tiny story → song → repetition → interaction
  → happy resolution), gentle pacing rules and one-learning-objective policy.
- New `app/tingting_brand.py`: central config for models, output paths, and
  the master visual style + character sheet (Ting Ting, Bobo, Mimi) that is
  appended to every Imagen/Veo prompt in code.
- **Removed all GCS storage** — storyboards, narration audio, scene clips and
  the final episode are saved locally under `output/<session_id>/`. Deleted
  `utils/gcs.py` and `utils/tracing.py`; `server.py` no longer creates buckets.
- LLM agents now use `TINGTING_LLM_MODEL` (default `gemini-3-flash-preview`).
- Video agent fixes: Veo long-running operation is now polled correctly;
  Veo native audio (music + character voices) is preserved and narration is
  mixed on top with ducking; per-scene narration lines and per-scene durations
  (4/6/8s); storyboard keyframe used as image-to-video seed; 1080p output.
- Narration switched from Cloud TTS (+GCS upload) to Gemini TTS, saved as
  local WAV.
- New storyboard QC tool `verify_storyboard_image`: Gemini vision check that
  object counts exactly match the number being taught, with one auto-retry.
- Post-production rewrite: crossfade offsets computed from real clip durations
  (ffprobe), inputs normalized to a common format, on-screen number overlays
  via drawtext, optional fixed brand intro/outro from `assets/brand/`,
  child-safe loudness normalization, local master output.

## v0.3.1 - Refactor deployment and model updates

- Upgraded image generation model to `imagen-4.0-ultra-generate-001`.
- Set the location for video generation model to `us-central1`.
- Added `GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_LOCATION` to the backend deployment in `Makefile`.
- Removed Terraform-based deployment, updated `README.md` and `Makefile` to reflect this change.

## v0.3.0 - Code updates based on agent-starter-pack 0.15.4

- Updated the codebase to align with the changes in `agent-starter-pack` version 0.15.4.

## v0.2.0 - Moved to a director workflow architecture

- Refactored the agent workflow from a sequential process to a director-based architecture for improved orchestration and flexibility.

## v0.1.0 - Initial version

- Initial release of the short movie generation agents.
- Implemented a sequential workflow for story generation, screenplay creation, storyboarding, and video production.
