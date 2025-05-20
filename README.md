# Music Video (or any) to Animation

This script provides an automated pipeline for transforming videos into animation. It processes videos by extracting keyframes, using gpt's new image gen model to stylaize the key frames, and then Kling via Replicate to bridge the gaps. Then everything is stitched together as a final video. 

## Prerequisites

1.  **Python**: Version 3.7 or later.
2.  **FFmpeg**: Required by `moviepy` for video 
3.  **API Credentials**:
    *   **OpenAI API Key**: Must be available as the environment variable `OPENAI_API_KEY`.
    *   **Replicate API Token**: Must be available as the environment variable `REPLICATE_API_TOKEN`.
4.  **Python Dependencies**: Install using the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

## `requirements.txt`

The `requirements.txt` file contains:
```
opencv-python
replicate
openai
requests
tqdm
moviepy==1.0.3
```

## Configuration

Script behavior is configured via global variables within `distr_attempt.py`:

*   `OPENAI_PROMPT`: Defines the prompt for OpenAI image stylization.
*   `REPLICATE_KLING_MODEL`: Specifies the Replicate Kling model version.
*   `KLING_PROMPT`: Sets the prompt for Kling animation generation.
*   Additional Kling parameters (`KLING_CFG_SCALE`, `KLING_ASPECT_RATIO`, `KLING_NEGATIVE_PROMPT`) and directory paths are also configurable.

## Usage

Execute the script from the command line:
```bash
python distr_process.py --video_path [PATH_TO_VIDEO] --keyframe_fps [FPS_RATE]
```

**Command-Line Arguments:**

*   `--video_path`: Path to the input video file (MP4 format). (Default: `"10_second.mp4"`)
*   `--keyframe_fps`: Rate for keyframe extraction (frames per second). E.g., `0.5` for one keyframe every two seconds. (Default: `1`)

## Operational Workflow

1.  **Keyframe Extraction**: Keyframes are extracted from the input video and stored in `keyframes_data/<video_name>/`.
2.  **OpenAI Image Styling**: Raw keyframes are styled using the OpenAI API. Styled images are saved to `openai_processed_keyframes/<video_name>/`. The pipeline halts if this stage does not complete for all keyframes.
3.  **Kling Animation Generation**: Styled keyframes are used to generate animated segments via the Replicate Kling API. Segments are saved to `kling_segments/<video_name>/`. The pipeline halts if this stage does not complete for all segments.
4.  **Video Finalization**: Generated segments are speed-adjusted to match the `target_keyframe_fps`, concatenated, and the original video's audio is applied. The final output is saved to `final_videos/`.
