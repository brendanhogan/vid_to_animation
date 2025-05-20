import os
import cv2
import time
import math
import base64 
import shutil
import openai 
import argparse
import requests
import replicate
from tqdm import tqdm
from pathlib import Path
import concurrent.futures 
import moviepy.video.fx.all as vfx 
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips, ImageClip


# --- Configuration ---
BASE_DIR = Path(__file__).parent
KEYFRAMES_ROOT_DIR = BASE_DIR / "keyframes_data" # Raw extracted keyframes
OPENAI_KEYFRAMES_ROOT_DIR = BASE_DIR / "openai_processed_keyframes" # Keyframes after OpenAI styling
KLING_SEGMENTS_ROOT_DIR = BASE_DIR / "kling_segments"
FINAL_VIDEOS_ROOT_DIR = BASE_DIR / "final_videos"

# OpenAI Configuration
OPENAI_API_KEY_ENV_VAR = "OPENAI_API_KEY"
OPENAI_PROMPT = "Take this image and make it Studio Ghibli style, but keep all the details otherwise the same."

# Replicate Kling Configuration
REPLICATE_API_TOKEN_ENV_VAR = "REPLICATE_API_TOKEN"
REPLICATE_KLING_MODEL = "kwaivgi/kling-v1.6-pro"
KLING_PROMPT = "studo ghibli style music video animation" 
KLING_CFG_SCALE = 0.5
KLING_ASPECT_RATIO = "16:9"
KLING_NEGATIVE_PROMPT = "low clarity, low resolution, low quality, blurriness"

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def get_video_details(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, 0, 0, 0, 0 
    native_fps = cap.get(cv2.CAP_PROP_FPS)
    total_native_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_native_frames / native_fps if native_fps > 0 else 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return native_fps, total_native_frames, duration, width, height

def extract_keyframes(video_path_str: str, target_keyframe_fps: float, video_name: str):
    print(f"\n--- Phase 1: Extracting Raw Keyframes for {video_name} at {target_keyframe_fps} FPS ---")
    video_path = Path(video_path_str)
    keyframes_dir = KEYFRAMES_ROOT_DIR / video_name
    ensure_dir(keyframes_dir)

    native_fps, total_native_frames, video_duration, _, _ = get_video_details(video_path)
    if native_fps is None: print(f"Error: Could not open video {video_path_str}"); return [], 0
    if target_keyframe_fps <= 0: print("Error: Target keyframe FPS must be positive."); return [], 0
    if native_fps == 0: print("Error: Native FPS of video is 0. Cannot process."); return [], 0

    native_frame_interval = native_fps / target_keyframe_fps
    expected_num_keyframes = 0
    if total_native_frames > 0:
        _expected_count = 0
        _temp_cursor = 0.0
        while _temp_cursor < total_native_frames:
            _expected_count += 1
            _temp_cursor += native_frame_interval
            if abs(native_frame_interval) < 1e-6 and _expected_count > 1: _expected_count -=1; break
        expected_num_keyframes = max(1, _expected_count)
    
    print(f"Native FPS: {native_fps:.2f}, Total Frames: {total_native_frames}, Duration: {video_duration:.2f}s")
    print(f"Target Keyframe FPS: {target_keyframe_fps:.2f}, Native Frame Interval: {native_frame_interval:.2f}")
    print(f"Expecting approx {expected_num_keyframes} raw keyframes.")

    cap = cv2.VideoCapture(str(video_path))
    current_keyframe_idx = 0
    native_frame_cursor = 0.0
    extracted_keyframe_paths = []
    actual_extracted_this_run = 0
    already_exist_count = 0

    # Safety break for loop, in case of extreme FPS values leading to too many frames
    MAX_KEYFRAMES_TO_EXTRACT = 10000 

    with tqdm(total=expected_num_keyframes, desc="Extracting raw keyframes", unit="keyframe") as pbar:
        processed_indices = set() # To avoid double-processing due to rounding with native_frame_cursor
        while current_keyframe_idx < MAX_KEYFRAMES_TO_EXTRACT:
            target_native_frame_num = round(native_frame_cursor)

            if target_native_frame_num in processed_indices and current_keyframe_idx > 0:
                # If this frame index was already processed due to rounding, advance cursor and continue
                native_frame_cursor += native_frame_interval
                if abs(native_frame_interval) < 1e-6 : break # Safety for effectively zero interval
                continue

            if target_native_frame_num >= total_native_frames:
                if not extracted_keyframe_paths and total_native_frames > 0 and current_keyframe_idx == 0:
                    # Edge case: Video is shorter than one frame interval. Try to get the first frame.
                    target_native_frame_num = 0
                else:
                    break # Reached or passed the end of the video
            
            if total_native_frames == 0 and current_keyframe_idx > 0: break

            keyframe_filename = f"keyframe_{current_keyframe_idx:05d}.jpg"
            keyframe_path = keyframes_dir / keyframe_filename

            if keyframe_path.exists():
                already_exist_count +=1
            else:
                if total_native_frames == 0: break
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_native_frame_num)
                ret, frame = cap.read()
                if not ret:
                    # If we expected more frames but couldn't read, stop.
                    if current_keyframe_idx < expected_num_keyframes -1: # -1 because current_keyframe_idx is 0-indexed
                         print(f"\nWarning: Could not read frame {target_native_frame_num} for keyframe_{current_keyframe_idx:05d}, but more were expected. Stopping early.")
                    break 
                cv2.imwrite(str(keyframe_path), frame)
                actual_extracted_this_run +=1            
            
            extracted_keyframe_paths.append(str(keyframe_path))
            processed_indices.add(target_native_frame_num)
            pbar.update(1)            
            current_keyframe_idx += 1
            native_frame_cursor += native_frame_interval
            if abs(native_frame_interval) < 1e-6 : break # Safety for effectively zero interval
            if current_keyframe_idx >= expected_num_keyframes: # Stop if we hit expected
                 break
        if current_keyframe_idx >= MAX_KEYFRAMES_TO_EXTRACT:
            print(f"\nWarning: Stopped keyframe extraction at {MAX_KEYFRAMES_TO_EXTRACT} frames due to limit.")

    cap.release()
    all_found_keyframes = sorted([str(p) for p in keyframes_dir.glob("keyframe_*.jpg")])
    print(f"Raw keyframe extraction summary: {actual_extracted_this_run} new, {already_exist_count} existed. Total found: {len(all_found_keyframes)}.")

    if not all_found_keyframes and total_native_frames > 0:
        print("Error: No raw keyframes were extracted, though video has content. Check FPS settings or video integrity.")
        return [], 0
    if not all_found_keyframes and total_native_frames == 0:
        print("Video appears to have no frames. No keyframes extracted.")
        return [],0
        
    final_expected_kf_count = len(all_found_keyframes)
    if expected_num_keyframes > final_expected_kf_count:
        print(f"Warning: Initially expected {expected_num_keyframes} keyframes, but only {final_expected_kf_count} were extracted/found. This might be due to video length or FPS settings.")
    elif final_expected_kf_count > expected_num_keyframes and expected_num_keyframes > 0:
        print(f"Note: Found {final_expected_kf_count} keyframes, initially expected {expected_num_keyframes}. Using actual count.")

    print(f"Proceeding with {len(all_found_keyframes)} raw keyframes for OpenAI processing.")
    return all_found_keyframes, final_expected_kf_count


def process_keyframe_with_openai(openai_client, original_keyframe_path_str: str, openai_output_dir: Path, video_name: str):
    original_keyframe_path = Path(original_keyframe_path_str)
    openai_styled_keyframe_filename = original_keyframe_path.name
    openai_styled_keyframe_path = openai_output_dir / openai_styled_keyframe_filename

    if openai_styled_keyframe_path.exists():
        return True, str(openai_styled_keyframe_path)

    try:
        with open(original_keyframe_path, "rb") as image_file:
            response = openai_client.images.edit(
                image=image_file,
                prompt=OPENAI_PROMPT,
                model="gpt-image-1",
                size="1024x1024", 
                quality="high"
            )
        image_b64 = response.data[0].b64_json
        image_bytes = base64.b64decode(image_b64)
        with open(openai_styled_keyframe_path, "wb") as f:
            f.write(image_bytes)
        return True, str(openai_styled_keyframe_path)
    except openai.APIError as e:
        print(f"OpenAI API Error processing {original_keyframe_path.name}: Code {e.code}, Type: {e.type}, Message: {e.message}")
        return False, str(e)
    except Exception as e:
        print(f"Generic error processing {original_keyframe_path.name} with OpenAI: {e}")
        return False, str(e)


def run_openai_on_keyframes(original_keyframe_paths: list, video_name: str, expected_raw_kf_count: int):
    print(f"\n--- Phase 2: Styling Keyframes with OpenAI for {video_name} ---")    
    if not original_keyframe_paths: print("No raw keyframes to process with OpenAI. Skipping."); return []
    if not os.getenv(OPENAI_API_KEY_ENV_VAR): print(f"Error: {OPENAI_API_KEY_ENV_VAR} not set. Cannot use OpenAI."); return []
    
    try: openai_client = openai.OpenAI(api_key=os.getenv(OPENAI_API_KEY_ENV_VAR))
    except Exception as e: print(f"Failed to initialize OpenAI client: {e}"); return []

    openai_output_dir = OPENAI_KEYFRAMES_ROOT_DIR / video_name
    ensure_dir(openai_output_dir)

    styled_keyframe_paths = [] 
    tasks_to_submit = []
    
    # Pre-check for existing styled frames and prepare tasks
    for original_kf_path_str in original_keyframe_paths:
        original_keyframe_path = Path(original_kf_path_str)
        openai_styled_keyframe_filename = original_keyframe_path.name
        openai_styled_keyframe_path_check = openai_output_dir / openai_styled_keyframe_filename
        if openai_styled_keyframe_path_check.exists():
            styled_keyframe_paths.append(str(openai_styled_keyframe_path_check))
        else:
            tasks_to_submit.append(original_kf_path_str)

    newly_styled_count = 0
    failed_openai_frames = 0
    already_styled_count = len(styled_keyframe_paths)

    if not tasks_to_submit:
        print(f"OpenAI keyframe styling summary: 0 new styled (all {already_styled_count} already existed), 0 failed.")
        # Sort to ensure consistent order if all existed
        styled_keyframe_paths.sort(key=lambda p: Path(p).name) 
        return styled_keyframe_paths

    # Max workers can be tuned. os.cpu_count() is a common default, but for I/O bound, can be higher.
    # Let's cap it to a reasonable number like 10 to avoid overwhelming the API or local resources.
    max_workers = min(10, os.cpu_count() + 4 if os.cpu_count() else 8) 

    print(f"Found {already_styled_count} already styled keyframes. Submitting {len(tasks_to_submit)} new keyframes for OpenAI styling using up to {max_workers} parallel workers...")

    # Store futures to retrieve results in order if needed, though order of completion may vary
    future_to_path = {}

    with tqdm(total=len(tasks_to_submit), desc="Styling keyframes (OpenAI)", unit="keyframe") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            for original_kf_path_str_to_process in tasks_to_submit:
                future = executor.submit(process_keyframe_with_openai, openai_client, original_kf_path_str_to_process, openai_output_dir, video_name)
                future_to_path[future] = original_kf_path_str_to_process

            for future in concurrent.futures.as_completed(future_to_path):
                original_path_submitted = future_to_path[future]
                try:
                    success, result_path_or_msg = future.result()
                    if success:
                        styled_keyframe_paths.append(result_path_or_msg) # result_path_or_msg is the path here
                        newly_styled_count += 1
                    else:
                        failed_openai_frames += 1
                        # Error message is already printed by process_keyframe_with_openai
                except Exception as exc:
                    failed_openai_frames += 1
                    print(f"Exception for {Path(original_path_submitted).name} during OpenAI processing: {exc}")
                pbar.update(1)
    
    print(f"OpenAI keyframe styling summary: {newly_styled_count} new styled, {already_styled_count} already existed, {failed_openai_frames} failed.")
    
    # Ensure final list is sorted by original keyframe order for consistency before returning
    # This is important as parallel execution might complete out of order
    temp_map = {Path(p).name: p for p in styled_keyframe_paths} # Map filename to full path
    sorted_styled_paths = []
    for original_kf_path_str in original_keyframe_paths: # Iterate in the original order
        filename = Path(original_kf_path_str).name
        if filename in temp_map:
            sorted_styled_paths.append(temp_map[filename])
    
    print(f"Total OpenAI-styled keyframes available: {len(sorted_styled_paths)} out of {len(original_keyframe_paths)} raw keyframes attempted.")

    # The main function will check for completeness based on expected_raw_kf_count
    if failed_openai_frames > 0:
        print(f"Warning: {failed_openai_frames} keyframes failed OpenAI processing during this run.")

    return sorted_styled_paths


def _extract_downloadable_url_from_kling_output(raw_output, segment_filename_for_log: str) -> str | None:
    """Tries to extract a usable HTTP URL from the raw output of replicate.run()."""
    print(raw_output)
    return raw_output.url

    
# Helper function for parallel Kling segment generation
def _generate_and_download_single_kling_segment(start_image_path_str: str, end_image_path_str: str, 
                                                segment_filepath: Path, kling_api_request_duration: float, 
                                                video_name: str, pbar: tqdm):
    start_image_path = Path(start_image_path_str)
    end_image_path = Path(end_image_path_str)
    segment_filename = segment_filepath.name

    try:
        with open(start_image_path, "rb") as start_img_file, open(end_image_path, "rb") as end_img_file:
            kling_input = {
                "prompt": KLING_PROMPT, "duration": kling_api_request_duration,
                "cfg_scale": KLING_CFG_SCALE, "start_image": start_img_file,
                "end_image": end_img_file, "aspect_ratio": KLING_ASPECT_RATIO,
                "negative_prompt": KLING_NEGATIVE_PROMPT
            }
            output_from_replicate = replicate.run(REPLICATE_KLING_MODEL, input=kling_input)
        
        actual_url_to_download = _extract_downloadable_url_from_kling_output(output_from_replicate, segment_filename)

        if actual_url_to_download:
            response = requests.get(actual_url_to_download, stream=True)
            response.raise_for_status()
            with open(segment_filepath, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192): f.write(chunk)
            return True, str(segment_filepath)
        else:
            print(f"Critical Error: Failed to obtain a valid downloadable URL for {segment_filename}. Check logs from _extract_downloadable_url_from_kling_output.")
            return False, f"URL extraction failed for {segment_filename}"
    except Exception as e:
        print(f"Error during Kling segment generation or download for {segment_filename}: {e}")
        if segment_filepath.exists(): 
            try: segment_filepath.unlink(); # print(f"Removed potentially corrupt file: {segment_filepath}")
            except OSError as ose: print(f"Error removing file {segment_filepath}: {ose}")
        return False, f"Exception for {segment_filename}: {str(e)}"


def generate_kling_segments(styled_keyframe_paths: list, target_keyframe_fps: float, video_name: str):
    print(f"\n--- Phase 3: Generating Video Segments with Kling for {video_name} (using OpenAI-styled keyframes) ---")
    if len(styled_keyframe_paths) < 2: print("Not enough styled keyframes (<2) for Kling. Skipping."); return []
    
    segments_dir = KLING_SEGMENTS_ROOT_DIR / video_name
    ensure_dir(segments_dir)
    ideal_segment_duration_per_keyframe_interval = 1.0 / target_keyframe_fps
    # Kling currently has a max duration of 5s for its generation model, even if you request longer.
    # So, we request 5s. The speedup logic in stitch_and_finalize will handle adjusting it to ideal_segment_duration.
    kling_api_request_duration = 5.0 # Max supported by current Kling model
    print(f"Ideal duration for each segment slot: {ideal_segment_duration_per_keyframe_interval:.2f}s. Kling API will be requested to generate segments of {kling_api_request_duration:.2f}s (will be speed-adjusted later if needed).")

    if not os.getenv(REPLICATE_API_TOKEN_ENV_VAR): print(f"Error: {REPLICATE_API_TOKEN_ENV_VAR} not set. Cannot call Kling."); return []

    tasks = [] # List of tuples: (start_image_path_str, end_image_path_str, segment_filepath)
    all_expected_segment_paths_strs = [] # For final sorting/collecting

    for i in range(len(styled_keyframe_paths) - 1):
        start_image_path = Path(styled_keyframe_paths[i])
        end_image_path = Path(styled_keyframe_paths[i+1])
        segment_filename = f"segment_styled_{start_image_path.stem}_to_{end_image_path.stem}.mp4"
        segment_filepath = segments_dir / segment_filename
        all_expected_segment_paths_strs.append(str(segment_filepath))

        if not segment_filepath.exists():
            tasks.append((str(start_image_path), str(end_image_path), segment_filepath))
    
    generated_segment_paths = [p for p in all_expected_segment_paths_strs if Path(p).exists()] # Collect already existing
    already_existed_segments = len(generated_segment_paths)
    newly_generated_segments = 0
    failed_segments = 0

    if not tasks:
        print(f"Kling segment generation summary: 0 new (all {already_existed_segments} already existed), 0 failed.")
        generated_segment_paths.sort() # Ensure order
        return generated_segment_paths

    # Max workers for Kling. This can be more aggressive if Replicate handles rate limiting well.
    # Let's keep it moderate, e.g., 5, as these are heavier tasks than OpenAI image edits.
    max_workers = min(128, os.cpu_count() + 4 if os.cpu_count() else 8)
    print(f"Found {already_existed_segments} already generated Kling segments. Submitting {len(tasks)} new segments for generation using up to {max_workers} parallel workers...")

    future_to_segment_file = {}
    with tqdm(total=len(tasks), desc="Generating Kling segments", unit="segment") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            for start_img_str, end_img_str, seg_fp in tasks:
                future = executor.submit(_generate_and_download_single_kling_segment, 
                                         start_img_str, end_img_str, seg_fp, 
                                         kling_api_request_duration, video_name, pbar)
                future_to_segment_file[future] = seg_fp

            for future in concurrent.futures.as_completed(future_to_segment_file):
                segment_file_path_for_task = future_to_segment_file[future]
                try:
                    success, result_path_or_msg = future.result()
                    if success:
                        generated_segment_paths.append(result_path_or_msg)
                        newly_generated_segments += 1
                    else:
                        failed_segments += 1
                        # Error is logged by the helper function
                except Exception as exc:
                    failed_segments += 1
                    print(f"Exception for segment {segment_file_path_for_task.name} during Kling processing: {exc}")
                    if segment_file_path_for_task.exists():
                        try: segment_file_path_for_task.unlink()
                        except OSError as ose: print(f"Error removing file {segment_file_path_for_task.name}: {ose}")
                pbar.update(1)
    
    print(f"Kling segment generation summary: {newly_generated_segments} new, {already_existed_segments} existed, {failed_segments} failed.")
    final_sorted_segment_paths = []
    temp_map = {Path(p).name: p for p in generated_segment_paths}
    for expected_path_str in all_expected_segment_paths_strs:
        filename = Path(expected_path_str).name
        if filename in temp_map:
            final_sorted_segment_paths.append(temp_map[filename])

    print(f"Total Kling segments available: {len(final_sorted_segment_paths)} out of {len(styled_keyframe_paths) -1} attempts.")

    if failed_segments > 0:
        print(f"Warning: {failed_segments} Kling segments failed generation during this run. Re-run to retry. The script will check for completeness before stitching.")
    
    return final_sorted_segment_paths


def stitch_and_finalize(segment_paths: list, original_video_path_str: str, final_video_name_base: str, target_keyframe_fps: float, compress: bool = False):
    print(f"\n--- Phase 4: Stitching Segments and Finalizing Video for {final_video_name_base} ---")
    if not segment_paths: print("No video segments to stitch. Exiting."); return

    ensure_dir(FINAL_VIDEOS_ROOT_DIR)
    final_video_path = FINAL_VIDEOS_ROOT_DIR / f"{final_video_name_base}_openai_kling_{KLING_PROMPT.replace(' ','_').replace('&','and')}_{target_keyframe_fps:.2f}fps{'_{}compressed'.format('' if not compress else '')}.mp4"
    clips = [] 
    final_stitched_clip = None # Initialize for finally block
    original_audio = None # Initialize for finally block
    final_clip_with_audio = None # Initialize for finally block

    try:
        print("Loading video segments...")
        ideal_segment_duration = 1.0 / target_keyframe_fps
        clips = []
        for segment_path_str in tqdm(segment_paths, desc="Loading segments"):
            segment_file_path = Path(segment_path_str)
            try:
                clip = VideoFileClip(segment_path_str)

                if clip.duration == 0:
                    print(f"Warning: Segment {segment_file_path.name} has zero duration. Skipping this segment.")
                    continue

                if compress:
                    # Use 10 FPS while preserving first and last frames
                    target_frames = int(ideal_segment_duration * 10)  # 10 FPS
                    if target_frames < 2:
                        target_frames = 2  # Ensure we have at least 2 frames (first and last)
                    
                    # Calculate frame times, ensuring first and last frames are included
                    frame_times = []
                    if target_frames == 2:
                        frame_times = [0, clip.duration]
                    else:
                        # Evenly space frames between first and last
                        step = (clip.duration) / (target_frames - 1)
                        frame_times = [i * step for i in range(target_frames)]
                    
                    # Create new clip with selected frames
                    clip = clip.set_fps(10)  # Set to 10 FPS
                    clip = clip.set_duration(ideal_segment_duration)
                else:
                    # Original speed adjustment logic
                    if abs(clip.duration - ideal_segment_duration) > 0.01: # 10ms tolerance
                        speed_factor = clip.duration / ideal_segment_duration
                        if speed_factor <= 0:
                            print(f"Warning: Calculated invalid speed factor ({speed_factor:.2f}) for segment {segment_file_path.name}. Skipping speed adjustment.")
                        else:
                            print(f"Adjusting speed of segment {segment_file_path.name} by {speed_factor:.2f}x (original: {clip.duration:.2f}s, target: {ideal_segment_duration:.2f}s)")
                            clip = clip.fx(vfx.speedx, speed_factor)
                clips.append(clip)
            except Exception as e:
                print(f"Error loading or processing segment {segment_file_path.name}: {e}. Skipping this segment.")

        if not clips: print("No clips loaded or all skipped, cannot stitch."); return
            
        final_stitched_clip = concatenate_videoclips(clips, method="compose")
        
        print("Loading audio from original video...")
        original_audio = AudioFileClip(original_video_path_str)
        
        print("Setting audio for the final clip...")
        final_clip_with_audio = final_stitched_clip.set_audio(original_audio)

        print(f"Writing final video to {final_video_path}...")
        final_clip_with_audio.write_videofile(
            str(final_video_path), codec="libx264", audio_codec="aac",
            temp_audiofile=str(FINAL_VIDEOS_ROOT_DIR / f"{final_video_name_base}_temp_audio.m4a"),
            remove_temp=True, threads=os.cpu_count() or 4, logger='bar',
            fps=10,  # Set output to 10 FPS
            bitrate="2000k" if compress else None  # Reduced bitrate since we're using lower FPS
        )
        print(f"Final video saved successfully: {final_video_path}")
    except Exception as e:
        print(f"Error during video stitching or finalization: {e}")
        import traceback; traceback.print_exc()
    finally:
        for clip_obj in clips: clip_obj.close()
        if original_audio: original_audio.close()
        if final_stitched_clip: final_stitched_clip.close()
        if final_clip_with_audio: final_clip_with_audio.close()


def create_keyframe_slideshow(keyframe_paths: list, original_video_path_str: str, final_video_name_base: str, target_keyframe_fps: float):
    print(f"\n--- Creating Keyframe Slideshow for {final_video_name_base} ---")
    if not keyframe_paths: print("No keyframes to process. Exiting."); return

    ensure_dir(FINAL_VIDEOS_ROOT_DIR)
    final_video_path = FINAL_VIDEOS_ROOT_DIR / f"{final_video_name_base}_keyframes_only_{target_keyframe_fps:.2f}fps.mp4"
    
    try:
        print("Loading keyframes...")
        clips = []
        frame_duration = 1.0 / target_keyframe_fps  # Duration for each frame
        
        for keyframe_path in tqdm(keyframe_paths, desc="Loading keyframes"):
            try:
                # Use ImageClip instead of VideoFileClip for image files
                clip = ImageClip(keyframe_path).set_duration(frame_duration)
                clips.append(clip)
            except Exception as e:
                print(f"Error loading keyframe {keyframe_path}: {e}. Skipping this frame.")
                continue

        if not clips: print("No clips loaded, cannot create slideshow."); return
            
        final_clip = concatenate_videoclips(clips, method="compose")
        
        print("Loading audio from original video...")
        original_audio = AudioFileClip(original_video_path_str)
        
        print("Setting audio for the final clip...")
        final_clip_with_audio = final_clip.set_audio(original_audio)

        print(f"Writing final video to {final_video_path}...")
        final_clip_with_audio.write_videofile(
            str(final_video_path), codec="libx264", audio_codec="aac",
            temp_audiofile=str(FINAL_VIDEOS_ROOT_DIR / f"{final_video_name_base}_temp_audio.m4a"),
            remove_temp=True, threads=os.cpu_count() or 4, logger='bar',
            fps=target_keyframe_fps  # Use the keyframe FPS as the video FPS
        )
        print(f"Final video saved successfully: {final_video_path}")
    except Exception as e:
        print(f"Error during slideshow creation: {e}")
        import traceback; traceback.print_exc()
    finally:
        for clip in clips: clip.close()
        if 'final_clip' in locals(): final_clip.close()
        if 'original_audio' in locals(): original_audio.close()
        if 'final_clip_with_audio' in locals(): final_clip_with_audio.close()


def create_enhanced_slideshow(keyframe_paths: list, original_video_path_str: str, final_video_name_base: str, target_keyframe_fps: float):
    print(f"\n--- Creating Enhanced Slideshow for {final_video_name_base} ---")
    if not keyframe_paths: print("No keyframes to process. Exiting."); return

    ensure_dir(FINAL_VIDEOS_ROOT_DIR)
    final_video_path = FINAL_VIDEOS_ROOT_DIR / f"{final_video_name_base}_enhanced_slideshow_{target_keyframe_fps:.2f}fps.mp4"
    
    try:
        print("Loading keyframes and checking for Kling segments...")
        clips = []
        frame_duration = 1.0 / target_keyframe_fps  # Duration for each frame
        video_name = Path(original_video_path_str).stem
        segments_dir = KLING_SEGMENTS_ROOT_DIR / video_name
        
        for i in range(len(keyframe_paths)):
            current_keyframe = keyframe_paths[i]
            try:
                # Add the current keyframe
                keyframe_clip = ImageClip(current_keyframe).set_duration(frame_duration)
                clips.append(keyframe_clip)
                
                # If this isn't the last keyframe, check for a Kling segment
                if i < len(keyframe_paths) - 1:
                    next_keyframe = keyframe_paths[i + 1]
                    # Construct the expected Kling segment filename
                    current_name = Path(current_keyframe).stem
                    next_name = Path(next_keyframe).stem
                    kling_segment_name = f"segment_styled_{current_name}_to_{next_name}.mp4"
                    kling_segment_path = segments_dir / kling_segment_name
                    
                    if kling_segment_path.exists():
                        print(f"Found Kling segment between {current_name} and {next_name}, sampling frames...")
                        # Load the Kling segment
                        kling_clip = VideoFileClip(str(kling_segment_path))
                        
                        # Calculate how many frames we want to sample
                        num_frames_to_sample = 10  # We'll take 10 frames from the Kling segment
                        frame_interval = kling_clip.duration / (num_frames_to_sample + 1)
                        
                        # Sample frames evenly throughout the segment
                        for j in range(1, num_frames_to_sample + 1):
                            frame_time = j * frame_interval
                            frame = kling_clip.get_frame(frame_time)
                            frame_clip = ImageClip(frame).set_duration(frame_duration / (num_frames_to_sample + 1))
                            clips.append(frame_clip)
                        
                        kling_clip.close()
                    else:
                        print(f"No Kling segment found between {current_name} and {next_name}, skipping...")
                
            except Exception as e:
                print(f"Error processing keyframe {current_keyframe}: {e}. Skipping this frame.")
                continue

        if not clips: print("No clips loaded, cannot create slideshow."); return
            
        final_clip = concatenate_videoclips(clips, method="compose")
        
        print("Loading audio from original video...")
        original_audio = AudioFileClip(original_video_path_str)
        
        print("Setting audio for the final clip...")
        final_clip_with_audio = final_clip.set_audio(original_audio)

        print(f"Writing final video to {final_video_path}...")
        final_clip_with_audio.write_videofile(
            str(final_video_path), codec="libx264", audio_codec="aac",
            temp_audiofile=str(FINAL_VIDEOS_ROOT_DIR / f"{final_video_name_base}_temp_audio.m4a"),
            remove_temp=True, threads=os.cpu_count() or 4, logger='bar',
            fps=target_keyframe_fps  # Use the keyframe FPS as the video FPS
        )
        print(f"Final video saved successfully: {final_video_path}")
    except Exception as e:
        print(f"Error during enhanced slideshow creation: {e}")
        import traceback; traceback.print_exc()
    finally:
        for clip in clips: clip.close()
        if 'final_clip' in locals(): final_clip.close()
        if 'original_audio' in locals(): original_audio.close()
        if 'final_clip_with_audio' in locals(): final_clip_with_audio.close()


def create_kling_combined_video(original_video_path_str: str, final_video_name_base: str):
    print(f"\n--- Creating Combined Kling Video for {final_video_name_base} ---")
    
    ensure_dir(FINAL_VIDEOS_ROOT_DIR)
    final_video_path = FINAL_VIDEOS_ROOT_DIR / f"{final_video_name_base}_kling_combined.mp4"
    
    try:
        # Get original video duration
        original_clip = VideoFileClip(original_video_path_str)
        target_duration = original_clip.duration
        original_clip.close()
        
        print(f"Target duration from original video: {target_duration:.2f} seconds")
        
        # Find all Kling segments
        video_name = Path(original_video_path_str).stem
        segments_dir = KLING_SEGMENTS_ROOT_DIR / video_name
        if not segments_dir.exists():
            print("Error: No Kling segments directory found.")
            return
            
        kling_segments = sorted([str(p) for p in segments_dir.glob("segment_*.mp4")])
        if not kling_segments:
            print("Error: No Kling segments found.")
            return
            
        print(f"Found {len(kling_segments)} Kling segments")
        
        # Load all segments
        print("Loading Kling segments...")
        clips = []
        for segment_path in tqdm(kling_segments, desc="Loading segments"):
            try:
                clip = VideoFileClip(segment_path)
                clips.append(clip)
            except Exception as e:
                print(f"Error loading segment {segment_path}: {e}. Skipping this segment.")
                continue
                
        if not clips:
            print("No clips loaded, cannot create combined video.")
            return
            
        # Combine all segments
        print("Combining segments...")
        combined_clip = concatenate_videoclips(clips, method="compose")
        
        # Calculate speed factor to match original duration
        speed_factor = combined_clip.duration / target_duration
        print(f"Adjusting speed by factor {speed_factor:.2f}x to match original duration")
        final_clip = combined_clip.fx(vfx.speedx, speed_factor)
        
        print("Loading audio from original video...")
        original_audio = AudioFileClip(original_video_path_str)
        
        print("Setting audio for the final clip...")
        final_clip_with_audio = final_clip.set_audio(original_audio)
        
        print(f"Writing final video to {final_video_path}...")
        final_clip_with_audio.write_videofile(
            str(final_video_path), codec="libx264", audio_codec="aac",
            temp_audiofile=str(FINAL_VIDEOS_ROOT_DIR / f"{final_video_name_base}_temp_audio.m4a"),
            remove_temp=True, threads=os.cpu_count() or 4, logger='bar'
        )
        print(f"Final video saved successfully: {final_video_path}")
        
    except Exception as e:
        print(f"Error during video creation: {e}")
        import traceback; traceback.print_exc()
    finally:
        for clip in clips: clip.close()
        if 'combined_clip' in locals(): combined_clip.close()
        if 'final_clip' in locals(): final_clip.close()
        if 'original_audio' in locals(): original_audio.close()
        if 'final_clip_with_audio' in locals(): final_clip_with_audio.close()


def main():
    parser = argparse.ArgumentParser(description="Full pipeline: Keyframes -> OpenAI Style -> Kling Segments -> Final Video.")
    parser.add_argument("--video_path", default="dmb.mp4", type=str, help="Path to input MP4.")
    parser.add_argument("--keyframe_fps", default=1, type=float, help="Target FPS for keyframes (e.g., 0.5 for 1 keyframe every 2s).")
    parser.add_argument("--skipopenai", action="store_true", help="Skip OpenAI styling stage and use raw keyframes directly.")
    parser.add_argument("--skipkling", action="store_true", help="Skip Kling segment generation and use existing segments only.")
    parser.add_argument("--compress", action="store_true", help="Compress output video by sampling fewer frames and controlling bitrate.")
    parser.add_argument("--just_key_frames", action="store_true", help="Create a simple slideshow of keyframes without Kling animation.")
    parser.add_argument("--enhanced_slideshow", action="store_true", help="Create a slideshow with keyframes and sampled frames from Kling segments when available.")
    parser.add_argument("--combine_kling", action="store_true", help="Combine all Kling segments and match original video duration.")
    
    args = parser.parse_args()
    if not Path(args.video_path).exists(): print(f"Error: Video file not found: {args.video_path}"); return
    if args.keyframe_fps <= 0: print(f"Error: Keyframe FPS must be positive: {args.keyframe_fps}"); return

    video_name = Path(args.video_path).stem

    # If combine_kling is set, just combine the Kling segments and exit
    if args.combine_kling:
        create_kling_combined_video(args.video_path, video_name)
        return

    # --- Run Pipeline ---
    # Phase 1: Extract Keyframes
    raw_keyframe_paths, expected_raw_kf_count = extract_keyframes(args.video_path, args.keyframe_fps, video_name)
    
    # Phase 2: OpenAI Styling (or skip)
    if args.skipopenai:
        print("\n--- Skipping OpenAI styling stage as requested ---")
        openai_styled_keyframe_paths = raw_keyframe_paths
        print(f"Using {len(openai_styled_keyframe_paths)} raw keyframes directly.")
    else:
        print(f"Proceeding to OpenAI styling with {expected_raw_kf_count} keyframes...")
        openai_styled_keyframe_paths = run_openai_on_keyframes(raw_keyframe_paths, video_name, expected_raw_kf_count)
        
        if len(openai_styled_keyframe_paths) != expected_raw_kf_count:
            print(f"OpenAI styling stage is incomplete. Expected {expected_raw_kf_count} styled keyframes, but successfully processed/found {len(openai_styled_keyframe_paths)}.")
            return
            
        print(f"OpenAI styling stage complete: {len(openai_styled_keyframe_paths)} styled keyframes ready.")

    # If enhanced_slideshow is set, create enhanced slideshow and exit
    if args.enhanced_slideshow:
        create_enhanced_slideshow(openai_styled_keyframe_paths, args.video_path, video_name, args.keyframe_fps)
        return
    # If just_key_frames is set, create a simple slideshow and exit
    elif args.just_key_frames:
        create_keyframe_slideshow(openai_styled_keyframe_paths, args.video_path, video_name, args.keyframe_fps)
        return

    # Phase 3: Kling Segment Generation (or skip)
    if args.skipkling:
        print("\n--- Skipping Kling segment generation as requested ---")
        # Find existing Kling segments
        segments_dir = KLING_SEGMENTS_ROOT_DIR / video_name
        if not segments_dir.exists():
            print("Error: No existing Kling segments directory found. Cannot skip Kling generation without existing segments.")
            return
        kling_segment_paths = sorted([str(p) for p in segments_dir.glob("segment_*.mp4")])
        if not kling_segment_paths:
            print("Error: No existing Kling segments found. Cannot skip Kling generation without existing segments.")
            return
        print(f"Found {len(kling_segment_paths)} existing Kling segments.")
    else:
        kling_segment_paths = generate_kling_segments(openai_styled_keyframe_paths, args.keyframe_fps, video_name)
        
        expected_num_segments = len(openai_styled_keyframe_paths) - 1
        # expected_num_segments will be >= 1 at this point due to the len(openai_styled_keyframe_paths) < 2 check above.

        if len(kling_segment_paths) != expected_num_segments:
            print(f"Kling segment generation stage is incomplete. Expected {expected_num_segments} segments, but successfully processed/found {len(kling_segment_paths)}.")
            return

        print(f"Kling segment generation stage complete: {len(kling_segment_paths)} segments ready for finalization.")
    
    # Phase 4: Stitching and Finalization
    stitch_and_finalize(kling_segment_paths, args.video_path, video_name, args.keyframe_fps, args.compress)
    
    print("\nProcessing finished.")

if __name__ == "__main__":
    ensure_dir(KEYFRAMES_ROOT_DIR)
    ensure_dir(OPENAI_KEYFRAMES_ROOT_DIR)
    ensure_dir(KLING_SEGMENTS_ROOT_DIR)
    ensure_dir(FINAL_VIDEOS_ROOT_DIR)
    main()
