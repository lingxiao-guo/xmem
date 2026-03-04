import argparse
import io
from pathlib import Path

from google import genai
from PIL import Image
from tqdm import tqdm

_client = None


def get_client():
    global _client
    if _client is None:
        try:
            _client = genai.Client()
        except ValueError as exc:
            raise ValueError(
                "Failed to initialize Gemini client. Set `GEMINI_API_KEY` in the "
                "environment (or configure Vertex AI parameters) and rerun."
            ) from exc
    return _client


base_inpaint_prompt = (
    "Remove all bright green masked region(s). "
    "Also remove any visible robot parts in the foreground, including arm(s) and any other body parts such as torso, mobile base, or links if present. "
    "Most images may only contain arm(s), but if additional robot body parts are visible, remove them too. "
    "Reconstruct the missing background so it looks natural and seamless. "
    "The replacement should match surrounding areas. "
    "IMPORTANT: Do not modify pixels outside removed masked/robot regions."
)
judge_prompt = (
    "You are a strict image reviewer for inpainting quality. "
    "You will be given two images: first is the original input, second is the generated result. "
    "Check the generated result for these requirements: "
    "1) all bright green masked regions are fully removed; "
    "2) any visible robot parts (arms, torso, base, links, or other body parts) are fully removed; "
    "3) the filled background looks natural and seamless relative to nearby pixels; "
    "4) no changes occur outside the originally masked/robot regions when compared to the original. "
    "Reply in this exact format: first line is KEEP or RETRY, second line starts with REASON: and gives one concise failure reason if RETRY (or OK if KEEP)."
)
corrective_prompt_system = (
    "You are improving an image-inpainting prompt after a failed attempt. "
    "You will receive the original image and the failed generated image. "
    "Generate a corrected prompt for the next inpainting try. "
    "The corrected prompt must focus on fixing the failure while preserving all constraints. "
    "Always require removal of green masked regions and visible robot parts (arms, torso, base, links, or other body parts), "
    "natural/seamless background completion, and no edits outside masked/robot regions. "
    "Return only the corrected prompt text."
)


def judge_inpainted_result(original_image, generated_image):
    response = get_client().models.generate_content(
        model="gemini-2.5-flash",
        contents=[judge_prompt, original_image, generated_image],
    )
    text = (response.text or "").strip()
    if not text:
        return False, "No judgment text returned."

    first_line = text.splitlines()[0].strip().upper()
    keep = first_line.startswith("KEEP")

    reason = ""
    for line in text.splitlines()[1:]:
        if line.strip().upper().startswith("REASON:"):
            reason = line.split(":", 1)[1].strip()
            break
    if not reason:
        reason = "No explicit reason returned."
    return keep, reason


def build_corrective_prompt(original_image, generated_image, prior_prompt, failure_reason):
    corrective_request = (
        f"Previous inpainting prompt:\n{prior_prompt}\n\n"
        f"Observed failure reason:\n{failure_reason}\n\n"
        "Write an improved retry prompt now."
    )
    response = get_client().models.generate_content(
        model="gemini-2.5-flash",
        contents=[corrective_prompt_system, corrective_request, original_image, generated_image],
    )
    text = (response.text or "").strip()
    return text or prior_prompt

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def iter_image_paths(image_dir, downsample_ratio):
    if downsample_ratio <= 0:
        raise ValueError("downsample_ratio must be positive.")
    if not image_dir.is_dir():
        raise ValueError(f"Image directory not found: {image_dir}")

    image_paths = sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise ValueError(f"No images found in {image_dir}")

    for path in image_paths[::downsample_ratio]:
        yield path


def part_to_pil_image(part):
    try:
        image_obj = part.as_image()
    except Exception:
        image_obj = None

    if image_obj is not None:
        image_bytes = getattr(image_obj, "image_bytes", None)
        if image_bytes:
            return Image.open(io.BytesIO(image_bytes))

    inline_data = getattr(part, "inline_data", None)
    inline_bytes = getattr(inline_data, "data", None) if inline_data else None
    if inline_bytes:
        return Image.open(io.BytesIO(inline_bytes))

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=Path, required=True)
    parser.add_argument("--downsample_ratio", type=int, default=1, help="Stride for selecting images from the directory.")
    parser.add_argument("--max_tries", type=int, default=3, help="Maximum inpainting attempts per image.")
    args = parser.parse_args()

    output_dir = args.image_dir.parent / "images"
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = list(iter_image_paths(args.image_dir, args.downsample_ratio))

    for image_path in tqdm(
        image_paths,
        desc="Inpainting",
        unit="image",
    ):
        image = Image.open(image_path).convert("RGB")
        target_size = image.size
        output_path = output_dir / image_path.name

        current_prompt = base_inpaint_prompt
        saved = False
        for attempt_idx in range(args.max_tries):
            response = get_client().models.generate_content(
                model="gemini-3.1-flash-image", # for harder tasks use "gemini-3-pro-image-preview"
                contents=[current_prompt, image],
            )

            result_image = None
            for part in response.parts:
                if part.text is not None:
                    print(part.text)
                elif part.inline_data is not None:
                    result_image = part_to_pil_image(part)
                    if result_image is not None:
                        break

            if result_image is None:
                print(f"No generated image for {image_path.name} (attempt {attempt_idx + 1}/{args.max_tries}).")
                continue

            result_image = result_image.resize(target_size, resample=Image.LANCZOS)
            keep, reason = judge_inpainted_result(image, result_image)
            result_image.save(output_path)
            if keep:
                result_image.save(output_path)
                saved = True
                break

            if attempt_idx < args.max_tries - 1:
                current_prompt = build_corrective_prompt(
                    original_image=image,
                    generated_image=result_image,
                    prior_prompt=current_prompt,
                    failure_reason=reason,
                )

        if not saved:
            print(f"FAILED to inpaint {image_path.name}")


if __name__ == "__main__":
    main()
