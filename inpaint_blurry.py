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


base_restore_prompt = (
    "Improve image clarity by reducing blur (motion blur and defocus blur), "
    "recovering natural sharpness and fine details where supported by context. "
    "Preserve scene layout, object geometry, camera perspective, and composition. "
    "Keep colors and exposure natural without over-saturation or over-sharpening. "
    "Do not hallucinate new objects, text, logos, or structures. "
    "Do not crop, rotate, or change aspect ratio."
)
judge_prompt = (
    "You are a strict reviewer for image clarity restoration quality. "
    "You will receive two images: first is the original blurry input, second is the restored result. "
    "Check the restored result for these requirements: "
    "1) clarity is improved versus the input; "
    "2) no major artifacts are introduced (ringing, halos, double edges, warped regions, fake textures); "
    "3) no hallucinated objects/text/logos are added; "
    "4) scene geometry and overall content remain consistent with the input. "
    "Reply in this exact format: first line is KEEP or RETRY, second line starts with REASON: "
    "and gives one concise failure reason if RETRY (or OK if KEEP)."
)
corrective_prompt_system = (
    "You are improving an image-restoration prompt after a failed attempt. "
    "You will receive the original blurry image and the failed restored image. "
    "Generate a corrected prompt for the next try. "
    "The corrected prompt must target the observed failure while preserving constraints: "
    "improve clarity, avoid artifacts/hallucinations, keep scene geometry/content consistent, "
    "and keep natural color/exposure. "
    "Return only the corrected prompt text."
)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def judge_restored_result(original_image, generated_image, model):
    response = get_client().models.generate_content(
        model=model,
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


def build_corrective_prompt(original_image, generated_image, prior_prompt, failure_reason, model):
    corrective_request = (
        f"Previous restoration prompt:\n{prior_prompt}\n\n"
        f"Observed failure reason:\n{failure_reason}\n\n"
        "Write an improved retry prompt now."
    )
    response = get_client().models.generate_content(
        model=model,
        contents=[corrective_prompt_system, corrective_request, original_image, generated_image],
    )
    text = (response.text or "").strip()
    return text or prior_prompt


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


def extract_result_image(response):
    parts = getattr(response, "parts", None) or []
    for part in parts:
        text = getattr(part, "text", None)
        if text:
            print(text)
        result_image = part_to_pil_image(part)
        if result_image is not None:
            return result_image

    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        candidate_parts = getattr(content, "parts", None) if content else None
        if not candidate_parts:
            continue
        for part in candidate_parts:
            text = getattr(part, "text", None)
            if text:
                print(text)
            result_image = part_to_pil_image(part)
            if result_image is not None:
                return result_image

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--downsample_ratio", type=int, default=1, help="Stride for selecting images from the directory.")
    parser.add_argument("--max_tries", type=int, default=3, help="Maximum restoration attempts per image.")
    parser.add_argument(
        "--restore_model",
        type=str,
        default="gemini-2.5-flash-image",
        help="Gemini model used for image restoration.",
    )
    parser.add_argument(
        "--judge_model",
        type=str,
        default="gemini-2.5-flash",
        help="Gemini model used for quality judging and prompt correction.",
    )
    parser.add_argument(
        "--extra_prompt",
        type=str,
        default="",
        help="Optional extra instructions appended to the base restoration prompt.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or (args.image_dir.parent / "images_restored")
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = list(iter_image_paths(args.image_dir, args.downsample_ratio))

    for image_path in tqdm(image_paths, desc="Restoring", unit="image"):
        image = Image.open(image_path).convert("RGB")
        target_size = image.size
        output_path = output_dir / image_path.name

        current_prompt = base_restore_prompt
        if args.extra_prompt.strip():
            current_prompt = f"{current_prompt} {args.extra_prompt.strip()}"

        saved = False
        for attempt_idx in range(args.max_tries):
            response = get_client().models.generate_content(
                model=args.restore_model,
                contents=[current_prompt, image],
            )

            result_image = extract_result_image(response)
            if result_image is None:
                print(
                    f"No generated image for {image_path.name} "
                    f"(attempt {attempt_idx + 1}/{args.max_tries})."
                )
                continue

            result_image = result_image.convert("RGB").resize(target_size, resample=Image.LANCZOS)
            keep, reason = judge_restored_result(
                original_image=image,
                generated_image=result_image,
                model=args.judge_model,
            )
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
                    model=args.judge_model,
                )

        if not saved:
            print(f"FAILED to restore {image_path.name}")


if __name__ == "__main__":
    main()
