"""Text and vision generation using MLX (GPU-accelerated on Apple Silicon).

Supports:
- Qwen3-4B: Text generation
- Gemma-3-4B: Vision-language (text + image)
"""

import argparse
import sys
from typing import Optional

from mlx_lm import generate as lm_generate
from mlx_lm import load as lm_load
from mlx_lm.sample_utils import make_sampler


# Available models
MODELS = {
    "qwen": "mlx-community/Qwen3-4B-8bit",
    "gemma": "mlx-community/gemma-3-4b-it-8bit",
}

DEFAULT_MODEL = "qwen"


def generate_text(
    prompt: str,
    model_name: str = DEFAULT_MODEL,
    max_tokens: int = 512,
    temperature: float = 0.7,
    verbose: bool = True,
) -> str:
    """Generate text using Qwen3-4B model.

    Args:
        prompt: The input prompt or question.
        model_name: Model key or HuggingFace path.
        max_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative).
        verbose: Whether to print tokens as they're generated.

    Returns:
        Generated text response.
    """
    model_path = MODELS.get(model_name, model_name)

    print(f"Loading model: {model_path}")
    print("Using Apple Silicon GPU (Metal) 🚀")
    print("-" * 50)

    # Load model and tokenizer
    model, tokenizer = lm_load(model_path)

    # Apply chat template if available
    if tokenizer.chat_template is not None:
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
    else:
        formatted_prompt = prompt

    # Create sampler with temperature
    sampler = make_sampler(temp=temperature)

    # Generate response
    response = lm_generate(
        model,
        tokenizer,
        prompt=formatted_prompt,
        max_tokens=max_tokens,
        sampler=sampler,
        verbose=verbose,
    )

    return response


def generate_with_image(
    prompt: str,
    image_path: str,
    model_name: str = "gemma",
    max_tokens: int = 512,
    temperature: float = 0.0,
    verbose: bool = True,
) -> str:
    """Generate text from image using Gemma-3-4B vision model.

    Args:
        prompt: The input prompt describing what to do with the image.
        image_path: Path to the image file.
        model_name: Model key or HuggingFace path.
        max_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature.
        verbose: Whether to print tokens as they're generated.

    Returns:
        Generated text response.
    """
    from mlx_vlm import generate as vlm_generate
    from mlx_vlm import load as vlm_load
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_config

    model_path = MODELS.get(model_name, model_name)

    print(f"Loading vision model: {model_path}")
    print(f"Image: {image_path}")
    print("Using Apple Silicon GPU (Metal) 🚀")
    print("-" * 50)

    # Load model, processor, and config
    model, processor = vlm_load(model_path)
    config = load_config(model_path)

    # Apply chat template with image
    formatted_prompt = apply_chat_template(
        processor,
        config,
        prompt,
        num_images=1,
    )

    # Generate response
    response = vlm_generate(
        model,
        processor,
        formatted_prompt,
        image=image_path,
        max_tokens=max_tokens,
        temperature=temperature,
        verbose=verbose,
    )

    return response


def interactive_chat(
    model_name: str = DEFAULT_MODEL,
    max_tokens: int = 512,
    temperature: float = 0.7,
) -> None:
    """Run an interactive chat session with the model.

    Args:
        model_name: Model key or HuggingFace path.
        max_tokens: Maximum number of tokens to generate per response.
        temperature: Sampling temperature.
    """
    model_path = MODELS.get(model_name, model_name)

    print(f"Loading model: {model_path}")
    print("Using Apple Silicon GPU (Metal) 🚀")

    # Load model and tokenizer
    model, tokenizer = lm_load(model_path)

    # Create sampler with temperature
    sampler = make_sampler(temp=temperature)

    print("\n" + "=" * 50)
    print("Interactive Chat (type 'quit' or 'exit' to stop)")
    print("=" * 50 + "\n")

    conversation: list[dict[str, str]] = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        # Add user message to conversation
        conversation.append({"role": "user", "content": user_input})

        # Apply chat template
        if tokenizer.chat_template is not None:
            formatted_prompt = tokenizer.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=False,
            )
        else:
            formatted_prompt = user_input

        # Generate response
        print("\nAssistant: ", end="", flush=True)
        response = lm_generate(
            model,
            tokenizer,
            prompt=formatted_prompt,
            max_tokens=max_tokens,
            sampler=sampler,
            verbose=True,
        )

        # Add assistant response to conversation
        conversation.append({"role": "assistant", "content": response})
        print("\n")


def main() -> None:
    """Main entry point for the generation script."""
    parser = argparse.ArgumentParser(
        description="Text/Vision generation using MLX on Apple Silicon GPU",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Text generation (Qwen3)
  %(prog)s "What is Python?"
  %(prog)s -i                              # Interactive chat

  # Vision-language (Gemma-3)
  %(prog)s "Describe this image" --image photo.jpg
  %(prog)s "What's in this picture?" --image screenshot.png

Models:
  qwen   - Qwen3-4B-8bit (text only, default)
  gemma  - Gemma-3-4B-it-8bit (vision + text)
        """,
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        default=None,
        help="The prompt or question",
    )
    parser.add_argument(
        "-m", "--model",
        default=DEFAULT_MODEL,
        choices=["qwen", "gemma"],
        help="Model to use (default: qwen)",
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Path to image (uses Gemma vision model)",
    )
    parser.add_argument(
        "-t", "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate (default: 512)",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=0.7,
        help="Sampling temperature 0.0-1.0 (default: 0.7)",
    )
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Run in interactive chat mode (text only)",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Don't stream tokens (print final response only)",
    )

    args = parser.parse_args()

    # Interactive mode
    if args.interactive:
        interactive_chat(
            model_name=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temp,
        )
        return

    # Single prompt mode
    if args.prompt is None:
        parser.print_help()
        print("\nError: Please provide a prompt or use -i for interactive mode")
        sys.exit(1)

    try:
        # Vision mode (with image)
        if args.image:
            response = generate_with_image(
                prompt=args.prompt,
                image_path=args.image,
                model_name="gemma",  # Always use Gemma for images
                max_tokens=args.max_tokens,
                temperature=args.temp,
                verbose=not args.quiet,
            )
        # Text mode
        else:
            response = generate_text(
                prompt=args.prompt,
                model_name=args.model,
                max_tokens=args.max_tokens,
                temperature=args.temp,
                verbose=not args.quiet,
            )

        if args.quiet:
            print(response)

    except KeyboardInterrupt:
        print("\nInterrupted")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
