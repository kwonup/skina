"""Best checkpoint로 이미지 한 장을 추론하고 config의 Top-K를 출력한다."""

import argparse
from pathlib import Path

from PIL import Image

from src.config import get_checkpoint_path, load_config
from src.models import MODEL_NAMES
from src.pipeline.inference import load_inference_model


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="피부 이미지 한 장의 Top-K를 예측합니다.")
    parser.add_argument("--config", type=Path, help="사용할 JSON config 파일")
    parser.add_argument("--model", choices=MODEL_NAMES, help="config의 model을 덮어씁니다.")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--top-k", type=int, help="config의 top_k를 덮어씁니다.")
    parser.add_argument(
        "--checkpoint", type=Path, help="생략하면 outputs/models/{model}_best.pth"
    )
    return parser.parse_args()


def run_prediction(
    model_name: str,
    image_path: Path,
    checkpoint_path: Path = None,
    image_size: int = 224,
    top_k: int = 3,
):
    """설정된 checkpoint와 전처리로 이미지 한 장의 Top-K를 반환한다."""
    image_path = Path(image_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {image_path}")

    checkpoint_path = checkpoint_path or (
        PROJECT_ROOT / "outputs" / "models" / f"{model_name}_best.pth"
    )
    inference_model = load_inference_model(
        checkpoint_path,
        expected_model_name=model_name,
        expected_image_size=image_size,
    )
    with Image.open(image_path) as image:
        output = inference_model.predict(image, top_k=top_k)

    print("Device:", inference_model.device)
    print(f"\n=== Top-{top_k} Prediction ===")
    results = []
    for prediction in output.predictions:
        result = {
            "rank": prediction.rank,
            "class_name": prediction.class_name,
            "probability": prediction.probability,
        }
        results.append(result)
        print(
            f"{result['rank']}. {result['class_name']:<28} "
            f"{result['probability'] * 100:6.2f}%"
        )
    print(f"Inference time: {output.inference_time_ms:.2f} ms")
    return results


def main() -> None:
    args = parse_args()
    if args.config is None and args.model is None:
        raise SystemExit("--config 또는 --model 중 하나는 반드시 지정해야 합니다.")

    if args.config is not None:
        config = load_config(args.config)
        model_name = args.model or config["model"]
        checkpoint_path = args.checkpoint or get_checkpoint_path(config, model_name)
        image_size = config["data"]["image_size"]
        top_k = (
            args.top_k
            if args.top_k is not None
            else config["inference"]["top_k"]
        )
    else:
        model_name = args.model
        checkpoint_path = args.checkpoint
        image_size = 224
        top_k = args.top_k if args.top_k is not None else 3

    run_prediction(
        model_name=model_name,
        image_path=args.image,
        checkpoint_path=checkpoint_path,
        image_size=image_size,
        top_k=top_k,
    )


if __name__ == "__main__":
    main()
