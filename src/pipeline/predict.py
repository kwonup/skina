"""Best checkpoint로 이미지 한 장을 추론하고 config의 Top-K를 출력한다."""

import argparse
from pathlib import Path

import torch
from PIL import Image

from src.config import get_checkpoint_path, load_config
from src.data.dataset import create_eval_transform
from src.models import MODEL_NAMES, create_model
from src.pipeline.train import get_device


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
    if not image_path.is_file():
        raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {image_path}")

    checkpoint_path = checkpoint_path or (
        PROJECT_ROOT / "outputs" / "models" / f"{model_name}_best.pth"
    )
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint를 찾을 수 없습니다: {checkpoint_path}")

    device = get_device()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_model = checkpoint.get("model_name", model_name)
    if checkpoint_model != model_name:
        raise ValueError(
            f"요청 모델은 {model_name}이지만 checkpoint 모델은 {checkpoint_model}입니다."
        )
    if checkpoint.get("image_size", image_size) != image_size:
        raise ValueError(
            "checkpoint 학습 image_size와 config의 image_size가 다릅니다: "
            f"{checkpoint['image_size']} != {image_size}"
        )

    class_names = checkpoint["class_names"]
    if not 1 <= top_k <= len(class_names):
        raise ValueError(f"top_k는 1~{len(class_names)} 범위여야 합니다: {top_k}")

    model = create_model(model_name, len(class_names), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    image = Image.open(image_path).convert("RGB")
    image_tensor = create_eval_transform(image_size)(image).unsqueeze(0).to(device)
    with torch.no_grad():
        probabilities = torch.softmax(model(image_tensor), dim=1)
        top_probabilities, top_indices = torch.topk(
            probabilities, k=top_k, dim=1
        )

    print("Device:", device)
    print(f"\n=== Top-{top_k} Prediction ===")
    results = []
    for rank, (probability, index) in enumerate(
        zip(top_probabilities[0].cpu(), top_indices[0].cpu()), start=1
    ):
        result = {
            "rank": rank,
            "class_name": class_names[index.item()],
            "probability": probability.item(),
        }
        results.append(result)
        print(
            f"{rank}. {result['class_name']:<28} "
            f"{result['probability'] * 100:6.2f}%"
        )
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
