"""네 모델의 마지막 convolution feature를 이용해 Grad-CAM overlay를 저장한다."""

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image

from src.config import get_checkpoint_path, load_config
from src.data.dataset import IMAGE_SIZE, create_eval_transform
from src.models import MODEL_NAMES, create_model
from src.pipeline.train import get_device


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "plots" / "gradcam"


def get_default_target_layer(model, model_name: str):
    """각 구조에서 공간 특징이 남아 있는 마지막 convolution block을 선택한다."""
    if model_name == "cnn":
        return model.features[-1]
    if model_name == "resnet18":
        return model.layer4[-1]
    if model_name in {"efficientnet_b0", "mobilenet_v3"}:
        return model.features[-1]
    raise ValueError(f"지원하지 않는 모델입니다: {model_name}")


def get_layer_by_name(model, layer_name: str):
    """예: layer4.1 또는 features.8 형태로 target layer를 직접 찾는다."""
    layer = model
    for part in layer_name.split("."):
        if part.isdigit():
            layer = layer[int(part)]
        elif hasattr(layer, part):
            layer = getattr(layer, part)
        else:
            raise ValueError(f"target layer를 찾을 수 없습니다: {layer_name}")
    return layer


def make_gradcam(
    model,
    target_layer,
    input_tensor,
    target_index: int,
    output_size: int = IMAGE_SIZE,
) -> np.ndarray:
    """target class gradient와 activation으로 0~1 Grad-CAM map을 계산한다."""
    captured = {}

    def save_activation(_module, _inputs, output):
        captured["activation"] = output
        output.register_hook(lambda gradient: captured.update(gradient=gradient))

    handle = target_layer.register_forward_hook(save_activation)
    try:
        model.zero_grad()
        outputs = model(input_tensor)
        outputs[0, target_index].backward()
    finally:
        handle.remove()

    if "activation" not in captured or "gradient" not in captured:
        raise RuntimeError("선택한 layer에서 activation/gradient를 얻지 못했습니다.")

    activations = captured["activation"]
    gradients = captured["gradient"]
    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * activations).sum(dim=1, keepdim=True))
    cam = functional.interpolate(
        cam, size=(output_size, output_size), mode="bilinear", align_corners=False
    )[0, 0]
    cam -= cam.min()
    maximum = cam.max()
    if maximum > 0:
        cam /= maximum
    return cam.detach().cpu().numpy()


def parse_target_class(
    target_class: Optional[str], class_names, predicted_index: int
) -> int:
    """클래스명/인덱스 입력을 target index로 바꾸고, 생략 시 예측 클래스를 쓴다."""
    if target_class is None:
        return predicted_index
    if target_class.isdigit():
        index = int(target_class)
        if 0 <= index < len(class_names):
            return index
    elif target_class in class_names:
        return class_names.index(target_class)
    raise ValueError(
        f"알 수 없는 target class: {target_class}. 클래스명 또는 0~{len(class_names)-1} 사용"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="학습된 모델의 Grad-CAM을 생성합니다.")
    parser.add_argument("--config", type=Path, help="사용할 JSON config 파일")
    parser.add_argument("--model", choices=MODEL_NAMES, help="config의 model을 덮어씁니다.")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--target-class", help="클래스명 또는 인덱스. 기본값은 예측 클래스")
    parser.add_argument(
        "--target-layer", help="선택 사항. 예: ResNet18의 layer4.1"
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.config is None and args.model is None:
        raise SystemExit("--config 또는 --model 중 하나는 반드시 지정해야 합니다.")
    if not args.image.is_file():
        raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {args.image}")

    if args.config is not None:
        config = load_config(args.config)
        model_name = args.model or config["model"]
        image_size = config["data"]["image_size"]
        checkpoint_path = args.checkpoint or get_checkpoint_path(config, model_name)
    else:
        model_name = args.model
        image_size = IMAGE_SIZE
        checkpoint_path = args.checkpoint

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
    model = create_model(model_name, len(class_names), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    image = Image.open(args.image).convert("RGB").resize((image_size, image_size))
    input_tensor = create_eval_transform(image_size)(image).unsqueeze(0).to(device)
    with torch.no_grad():
        probabilities = torch.softmax(model(input_tensor), dim=1)
        predicted_index = probabilities.argmax(dim=1).item()
        confidence = probabilities[0, predicted_index].item()

    target_index = parse_target_class(
        args.target_class, class_names, predicted_index
    )
    target_layer = (
        get_layer_by_name(model, args.target_layer)
        if args.target_layer
        else get_default_target_layer(model, model_name)
    )
    cam = make_gradcam(
        model, target_layer, input_tensor, target_index, output_size=image_size
    )

    rgb_image = np.asarray(image, dtype=np.float32) / 255.0
    heatmap = plt.get_cmap("jet")(cam)[..., :3]
    overlay = np.clip(0.55 * rgb_image + 0.45 * heatmap, 0, 1)

    output_path = args.output or (
        OUTPUT_DIR / f"gradcam_{model_name}_{args.image.stem}.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((overlay * 255).astype(np.uint8)).save(output_path)

    print("Device:", device)
    print(f"Predicted: {class_names[predicted_index]} ({confidence * 100:.2f}%)")
    print("Grad-CAM target:", class_names[target_index])
    print("Saved:", output_path)


if __name__ == "__main__":
    main()
