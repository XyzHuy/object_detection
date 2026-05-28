from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from inference import collect_images, load_annotation_index, load_model, predict_image


DEFAULT_THRESHOLDS = (
    "0.001,0.002,0.003,0.005,0.0075,0.01,0.015,0.02,0.03,0.04,"
    "0.05,0.075,0.1,0.125,0.15,0.2,0.25,0.3,0.4,0.5,0.6"
)
DEFAULT_NMS_IOUS = "0.45,0.50,0.55,0.60,0.65,0.70,0.75"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Tune per-class confidence thresholds for official mAP@0.5 or F1")
    parser.add_argument("--checkpoint", help="Checkpoint to run when --predictions is not provided.")
    parser.add_argument("--predictions", help="Existing low-threshold predictions JSON to tune offline.")
    parser.add_argument("--source", default="final_public/public/val/images")
    parser.add_argument("--data_root", default="final_public/public")
    parser.add_argument("--split", default="val")
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--base_conf", type=float, default=0.001)
    parser.add_argument("--threshold_values", default=DEFAULT_THRESHOLDS)
    parser.add_argument("--nms_iou_values", default=DEFAULT_NMS_IOUS)
    parser.add_argument("--nms_iou", type=float, default=0.65, help="Label used with --predictions mode.")
    parser.add_argument("--max_det", type=int, default=100, help="Official evaluator max detections per image.")
    parser.add_argument(
        "--nms_max_det",
        type=int,
        default=300,
        help="Detections kept after model NMS before threshold filtering.",
    )
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument(
        "--min_map50",
        type=float,
        default=None,
        help=(
            "Minimum mAP@0.5 to keep while optimizing thresholds. Required for F1 selection metrics; "
            "optional with --selection_metric map to prefer fewer predictions above this mAP floor."
        ),
    )
    parser.add_argument(
        "--selection_metric",
        choices=("map", "macro_f1", "micro_f1", "weighted_f1"),
        default="map",
        help=(
            "Threshold selection objective. Use macro_f1 for usable inference thresholds while "
            "keeping mAP above --min_map50."
        ),
    )
    parser.add_argument("--official_evaluator", default="final_public/public/tools/evaluate_predictions.py")
    parser.add_argument("--ground_truth")
    parser.add_argument("--output_dir", default="threshold_sweeps")
    parser.add_argument(
        "--save_intermediate",
        action="store_true",
        help="Also write raw/prediction/threshold JSON files for every NMS IoU candidate.",
    )
    return parser.parse_args()


def parse_float_list(text: str) -> list[float]:
    values = sorted({float(item.strip()) for item in text.split(",") if item.strip()})
    if not values:
        raise ValueError("Expected at least one float value")
    return values


def load_official_module(evaluator_path: str | Path):
    evaluator_path = Path(evaluator_path)
    spec = importlib.util.spec_from_file_location("public_evaluate_predictions", evaluator_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load official evaluator from {evaluator_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def safe_tag(value: float) -> str:
    return f"{value:.4f}".rstrip("0").rstrip(".").replace(".", "p")


def evaluate_predictions(
    module,
    ground_truth: dict[str, Any],
    classes: list[str],
    image_info: dict[str, dict[str, Any]],
    predictions: list[dict[str, Any]],
    max_det: int,
) -> dict[str, Any]:
    normalized = module.normalize_predictions(
        predictions,
        classes=classes,
        image_info=image_info,
        max_detections_per_image=max_det,
        require_complete=True,
    )
    metrics = module.evaluate(
        ground_truth=ground_truth,
        predictions=normalized,
        classes=classes,
        iou_threshold=0.5,
    )
    return add_f1_metrics(metrics)


def add_f1_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    metrics = dict(metrics)
    per_class = metrics.get("per_class", {})
    weighted_total = 0
    weighted_f1 = 0.0
    class_f1s = []

    for class_metrics in per_class.values():
        precision = float(class_metrics.get("precision", 0.0))
        recall = float(class_metrics.get("recall", 0.0))
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        class_metrics["f1"] = round(f1, 6)
        class_f1s.append(f1)

        support = int(class_metrics.get("num_ground_truth", 0))
        weighted_total += support
        weighted_f1 += f1 * support

    micro_precision = float(metrics.get("micro_precision", 0.0))
    micro_recall = float(metrics.get("micro_recall", 0.0))
    metrics["micro_f1"] = round(
        2.0 * micro_precision * micro_recall / max(micro_precision + micro_recall, 1e-12),
        6,
    )
    metrics["macro_f1"] = round(float(sum(class_f1s) / max(len(class_f1s), 1)), 6)
    metrics["weighted_f1"] = round(float(weighted_f1 / max(weighted_total, 1)), 6)
    metrics["min_class_f1"] = round(float(min(class_f1s)) if class_f1s else 0.0, 6)
    metrics["total_false_positives"] = int(
        sum(class_metrics.get("false_positives", 0) for class_metrics in per_class.values())
    )
    return metrics


def metric_is_better(candidate: dict[str, Any], current: dict[str, Any] | None) -> bool:
    if current is None:
        return True
    candidate_map = candidate["mAP@0.5"]
    current_map = current["mAP@0.5"]
    if candidate_map > current_map + 1e-12:
        return True
    if abs(candidate_map - current_map) <= 1e-12:
        return candidate["num_predictions"] < current["num_predictions"]
    return False


def f1_metric_is_better(
    candidate: dict[str, Any],
    current: dict[str, Any] | None,
    selection_metric: str,
    min_map: float,
) -> bool:
    if candidate["mAP@0.5"] + 1e-12 < min_map:
        return False
    if current is None:
        return True

    candidate_score = candidate[selection_metric]
    current_score = current[selection_metric]
    if candidate_score > current_score + 1e-12:
        return True
    if candidate_score < current_score - 1e-12:
        return False

    if candidate["min_class_f1"] > current["min_class_f1"] + 1e-12:
        return True
    if candidate["min_class_f1"] < current["min_class_f1"] - 1e-12:
        return False

    if candidate["mAP@0.5"] > current["mAP@0.5"] + 1e-12:
        return True
    if candidate["mAP@0.5"] < current["mAP@0.5"] - 1e-12:
        return False

    return candidate["num_predictions"] < current["num_predictions"]


def selected_metric_is_better(
    candidate: dict[str, Any],
    current: dict[str, Any] | None,
    selection_metric: str,
) -> bool:
    if selection_metric == "map":
        return metric_is_better(candidate, current)
    return f1_metric_is_better(
        candidate=candidate,
        current=current,
        selection_metric=selection_metric,
        min_map=float("-inf"),
    )


def filter_predictions(
    predictions: list[dict[str, Any]],
    thresholds: dict[str, float],
) -> list[dict[str, Any]]:
    filtered = []
    for entry in predictions:
        boxes = [
            box
            for box in entry["boxes"]
            if float(box["confidence"]) >= thresholds.get(box["class"], 0.0)
        ]
        boxes.sort(key=lambda item: item["confidence"], reverse=True)
        filtered.append({"image_id": entry["image_id"], "boxes": boxes})
    return filtered


def generate_predictions(
    model,
    source: str | Path,
    data_root: str | Path,
    split: str,
    classes: list[str],
    img_size: int,
    base_conf: float,
    nms_iou: float,
    nms_max_det: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    annotation_index = load_annotation_index(data_root, split)
    predictions = []
    image_paths = collect_images(source)
    for image_path in tqdm(image_paths, desc=f"predict nms={nms_iou:g}", leave=False):
        result = predict_image(
            model,
            image_path,
            classes,
            img_size,
            device,
            base_conf,
            nms_iou,
            nms_max_det,
        )
        result["image_id"] = annotation_index.get(image_path.name, {}).get("id", image_path.name)
        predictions.append(result)
    return predictions


def tune_for_predictions(
    module,
    ground_truth: dict[str, Any],
    classes: list[str],
    image_info: dict[str, dict[str, Any]],
    raw_predictions: list[dict[str, Any]],
    threshold_values: list[float],
    base_conf: float,
    max_det: int,
    rounds: int,
    min_map50: float | None,
    selection_metric: str,
) -> tuple[dict[str, float], dict[str, Any], list[dict[str, Any]], float | None]:
    threshold_values = sorted({base_conf, *threshold_values})
    threshold_values = [value for value in threshold_values if value >= base_conf - 1e-12]

    thresholds = {class_name: base_conf for class_name in classes}
    best_predictions = filter_predictions(raw_predictions, thresholds)
    best_metrics = evaluate_predictions(
        module, ground_truth, classes, image_info, best_predictions, max_det
    )
    history = [
        {
            "stage": "baseline",
            "thresholds": dict(thresholds),
            "mAP@0.5": best_metrics["mAP@0.5"],
            "macro_f1": best_metrics["macro_f1"],
            "micro_f1": best_metrics["micro_f1"],
            "min_class_f1": best_metrics["min_class_f1"],
            "num_predictions": best_metrics["num_predictions"],
        }
    ]

    for round_idx in range(rounds):
        changed = False
        for class_name in classes:
            class_best_thresholds = dict(thresholds)
            class_best_metrics = best_metrics
            for threshold in threshold_values:
                candidate_thresholds = dict(thresholds)
                candidate_thresholds[class_name] = threshold
                candidate_predictions = filter_predictions(raw_predictions, candidate_thresholds)
                candidate_metrics = evaluate_predictions(
                    module,
                    ground_truth,
                    classes,
                    image_info,
                    candidate_predictions,
                    max_det,
                )
                if metric_is_better(candidate_metrics, class_best_metrics):
                    class_best_thresholds = candidate_thresholds
                    class_best_metrics = candidate_metrics

            if class_best_thresholds != thresholds:
                thresholds = class_best_thresholds
                best_metrics = class_best_metrics
                changed = True

            history.append(
                {
                    "stage": "coordinate",
                    "round": round_idx + 1,
                    "class": class_name,
                    "threshold": thresholds[class_name],
                    "mAP@0.5": best_metrics["mAP@0.5"],
                    "macro_f1": best_metrics["macro_f1"],
                    "micro_f1": best_metrics["micro_f1"],
                    "min_class_f1": best_metrics["min_class_f1"],
                    "num_predictions": best_metrics["num_predictions"],
                }
            )

        if not changed:
            break

    target_map = min_map50
    if selection_metric == "map" and target_map is not None:
        for class_name in classes:
            class_best_thresholds = dict(thresholds)
            class_best_metrics = best_metrics
            for threshold in sorted(threshold_values, reverse=True):
                candidate_thresholds = dict(thresholds)
                candidate_thresholds[class_name] = threshold
                candidate_predictions = filter_predictions(raw_predictions, candidate_thresholds)
                candidate_metrics = evaluate_predictions(
                    module,
                    ground_truth,
                    classes,
                    image_info,
                    candidate_predictions,
                    max_det,
                )
                if candidate_metrics["mAP@0.5"] + 1e-12 < target_map:
                    continue
                if candidate_metrics["num_predictions"] < class_best_metrics["num_predictions"]:
                    class_best_thresholds = candidate_thresholds
                    class_best_metrics = candidate_metrics
                elif (
                    candidate_metrics["num_predictions"] == class_best_metrics["num_predictions"]
                    and candidate_metrics["mAP@0.5"] > class_best_metrics["mAP@0.5"]
                ):
                    class_best_thresholds = candidate_thresholds
                    class_best_metrics = candidate_metrics

            if class_best_thresholds != thresholds:
                thresholds = class_best_thresholds
                best_metrics = class_best_metrics

            history.append(
                {
                    "stage": "compression",
                    "class": class_name,
                    "threshold": thresholds[class_name],
                    "min_mAP@0.5": target_map,
                    "mAP@0.5": best_metrics["mAP@0.5"],
                    "macro_f1": best_metrics["macro_f1"],
                    "micro_f1": best_metrics["micro_f1"],
                    "min_class_f1": best_metrics["min_class_f1"],
                    "num_predictions": best_metrics["num_predictions"],
                }
            )

    if selection_metric != "map":
        if target_map is None:
            raise ValueError("--min_map50 is required when --selection_metric is an F1 metric")
        for round_idx in range(rounds):
            changed = False
            for class_name in classes:
                class_best_thresholds = dict(thresholds)
                class_best_metrics = best_metrics
                for threshold in threshold_values:
                    candidate_thresholds = dict(thresholds)
                    candidate_thresholds[class_name] = threshold
                    candidate_predictions = filter_predictions(raw_predictions, candidate_thresholds)
                    candidate_metrics = evaluate_predictions(
                        module,
                        ground_truth,
                        classes,
                        image_info,
                        candidate_predictions,
                        max_det,
                    )
                    if f1_metric_is_better(
                        candidate_metrics,
                        class_best_metrics,
                        selection_metric=selection_metric,
                        min_map=target_map,
                    ):
                        class_best_thresholds = candidate_thresholds
                        class_best_metrics = candidate_metrics

                if class_best_thresholds != thresholds:
                    thresholds = class_best_thresholds
                    best_metrics = class_best_metrics
                    changed = True

                history.append(
                    {
                        "stage": "f1_coordinate",
                        "round": round_idx + 1,
                        "class": class_name,
                        "threshold": thresholds[class_name],
                        "selection_metric": selection_metric,
                        "min_mAP@0.5": target_map,
                        "mAP@0.5": best_metrics["mAP@0.5"],
                        "macro_f1": best_metrics["macro_f1"],
                        "micro_f1": best_metrics["micro_f1"],
                        "weighted_f1": best_metrics["weighted_f1"],
                        "min_class_f1": best_metrics["min_class_f1"],
                        "num_predictions": best_metrics["num_predictions"],
                        "total_false_positives": best_metrics["total_false_positives"],
                    }
                )

            if not changed:
                break

    return thresholds, best_metrics, history, target_map


def main() -> None:
    args = parse_args()
    if not args.checkpoint and not args.predictions:
        raise ValueError("Provide either --checkpoint or --predictions")
    if args.rounds < 1:
        raise ValueError("--rounds must be >= 1")
    if args.base_conf < 0 or args.base_conf > 1:
        raise ValueError("--base_conf must be in [0, 1]")
    if args.min_map50 is not None and not 0 <= args.min_map50 <= 1:
        raise ValueError("--min_map50 must be in [0, 1]")
    if args.selection_metric != "map" and args.min_map50 is None:
        raise ValueError("Provide --min_map50 when using an F1 selection metric")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    module = load_official_module(args.official_evaluator)
    ground_truth_path = args.ground_truth or str(Path(args.data_root) / "annotations" / f"{args.split}.json")
    ground_truth = module.load_json(Path(ground_truth_path))
    classes, image_info = module.validate_ground_truth(ground_truth)
    threshold_values = parse_float_list(args.threshold_values)

    if args.predictions:
        nms_iou_values = [args.nms_iou]
        raw_by_iou = {args.nms_iou: module.load_json(Path(args.predictions))}
    else:
        nms_iou_values = parse_float_list(args.nms_iou_values)
        raw_by_iou = {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = None
    if not args.predictions:
        model, checkpoint_data = load_model(args.checkpoint, len(classes), device)
        if checkpoint_data.get("classes") and checkpoint_data["classes"] != classes:
            raise ValueError(
                f"Checkpoint classes {checkpoint_data['classes']} do not match ground truth classes {classes}"
            )

    all_results = []
    best_result = None
    best_predictions = None

    for nms_iou in nms_iou_values:
        if nms_iou in raw_by_iou:
            raw_predictions = raw_by_iou[nms_iou]
        else:
            assert model is not None
            raw_predictions = generate_predictions(
                model=model,
                source=args.source,
                data_root=args.data_root,
                split=args.split,
                classes=classes,
                img_size=args.img_size,
                base_conf=args.base_conf,
                nms_iou=nms_iou,
                nms_max_det=args.nms_max_det,
                device=device,
            )
            if args.save_intermediate:
                write_json(output_dir / f"raw_predictions_nms_{safe_tag(nms_iou)}.json", raw_predictions)

        thresholds, metrics, history, min_map50 = tune_for_predictions(
            module=module,
            ground_truth=ground_truth,
            classes=classes,
            image_info=image_info,
            raw_predictions=raw_predictions,
            threshold_values=threshold_values,
            base_conf=args.base_conf,
            max_det=args.max_det,
            rounds=args.rounds,
            min_map50=args.min_map50,
            selection_metric=args.selection_metric,
        )
        filtered = filter_predictions(raw_predictions, thresholds)

        result = {
            "nms_iou": nms_iou,
            "selection_metric": args.selection_metric,
            "min_mAP@0.5": min_map50,
            "thresholds": thresholds,
            "metrics": metrics,
            "history": history,
        }
        all_results.append(result)
        if args.save_intermediate:
            write_json(output_dir / f"thresholds_nms_{safe_tag(nms_iou)}.json", result)
            write_json(output_dir / f"predictions_nms_{safe_tag(nms_iou)}.json", filtered)

        if best_result is None or selected_metric_is_better(
            metrics,
            best_result["metrics"],
            args.selection_metric,
        ):
            best_result = result
            best_predictions = filtered

        print(
            f"nms={nms_iou:g} mAP@0.5={metrics['mAP@0.5']:.6f} "
            f"macro_f1={metrics['macro_f1']:.6f} micro_f1={metrics['micro_f1']:.6f} "
            f"min_f1={metrics['min_class_f1']:.6f} preds={metrics['num_predictions']} "
            f"thresholds={thresholds}"
        )

    assert best_result is not None and best_predictions is not None
    best_score = evaluate_predictions(
        module, ground_truth, classes, image_info, best_predictions, args.max_det
    )
    config = {
        "checkpoint": args.checkpoint,
        "source_predictions": args.predictions,
        "data_root": args.data_root,
        "split": args.split,
        "img_size": args.img_size,
        "base_conf_threshold": args.base_conf,
        "nms_iou": best_result["nms_iou"],
        "selection_metric": args.selection_metric,
        "min_mAP@0.5": best_result["min_mAP@0.5"],
        "max_det": args.max_det,
        "nms_max_det": args.nms_max_det,
        "thresholds": best_result["thresholds"],
        "metrics": best_score,
    }
    write_json(output_dir / "sweep_results.json", all_results)
    write_json(output_dir / "best_thresholds.json", config)
    write_json(output_dir / "best_predictions.json", best_predictions)
    write_json(output_dir / "best_score.json", best_score)

    print(
        f"best nms={best_result['nms_iou']:g} mAP@0.5={best_score['mAP@0.5']:.6f} "
        f"macro_f1={best_score['macro_f1']:.6f} micro_f1={best_score['micro_f1']:.6f} "
        f"min_f1={best_score['min_class_f1']:.6f} preds={best_score['num_predictions']}"
    )
    print(f"wrote {output_dir / 'best_thresholds.json'}")
    print(f"wrote {output_dir / 'best_predictions.json'}")


if __name__ == "__main__":
    main()
