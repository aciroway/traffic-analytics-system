#!/usr/bin/env python3
"""
Детекция, трекинг и подсчёт транспортных средств на видео.

Стек: Ultralytics YOLOv8 (yolov8m.pt) + Supervision (0.20+) + OpenCV.
"""

import argparse
import sys

import cv2
import numpy as np
import supervision as sv
import torch
from ultralytics import YOLO

# COCO-классы: car (2), motorcycle (3), bus (5), truck (7)
VEHICLE_CLASS_IDS = {2, 3, 5, 7}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Детекция и подсчёт автомобилей на видео (YOLOv26 + Supervision)."
    )
    parser.add_argument(
        "--source", type=str, default='BishTraffic.mp4',
        help="BishTraffic.mp4"
    )
    parser.add_argument(
        "--output", type=str, default="output.mp4",
        help="Путь для сохранения результирующего видео."
    )
    parser.add_argument(
        "--conf", type=float, default=0.3,
        help="Порог уверенности детекции (0-1)."
    )
    parser.add_argument(
        "--weights", type=str, default="yolo26m.pt",
        help="Путь/имя весов YOLOv26."
    )
    return parser.parse_args()


def resolve_device() -> str:
    """Выбор устройства: GPU (CUDA), если доступно, иначе CPU."""
    if torch.cuda.is_available():
        device = "cuda"
        print(f"[INFO] CUDA доступна. Используется GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("[INFO] CUDA не найдена. Используется CPU.")
    return device


def open_capture(source: str) -> cv2.VideoCapture:
    """Открытие видеопотока с проверкой корректности источника."""
    src = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[ERROR] Не удалось открыть источник видео: {source}", file=sys.stderr)
        sys.exit(1)
    return cap


def create_writer(output_path: str, fps: float, width: int, height: int) -> cv2.VideoWriter:
    """Инициализация VideoWriter с проверкой успешного открытия."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        print(f"[ERROR] Не удалось создать выходной файл: {output_path}", file=sys.stderr)
        sys.exit(1)
    return writer


def main() -> None:
    args = parse_args()
    device = resolve_device()
    use_half = (device == "cuda")  # Ускорение на GPU за счет FP16

    print(f"[INFO] Загрузка модели {args.weights}...")
    model = YOLO(args.weights)
    model.to(device)

    cap = open_capture(args.source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] Видео: {frame_width}x{frame_height} @ {fps:.2f} FPS, кадров: {total_frames}")

    writer = create_writer(args.output, fps, frame_width, frame_height)

    # Инициализация трекера ByteTrack
    tracker = sv.ByteTrack()

    # Инициализация аннотаторов
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)

    # Линия подсчёта (горизонтально по центру)
    line_start = sv.Point(0, frame_height // 2)
    line_end = sv.Point(frame_width, frame_height // 2)
    line_zone = sv.LineZone(start=line_start, end=line_end)
    line_zone_annotator = sv.LineZoneAnnotator(
        thickness=2, text_thickness=1, text_scale=0.7
    )

    class_names = model.model.names
    frame_index = 0

    print("[INFO] Начало обработки...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Инференс YOLOv8
        results = model(frame, conf=args.conf, device=device, half=use_half, verbose=False)[0]

        # Преобразование в формат Supervision
        detections = sv.Detections.from_ultralytics(results)

        # Фильтрация только тех классов, которые входят в VEHICLE_CLASS_IDS
        if len(detections) > 0 and detections.class_id is not None:
            vehicle_mask = np.isin(detections.class_id, list(VEHICLE_CLASS_IDS))
            detections = detections[vehicle_mask]

        # Обновление треков
        detections = tracker.update_with_detections(detections)

        # Подсчёт пересечений линии
        line_zone.trigger(detections)

        # Безопасное формирование меток (с защитой от None в tracker_id)
        labels = []
        if detections.tracker_id is not None:
            for class_id, tracker_id, confidence in zip(
                    detections.class_id, detections.tracker_id, detections.confidence
            ):
                t_id = f"#{tracker_id}" if tracker_id is not None else ""
                labels.append(f"{class_names[class_id]} {t_id} {confidence:.2f}")

        # Отрисовка
        annotated_frame = frame.copy()
        annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)

        if labels:
            annotated_frame = label_annotator.annotate(
                scene=annotated_frame, detections=detections, labels=labels
            )

        annotated_frame = line_zone_annotator.annotate(
            frame=annotated_frame, line_counter=line_zone
        )

        writer.write(annotated_frame)

        frame_index += 1
        if frame_index % 50 == 0:
            print(
                f"[INFO] Кадров: {frame_index}"
                f"{f'/{total_frames}' if total_frames > 0 else ''} | "
                f"IN: {line_zone.in_count} | OUT: {line_zone.out_count}"
            )

    cap.release()
    writer.release()

    print("[INFO] Обработка завершена.")
    print(f"[RESULT] Вход (IN): {line_zone.in_count} | Выход (OUT): {line_zone.out_count}")
    print(f"[RESULT] Сохранено в: {args.output}")


if __name__ == "__main__":
    main()