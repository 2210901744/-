from __future__ import annotations

import ctypes
import difflib
import json
import os
import random
import re
import tempfile
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext


APP_DIR = Path(__file__).resolve().parent
CLICK_CONFIG_PATH = APP_DIR / "click_config.json"
PADDLE_CACHE_DIR = APP_DIR / "paddle_cache"
APP_TITLE = "Learning Screen Assistant"
DEFAULT_FOREGROUND_WINDOW = "默认前置窗口"


DEFAULT_CLICK_CONFIG = {
    "enabled": False,
    "auto_click_after_query": False,
    "window_keyword": DEFAULT_FOREGROUND_WINDOW,
    "coords": {
        "A": [0, 0],
        "B": [0, 0],
        "C": [0, 0],
        "D": [0, 0],
        "E": [0, 0],
        "F": [0, 0],
        "TRUE": [0, 0],
        "FALSE": [0, 0],
    },
    "click_delay_seconds": 0.35,
    "double_check_click_position": True,
    "coordinate_check_max_delta_pixels": 35,
    "coordinate_check_retry_delay_seconds": 0.08,
    "use_ocr_answer_positions": True,
    "click_submit_after_answer": True,
    "submit_click_delay_seconds": 0.3,
    "submit_button_texts": ["保存", "提交", "确定", "确认", "完成"],
    "post_submit_page_wait_seconds": 1.5,
    "click_next_after_answer": True,
    "click_next_after_submit": True,
    "next_click_delay_seconds": 0.4,
    "next_button_texts": ["下一题", "下题", "下一页", "下一个", "下一步", "继续"],
    "post_click_page_wait_seconds": 0.6,
    "prev_button_texts": ["上一题", "上题", "上一页", "上一个", "返回"],
    "restore_mouse_after_click": True,
    "mouse_restore_position": [0, 0],
    "capture_target_window_only": True,
    "question_region": {
        "enabled": False,
        "rect": [0, 0, 0, 0],
    },
    "answers_inside_question_region": False,
    "position_memory": {
        "schema_version": 2,
        "enabled": True,
        "search_radius": 180,
        "options": {},
        "buttons": {},
    },
    "loop_interval_seconds": 1.2,
    "loop_max_rounds": 50,
    "stop_on_repeated_question": True,
    "same_question_grace_rounds": 2,
    "random_fallback_enabled": True,
    "random_fallback_options": ["A", "B", "C", "D"],
    "random_fallback_use_option_bounds": True,
    "multi_random_extra_when_single": True,
    "multi_min_selected_answers": 2,
}


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def normalize_text(text: str) -> str:
    text = text or ""
    text = text.lower()
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[，。？！、；：,.?!;:\\|/\-—_（）()\[\]【】{}<>《》“”\"'~]", "", text)
    return text


def split_question_and_options(raw_text: str) -> dict[str, str]:
    text = raw_text.strip()
    result = {"question": text, "A": "", "B": "", "C": "", "D": "", "E": "", "F": ""}

    pattern = re.compile(
        r"(?:^|\n|\r|\s)(?:[□☐☑☒○●〇◯◎◉◇◆口回oO0]\s*)?([ABCDEFabcdef])\s*[\.．、:：\)]\s*",
        re.MULTILINE,
    )
    matches = list(pattern.finditer(text))
    if len(matches) >= 2:
        first = matches[0]
        result["question"] = text[: first.start()].strip()
    for idx, match in enumerate(matches):
        label = match.group(1).upper()
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        option_text = text[start:end].strip()
        option_lines = [
            line.strip()
            for line in option_text.splitlines()
            if line.strip() and not re.fullmatch(r"[?？。．.,，、:：;；\-_—]+", line.strip())
        ]
        result[label] = "\n".join(option_lines).strip()
    return result


def normalize_question_type(value: str) -> str:
    text = normalize_text(value)
    if text in {"multi", "multiple", "multiplechoice", "多选", "多项选择", "多选题", "不定项"}:
        return "multi"
    if text in {"judge", "judgement", "judgment", "truefalse", "tf", "判断", "判断题", "真假"}:
        return "judge"
    return "single"


def infer_question_type(raw_text: str, answer: str = "") -> str:
    text = normalize_text(raw_text)
    ans = normalize_text(answer).upper()
    if any(word in text for word in ("多选", "多项选择", "不定项", "至少两个", "全部正确", "哪些")):
        return "multi"
    if any(word in text for word in ("判断题", "判断正误", "正确错误", "是否正确", "对还是错", "对错")):
        return "judge"
    labels = re.findall(r"[ABCDEF]", ans)
    if len(set(labels)) > 1:
        return "multi"
    return "single"


@dataclass
class AnswerResult:
    source: str
    question: str
    answer: str
    confidence: float
    explanation: str
    question_type: str = "single"
    matched_question: str = ""


@dataclass
class OCRTextItem:
    text: str
    x: int
    y: int
    left: Optional[int] = None
    top: Optional[int] = None
    right: Optional[int] = None
    bottom: Optional[int] = None


class DeepSeekClient:
    def __init__(self) -> None:
        self.api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
        self.model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat").strip()
        self.base_url = os.environ.get(
            "DEEPSEEK_BASE_URL", "https://api.deepseek.com/chat/completions"
        ).strip()

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def ask(self, raw_text: str) -> AnswerResult:
        if not self.api_key:
            raise RuntimeError("未配置 DEEPSEEK_API_KEY。请复制 .env.example 为 .env 并填写 key。")

        parts = split_question_and_options(raw_text)
        prompt = f"""
你是学习辅助答题助手。请根据题目和选项判断题型与最可能答案。
只返回 JSON，不要返回 Markdown，不要输出多余文本。

JSON 格式：
{{
  "question_type": "single 或 multi 或 judge",
  "answer": "单选返回 A/B/C/D/E/F；多选返回按字母排序的组合如 AC；判断题返回 T 或 F",
  "confidence": 0.0,
  "explanation": "简短解释"
}}

题目：
{parts["question"]}

选项：
A. {parts["A"]}
B. {parts["B"]}
C. {parts["C"]}
D. {parts["D"]}
E. {parts["E"]}
F. {parts["F"]}

题型规则：
- 单选题 question_type=single，answer 只能是一个选项字母 A-F；
- 多选题 question_type=multi，answer 返回多个选项字母，例如 AB、ACD；
- 判断题 question_type=judge，正确/对/是 返回 T，错误/错/否 返回 F。

如果选项为空，请根据题干直接作答；如果无法判断，请降低 confidence。
""".strip()

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "你是严谨的学习辅导助手，优先给出可复核的解释。",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            self.base_url,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"DeepSeek HTTP {e.code}: {detail}") from e

        content = data["choices"][0]["message"]["content"].strip()
        parsed = self._parse_json_content(content)
        return AnswerResult(
            source="deepseek",
            question=raw_text,
            answer=str(parsed.get("answer", "")).strip(),
            confidence=float(parsed.get("confidence", 0.0) or 0.0),
            explanation=str(parsed.get("explanation", "")).strip(),
            question_type=normalize_question_type(
                str(parsed.get("question_type", "")).strip()
            ),
        )

    @staticmethod
    def _parse_json_content(content: str) -> dict[str, Any]:
        content = content.strip()
        fence = chr(96) * 3
        if content.startswith(fence):
            content = re.sub(r"^" + re.escape(fence) + r"(?:json)?\s*", "", content)
            content = re.sub(r"\s*" + re.escape(fence) + r"$", "", content)
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", content, flags=re.S)
            if match:
                return json.loads(match.group(0))
            raise RuntimeError(f"DeepSeek 返回不是可解析 JSON：{content[:500]}")


class OCREngine:
    def __init__(self) -> None:
        self._engine = None

    @staticmethod
    def _prepare_paddle_runtime_env() -> None:
        PADDLE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        os.environ["PADDLE_PDX_CACHE_HOME"] = str(PADDLE_CACHE_DIR)
        os.environ["PADDLE_PDX_MODEL_SOURCE"] = "bos"
        os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
        os.environ.setdefault("PADDLE_HOME", str(PADDLE_CACHE_DIR / "paddle_home"))
        os.environ.setdefault("PADDLEOCR_HOME", str(PADDLE_CACHE_DIR / "paddleocr_home"))
        os.environ.setdefault("MODELSCOPE_CACHE", str(PADDLE_CACHE_DIR / "modelscope"))
        os.environ.setdefault(
            "MODELSCOPE_CREDENTIALS_PATH",
            str(PADDLE_CACHE_DIR / "modelscope" / "credentials"),
        )
        os.environ.setdefault("HF_HOME", str(PADDLE_CACHE_DIR / "huggingface"))
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(PADDLE_CACHE_DIR / "huggingface" / "hub"))
        os.environ.setdefault("AISTUDIO_CACHE_HOME", str(PADDLE_CACHE_DIR / "aistudio"))

        # PaddleOCR 3.x + PaddlePaddle 3.x 在部分 Windows/CPU 环境会在
        # PIR/oneDNN 执行路径报 ConvertPirAttribute2RuntimeAttribute 错误。
        # 这些变量必须在 import paddle / paddleocr 之前设置。
        os.environ["FLAGS_enable_pir_api"] = "0"
        os.environ["FLAGS_use_mkldnn"] = "0"
        os.environ["FLAGS_enable_onednn"] = "0"
        os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"
        os.environ["PADDLE_PDX_DISABLE_MKLDNN_MODEL_BL"] = "True"
        os.environ["PADDLE_PDX_USE_PIR_TRT"] = "False"
        os.environ.setdefault("PADDLE_PDX_CPU_NUM_THREADS", "4")

    def _load(self):
        if self._engine is None:
            self._prepare_paddle_runtime_env()
            from paddleocr import PaddleOCR

            init_variants = [
                {
                    "lang": "ch",
                    "ocr_version": "PP-OCRv4",
                    "use_doc_orientation_classify": False,
                    "use_doc_unwarping": False,
                    "use_textline_orientation": False,
                    "text_det_limit_side_len": 960,
                },
                {
                    "lang": "ch",
                    "ocr_version": "PP-OCRv5",
                    "use_doc_orientation_classify": False,
                    "use_doc_unwarping": False,
                    "use_textline_orientation": False,
                    "text_det_limit_side_len": 960,
                },
                {
                    "lang": "ch",
                    "ocr_version": "PP-OCRv3",
                    "use_doc_orientation_classify": False,
                    "use_doc_unwarping": False,
                    "use_textline_orientation": False,
                    "text_det_limit_side_len": 960,
                },
            ]
            last_error: Optional[Exception] = None
            for kwargs in init_variants:
                try:
                    self._engine = PaddleOCR(**kwargs)
                    break
                except Exception as e:
                    last_error = e
            if self._engine is None:
                raise RuntimeError(f"初始化 PaddleOCR 失败：{last_error}") from last_error
        return self._engine

    def screenshot_to_text(self) -> str:
        image_path, _left, _top = self._capture_screen()
        return self.image_to_text(image_path)

    def screenshot_to_items(self) -> list[OCRTextItem]:
        image_path, left, top = self._capture_screen()
        return self.image_to_items(image_path, offset_x=left, offset_y=top)

    def screenshot_region_to_items(
        self,
        left: int,
        top: int,
        right: int,
        bottom: int,
    ) -> list[OCRTextItem]:
        image_path, capture_left, capture_top = self._capture_region(left, top, right, bottom)
        return self.image_to_items(image_path, offset_x=capture_left, offset_y=capture_top)

    @staticmethod
    def items_to_text(items: list[OCRTextItem]) -> str:
        return "\n".join(item.text for item in items if item.text).strip()

    def image_to_text(self, image_path: Path) -> str:
        items = self.image_to_items(image_path)
        if items:
            return "\n".join(item.text for item in items).strip()

        engine = self._load()
        try:
            result = engine.ocr(str(image_path), cls=True)
        except TypeError:
            result = engine.ocr(str(image_path))
        lines: list[str] = []
        self._collect_ocr_text(result, lines)
        return "\n".join(lines).strip()

    def image_to_items(
        self,
        image_path: Path,
        offset_x: int = 0,
        offset_y: int = 0,
    ) -> list[OCRTextItem]:
        engine = self._load()
        try:
            result = engine.ocr(str(image_path), cls=True)
        except TypeError:
            result = engine.ocr(str(image_path))
        items: list[OCRTextItem] = []
        self._collect_ocr_items(result, items, offset_x=offset_x, offset_y=offset_y)
        return items

    @classmethod
    def _collect_ocr_text(cls, value: Any, lines: list[str]) -> None:
        if value is None:
            return
        if isinstance(value, dict):
            for key in ("rec_texts", "texts"):
                texts = value.get(key)
                if isinstance(texts, list):
                    for text in texts:
                        if text:
                            lines.append(str(text))
                    return
            for item in value.values():
                cls._collect_ocr_text(item, lines)
            return
        if isinstance(value, str):
            if value.strip():
                lines.append(value.strip())
            return
        if isinstance(value, (list, tuple)):
            if (
                len(value) >= 2
                and isinstance(value[1], (list, tuple))
                and value[1]
                and isinstance(value[1][0], str)
            ):
                text = value[1][0].strip()
                if text:
                    lines.append(text)
                return
            for item in value:
                cls._collect_ocr_text(item, lines)

    @classmethod
    def _collect_ocr_items(
        cls,
        value: Any,
        items: list[OCRTextItem],
        offset_x: int = 0,
        offset_y: int = 0,
    ) -> None:
        if value is None:
            return
        if isinstance(value, dict):
            texts = value.get("rec_texts")
            if texts is None:
                texts = value.get("texts")
            boxes = None
            for key in ("rec_boxes", "rec_polys", "dt_polys", "boxes"):
                candidate = value.get(key)
                if candidate is not None:
                    boxes = candidate
                    break
            if hasattr(texts, "tolist"):
                texts = texts.tolist()
            if isinstance(texts, (list, tuple)) and boxes is not None:
                for text, box in zip(texts, boxes):
                    center = cls._box_center(box)
                    bounds = cls._box_bounds(box)
                    if text and center:
                        left = top = right = bottom = None
                        if bounds is not None:
                            left = int(bounds[0] + offset_x)
                            top = int(bounds[1] + offset_y)
                            right = int(bounds[2] + offset_x)
                            bottom = int(bounds[3] + offset_y)
                        items.append(
                            OCRTextItem(
                                text=str(text).strip(),
                                x=int(center[0] + offset_x),
                                y=int(center[1] + offset_y),
                                left=left,
                                top=top,
                                right=right,
                                bottom=bottom,
                            )
                        )
                return
            for item in value.values():
                cls._collect_ocr_items(item, items, offset_x=offset_x, offset_y=offset_y)
            return
        if isinstance(value, (list, tuple)):
            if (
                len(value) >= 2
                and isinstance(value[1], (list, tuple))
                and value[1]
                and isinstance(value[1][0], str)
            ):
                center = cls._box_center(value[0])
                bounds = cls._box_bounds(value[0])
                text = value[1][0].strip()
                if text and center:
                    left = top = right = bottom = None
                    if bounds is not None:
                        left = int(bounds[0] + offset_x)
                        top = int(bounds[1] + offset_y)
                        right = int(bounds[2] + offset_x)
                        bottom = int(bounds[3] + offset_y)
                    items.append(
                        OCRTextItem(
                            text=text,
                            x=int(center[0] + offset_x),
                            y=int(center[1] + offset_y),
                            left=left,
                            top=top,
                            right=right,
                            bottom=bottom,
                        )
                    )
                return
            for item in value:
                cls._collect_ocr_items(item, items, offset_x=offset_x, offset_y=offset_y)

    @staticmethod
    def _box_bounds(box: Any) -> Optional[tuple[float, float, float, float]]:
        try:
            if hasattr(box, "tolist"):
                box = box.tolist()
            if isinstance(box, (list, tuple)) and len(box) == 4 and all(
                isinstance(v, (int, float)) for v in box
            ):
                x1, y1, x2, y2 = [float(v) for v in box]
                left, right = sorted((x1, x2))
                top, bottom = sorted((y1, y2))
                return left, top, right, bottom
            points: list[tuple[float, float]] = []
            for point in box:
                if hasattr(point, "tolist"):
                    point = point.tolist()
                if isinstance(point, (list, tuple)) and len(point) >= 2:
                    points.append((float(point[0]), float(point[1])))
            if points:
                xs = [point[0] for point in points]
                ys = [point[1] for point in points]
                return min(xs), min(ys), max(xs), max(ys)
        except Exception:
            return None
        return None

    @staticmethod
    def _box_center(box: Any) -> Optional[tuple[float, float]]:
        try:
            if hasattr(box, "tolist"):
                box = box.tolist()
            if isinstance(box, (list, tuple)) and len(box) == 4 and all(
                isinstance(v, (int, float)) for v in box
            ):
                x1, y1, x2, y2 = [float(v) for v in box]
                return (x1 + x2) / 2, (y1 + y2) / 2
            points: list[tuple[float, float]] = []
            for point in box:
                if hasattr(point, "tolist"):
                    point = point.tolist()
                if isinstance(point, (list, tuple)) and len(point) >= 2:
                    points.append((float(point[0]), float(point[1])))
            if points:
                return (
                    sum(point[0] for point in points) / len(points),
                    sum(point[1] for point in points) / len(points),
                )
        except Exception:
            return None
        return None

    @staticmethod
    def _capture_screen() -> tuple[Path, int, int]:
        left = 0
        top = 0
        try:
            import mss
            from PIL import Image

            with mss.mss() as sct:
                monitor = sct.monitors[1]
                left = int(monitor.get("left", 0))
                top = int(monitor.get("top", 0))
                shot = sct.grab(monitor)
                img = Image.frombytes("RGB", shot.size, shot.rgb)
        except Exception:
            from PIL import ImageGrab

            img = ImageGrab.grab()

        path = Path(tempfile.gettempdir()) / "learning_screen_assistant_screen.png"
        img.save(path)
        return path, left, top

    @staticmethod
    def _capture_region(left: int, top: int, right: int, bottom: int) -> tuple[Path, int, int]:
        left = max(0, int(left))
        top = max(0, int(top))
        right = max(left + 1, int(right))
        bottom = max(top + 1, int(bottom))

        try:
            import mss
            from PIL import Image

            with mss.mss() as sct:
                shot = sct.grab(
                    {
                        "left": left,
                        "top": top,
                        "width": right - left,
                        "height": bottom - top,
                    }
                )
                img = Image.frombytes("RGB", shot.size, shot.rgb)
        except Exception:
            from PIL import ImageGrab

            img = ImageGrab.grab(bbox=(left, top, right, bottom))

        path = Path(tempfile.gettempdir()) / "learning_screen_assistant_region.png"
        img.save(path)
        return path, left, top


class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


class RECT(ctypes.Structure):
    _fields_ = [
        ("left", ctypes.c_long),
        ("top", ctypes.c_long),
        ("right", ctypes.c_long),
        ("bottom", ctypes.c_long),
    ]


class WindowClicker:
    MOUSEEVENTF_LEFTDOWN = 0x0002
    MOUSEEVENTF_LEFTUP = 0x0004
    SW_RESTORE = 9

    @staticmethod
    def _user32():
        if os.name != "nt":
            raise RuntimeError("自动点击功能目前仅支持 Windows。")
        return ctypes.windll.user32

    @classmethod
    def enable_dpi_awareness(cls) -> None:
        if os.name != "nt":
            return
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            try:
                ctypes.windll.shcore.SetProcessDpiAwareness(1)
            except Exception:
                try:
                    cls._user32().SetProcessDPIAware()
                except Exception:
                    pass

    @classmethod
    def get_cursor_position(cls) -> tuple[int, int]:
        user32 = cls._user32()
        point = POINT()
        if not user32.GetCursorPos(ctypes.byref(point)):
            raise RuntimeError("读取鼠标位置失败。")
        return int(point.x), int(point.y)

    @classmethod
    def move_mouse(cls, x: int, y: int) -> None:
        """Move cursor without clicking. Used after auto click to avoid covering OCR text."""
        user32 = cls._user32()
        if not user32.SetCursorPos(int(x), int(y)):
            raise RuntimeError("复位鼠标位置失败。")

    @staticmethod
    def uses_foreground_window(keyword: str) -> bool:
        keyword = (keyword or "").strip()
        return keyword in {
            "",
            DEFAULT_FOREGROUND_WINDOW,
            "前置窗口",
            "当前前置窗口",
            "foreground",
            "active",
        }

    @staticmethod
    def is_assistant_window_title(title: str) -> bool:
        normalized = (title or "").strip().lower()
        return bool(normalized) and normalized == APP_TITLE.lower()

    @classmethod
    def get_foreground_window(cls) -> tuple[int, str]:
        user32 = cls._user32()
        hwnd = user32.GetForegroundWindow()
        if not hwnd:
            raise RuntimeError("未能获取当前前置窗口。")

        length = user32.GetWindowTextLengthW(hwnd)
        title = DEFAULT_FOREGROUND_WINDOW
        if length > 0:
            buffer = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buffer, length + 1)
            title = buffer.value.strip() or DEFAULT_FOREGROUND_WINDOW
        return int(hwnd), title

    @classmethod
    def find_window_by_title(cls, keyword: str, exclude_title: str = APP_TITLE) -> tuple[int, str]:
        keyword = (keyword or "").strip()
        if not keyword:
            return cls.get_foreground_window()

        user32 = cls._user32()
        matches: list[tuple[int, str]] = []
        lowered_keyword = keyword.lower()
        lowered_exclude = (exclude_title or "").lower()

        enum_proc_type = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)

        @enum_proc_type
        def enum_proc(hwnd, _lparam):
            if not user32.IsWindowVisible(hwnd):
                return True
            length = user32.GetWindowTextLengthW(hwnd)
            if length <= 0:
                return True
            buffer = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buffer, length + 1)
            title = buffer.value.strip()
            if not title:
                return True
            lowered_title = title.lower()
            if lowered_keyword in lowered_title and lowered_title != lowered_exclude:
                matches.append((int(hwnd), title))
            return True

        user32.EnumWindows(enum_proc, 0)
        if not matches:
            raise RuntimeError(f"未找到标题包含“{keyword}”的可见窗口。")
        return matches[0]

    @classmethod
    def activate_target_window(cls, window_keyword: str, delay_seconds: float = 0.15) -> str:
        user32 = cls._user32()
        if cls.uses_foreground_window(window_keyword):
            _hwnd, title = cls.get_foreground_window()
            return title

        hwnd, title = cls.find_window_by_title(window_keyword)
        user32.ShowWindow(hwnd, cls.SW_RESTORE)
        user32.SetForegroundWindow(hwnd)
        time.sleep(max(0.0, float(delay_seconds)))
        return title

    @classmethod
    def get_target_window_rect(cls, window_keyword: str) -> Optional[tuple[int, int, int, int]]:
        user32 = cls._user32()
        if cls.uses_foreground_window(window_keyword):
            hwnd, _title = cls.get_foreground_window()
        else:
            hwnd, _title = cls.find_window_by_title(window_keyword)

        rect = RECT()
        if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
            return None
        width = int(rect.right - rect.left)
        height = int(rect.bottom - rect.top)
        if width <= 50 or height <= 50:
            return None
        return int(rect.left), int(rect.top), int(rect.right), int(rect.bottom)

    @classmethod
    def click(cls, x: int, y: int, window_keyword: str, delay_seconds: float = 0.35) -> str:
        if x <= 0 or y <= 0:
            raise RuntimeError("点击坐标无效。")

        user32 = cls._user32()
        if cls.uses_foreground_window(window_keyword):
            _hwnd, title = cls.get_foreground_window()
            if cls.is_assistant_window_title(title):
                raise RuntimeError("当前前置窗口是助手窗口，已阻止点击；请先切换到目标练习窗口。")
        else:
            title = cls.activate_target_window(window_keyword, delay_seconds=0.0)
        time.sleep(max(0.0, float(delay_seconds)))

        if not user32.SetCursorPos(int(x), int(y)):
            raise RuntimeError("移动鼠标失败。")
        # 这里不能因为 GetCursorPos 与目标点存在轻微差异就中断点击。
        # 在 Windows 缩放、多显示器、远程桌面或浏览器页面中，读回位置可能出现
        # 数像素级偏差；上一版在这里直接抛异常，会造成“点击执行失败”。
        try:
            cls.get_cursor_position()
        except Exception:
            pass
        time.sleep(0.05)
        user32.mouse_event(cls.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
        time.sleep(0.05)
        user32.mouse_event(cls.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
        return title


class AssistantApp:
    def __init__(self, root: tk.Tk) -> None:
        load_dotenv(APP_DIR / ".env")
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("920x800")
        WindowClicker.enable_dpi_awareness()

        # 本版本禁用本地题库：所有题目均直接请求 DeepSeek。
        self.deepseek = DeepSeekClient()
        self.ocr = OCREngine()
        self.last_result: Optional[AnswerResult] = None
        self.click_config = self._load_click_config()
        self.coord_vars: dict[str, tk.StringVar] = {}
        self.question_region_var = tk.StringVar()
        self._last_assistant_rect: Optional[tuple[int, int, int, int]] = None
        self.loop_stop_event = threading.Event()
        self.loop_running = False

        self._build_ui()

    def _build_ui(self) -> None:
        top = tk.Frame(self.root)
        top.pack(fill=tk.X, padx=10, pady=8)

        tk.Button(top, text="0 识别当前屏幕", command=self.capture_ocr_async).pack(side=tk.LEFT, padx=4)
        tk.Button(top, text="1 请求 DeepSeek", command=self.query_async).pack(side=tk.LEFT, padx=4)
        tk.Button(top, text="3 点击当前答案", command=lambda: self.auto_click_current_answer(manual=True)).pack(
            side=tk.LEFT, padx=4
        )
        tk.Button(top, text="4 开始答题", command=self.start_loop_async).pack(side=tk.LEFT, padx=4)
        tk.Button(top, text="5 停止循环", command=self.stop_loop).pack(side=tk.LEFT, padx=4)
        self._bind_shortcuts()

        status = "DeepSeek 已配置，所有题目将直接请求 DeepSeek" if self.deepseek.is_configured() else "DeepSeek 未配置：请先在 .env 中填写 DEEPSEEK_API_KEY"
        self.status_var = tk.StringVar(value=status)
        tk.Label(self.root, textvariable=self.status_var, anchor="w").pack(fill=tk.X, padx=12)

        self._build_click_ui()

        tk.Label(self.root, text="题目/选项文本（可由 OCR 自动填入，也可手动粘贴）：", anchor="w").pack(
            fill=tk.X, padx=12, pady=(8, 2)
        )
        self.input_text = scrolledtext.ScrolledText(self.root, height=14, wrap=tk.WORD)
        self.input_text.pack(fill=tk.BOTH, expand=True, padx=12, pady=4)

        tk.Label(self.root, text="答案：", anchor="w").pack(fill=tk.X, padx=12, pady=(8, 2))
        self.answer_text = scrolledtext.ScrolledText(self.root, height=12, wrap=tk.WORD)
        self.answer_text.pack(fill=tk.BOTH, expand=True, padx=12, pady=4)

    def _bind_shortcuts(self) -> None:
        shortcuts = {
            "0": self.capture_ocr_async,
            "1": self.query_async,
            "3": lambda: self.auto_click_current_answer(manual=True),
            "4": self.start_loop_async,
            "5": self.stop_loop,
        }

        def handler(event: tk.Event):
            widget = self.root.focus_get()
            widget_class = ""
            try:
                widget_class = widget.winfo_class() if widget is not None else ""
            except Exception:
                widget_class = ""
            if widget_class in {"Entry", "Text", "TEntry", "Spinbox"}:
                return None
            command = shortcuts.get(str(event.char))
            if command is None:
                return None
            command()
            return "break"

        for key in shortcuts:
            self.root.bind_all(f"<KeyPress-{key}>", handler)

    def _build_click_ui(self) -> None:
        frame = tk.LabelFrame(
            self.root,
            text="练习模式自动点击（默认关闭；仅用于你自己控制的练习窗口）",
        )
        frame.pack(fill=tk.X, padx=12, pady=(8, 2))

        self.click_enabled_var = tk.BooleanVar(value=bool(self.click_config.get("enabled")))
        self.auto_click_var = tk.BooleanVar(
            value=bool(self.click_config.get("auto_click_after_query"))
        )
        self.click_submit_var = tk.BooleanVar(
            value=bool(self.click_config.get("click_submit_after_answer", True))
        )
        self.click_next_var = tk.BooleanVar(
            value=bool(self.click_config.get("click_next_after_answer", True))
        )
        self.click_next_after_submit_var = tk.BooleanVar(
            value=bool(self.click_config.get("click_next_after_submit", True))
        )
        self.restore_mouse_var = tk.BooleanVar(
            value=bool(self.click_config.get("restore_mouse_after_click", True))
        )
        self.double_check_position_var = tk.BooleanVar(
            value=bool(self.click_config.get("double_check_click_position", True))
        )
        self.answers_inside_region_var = tk.BooleanVar(
            value=bool(self.click_config.get("answers_inside_question_region", False))
        )
        self.window_keyword_var = tk.StringVar(
            value=str(self.click_config.get("window_keyword") or DEFAULT_FOREGROUND_WINDOW)
        )

        tk.Checkbutton(frame, text="启用自动点击", variable=self.click_enabled_var).grid(
            row=0, column=0, sticky="w", padx=6, pady=3
        )
        tk.Checkbutton(frame, text="查询后自动点击", variable=self.auto_click_var).grid(
            row=0, column=1, sticky="w", padx=6, pady=3
        )
        tk.Checkbutton(frame, text="多选题点击保存", variable=self.click_submit_var).grid(
            row=0, column=2, sticky="w", padx=6, pady=3
        )
        tk.Checkbutton(frame, text="自动点击下一题", variable=self.click_next_var).grid(
            row=0, column=3, sticky="w", padx=6, pady=3
        )
        tk.Checkbutton(frame, text="保存后仍点下一题", variable=self.click_next_after_submit_var).grid(
            row=0, column=4, sticky="w", padx=6, pady=3
        )
        tk.Checkbutton(frame, text="点击后鼠标复位", variable=self.restore_mouse_var).grid(
            row=0, column=5, sticky="w", padx=6, pady=3
        )
        tk.Checkbutton(frame, text="点击前双重校验", variable=self.double_check_position_var).grid(
            row=0, column=6, sticky="w", padx=6, pady=3
        )
        tk.Checkbutton(
            frame,
            text="答案/选项在题目框选内",
            variable=self.answers_inside_region_var,
            command=self._refresh_question_region_var,
        ).grid(row=0, column=7, sticky="w", padx=6, pady=3)
        tk.Label(frame, text="目标窗口：").grid(row=0, column=8, sticky="e", padx=6)
        tk.Entry(frame, textvariable=self.window_keyword_var, width=26).grid(
            row=0, column=9, sticky="we", padx=6
        )

        self.loop_interval_var = tk.StringVar(
            value=str(self.click_config.get("loop_interval_seconds", 1.2))
        )
        self.loop_max_rounds_var = tk.StringVar(
            value=str(self.click_config.get("loop_max_rounds", 50))
        )

        tk.Label(frame, text="循环间隔秒：").grid(row=1, column=0, sticky="e", padx=4)
        tk.Entry(frame, textvariable=self.loop_interval_var, width=8).grid(
            row=1, column=1, padx=4, sticky="w"
        )
        tk.Label(frame, text="最多轮数：").grid(row=1, column=2, sticky="e", padx=4)
        tk.Entry(frame, textvariable=self.loop_max_rounds_var, width=8).grid(
            row=1, column=3, padx=4, sticky="w"
        )

        coords = self.click_config.get("coords") or {}
        click_labels = ("A", "B", "C", "D", "E", "F", "TRUE", "FALSE")
        for idx, label in enumerate(click_labels):
            x, y = self._coord_pair(coords.get(label))
            var = tk.StringVar(value=f"{x},{y}")
            self.coord_vars[label] = var
            row_base = 2 + (idx // 4) * 2
            col_base = (idx % 3) * 2
            col_base = (idx % 4) * 2
            display = {"TRUE": "判断正确", "FALSE": "判断错误"}.get(label, label)
            tk.Label(frame, text=f"{display} 坐标：").grid(
                row=row_base, column=col_base, sticky="e", padx=4
            )
            tk.Entry(frame, textvariable=var, width=11).grid(
                row=row_base, column=col_base + 1, padx=4
            )
            tk.Button(
                frame,
                text=f"3 秒后记录 {display}",
                command=lambda item=label: self.capture_coord_later(item),
            ).grid(row=row_base + 1, column=col_base, columnspan=2, padx=4, pady=(2, 5))

        tk.Button(frame, text="保存点击配置", command=self.save_click_config).grid(
            row=0, column=8, padx=6
        )
        tk.Button(frame, text="框选题目区域", command=self.select_question_region).grid(
            row=1, column=4, padx=6, sticky="w"
        )
        tk.Button(frame, text="清除题目框选", command=self.clear_question_region).grid(
            row=1, column=5, padx=6, sticky="w"
        )
        self._refresh_question_region_var()
        tk.Label(frame, textvariable=self.question_region_var, fg="#555555", anchor="w").grid(
            row=6, column=0, columnspan=8, sticky="we", padx=6, pady=(0, 2)
        )
        tk.Label(
            frame,
            text="提示：A-F 是普通选项；判断正确/判断错误是判断题专用。若目标窗口为默认前置窗口，程序会避免点击助手窗口本身。",
            fg="#555555",
            anchor="w",
        ).grid(row=7, column=0, columnspan=8, sticky="we", padx=6, pady=(0, 4))

        frame.columnconfigure(8, weight=1)

    def _normalize_question_region(self, value: Any = None) -> dict[str, Any]:
        region = value if isinstance(value, dict) else self.click_config.get("question_region")
        enabled = bool((region or {}).get("enabled")) if isinstance(region, dict) else False
        rect_value = (region or {}).get("rect") if isinstance(region, dict) else None
        rect = [0, 0, 0, 0]
        if isinstance(rect_value, (list, tuple)) and len(rect_value) >= 4:
            try:
                left, top, right, bottom = [int(float(v)) for v in rect_value[:4]]
                left, right = sorted((left, right))
                top, bottom = sorted((top, bottom))
                if right - left >= 20 and bottom - top >= 20:
                    rect = [left, top, right, bottom]
                else:
                    enabled = False
            except Exception:
                enabled = False
        else:
            enabled = False
        return {"enabled": enabled, "rect": rect}

    def _question_region_rect(self) -> Optional[tuple[int, int, int, int]]:
        region = self._normalize_question_region()
        if not region["enabled"]:
            return None
        left, top, right, bottom = region["rect"]
        return int(left), int(top), int(right), int(bottom)

    def _answers_inside_question_region(self) -> bool:
        var = getattr(self, "answers_inside_region_var", None)
        if var is not None:
            try:
                return bool(var.get())
            except Exception:
                pass
        return bool(self.click_config.get("answers_inside_question_region", False))

    def _refresh_question_region_var(self) -> None:
        region = self._normalize_question_region()
        answer_scope = (
            "答案/选项：在框选内查找"
            if self._answers_inside_question_region()
            else "答案/选项：在框选外/下方查找"
        )
        if region["enabled"]:
            left, top, right, bottom = region["rect"]
            self.question_region_var.set(
                f"题目识别区域：已启用 ({left},{top}) - ({right},{bottom})，大小 {right-left}×{bottom-top}；{answer_scope}"
            )
        else:
            self.question_region_var.set(
                f"题目识别区域：未设置（循环时将按目标窗口/全屏识别）；{answer_scope}"
            )

    @staticmethod
    def _virtual_screen_rect() -> tuple[int, int, int, int]:
        try:
            import mss

            with mss.mss() as sct:
                monitor = sct.monitors[0]
                left = int(monitor.get("left", 0))
                top = int(monitor.get("top", 0))
                width = int(monitor.get("width", 0))
                height = int(monitor.get("height", 0))
                if width > 0 and height > 0:
                    return left, top, left + width, top + height
        except Exception:
            pass

        if os.name == "nt":
            try:
                user32 = ctypes.windll.user32
                left = int(user32.GetSystemMetrics(76))  # SM_XVIRTUALSCREEN
                top = int(user32.GetSystemMetrics(77))  # SM_YVIRTUALSCREEN
                width = int(user32.GetSystemMetrics(78))  # SM_CXVIRTUALSCREEN
                height = int(user32.GetSystemMetrics(79))  # SM_CYVIRTUALSCREEN
                if width > 0 and height > 0:
                    return left, top, left + width, top + height
            except Exception:
                pass
        return 0, 0, 1920, 1080

    def select_question_region(self) -> None:
        self.set_status("请拖拽框选题目文字区域；按 Esc 取消。")
        self.root.withdraw()
        self.root.after(250, self._show_question_region_selector)

    def _show_question_region_selector(self) -> None:
        screen_left, screen_top, screen_right, screen_bottom = self._virtual_screen_rect()
        screen_width = max(1, screen_right - screen_left)
        screen_height = max(1, screen_bottom - screen_top)

        overlay = tk.Toplevel(self.root)
        overlay.overrideredirect(True)
        overlay.attributes("-topmost", True)
        try:
            overlay.attributes("-alpha", 0.25)
        except Exception:
            pass
        overlay.configure(bg="black")
        overlay.geometry(f"{screen_width}x{screen_height}+{screen_left}+{screen_top}")

        canvas = tk.Canvas(overlay, bg="black", highlightthickness=0, cursor="crosshair")
        canvas.pack(fill=tk.BOTH, expand=True)
        canvas.create_text(
            24,
            24,
            anchor="nw",
            text="拖拽框选题目文字区域，松开鼠标保存；Esc 取消",
            fill="white",
            font=("Microsoft YaHei UI", 18, "bold"),
        )

        state: dict[str, Any] = {"start": None, "rect_id": None}

        def to_screen(event: tk.Event) -> tuple[int, int]:
            return int(screen_left + event.x), int(screen_top + event.y)

        def on_press(event: tk.Event) -> None:
            state["start"] = to_screen(event)
            if state["rect_id"] is not None:
                canvas.delete(state["rect_id"])
            state["rect_id"] = canvas.create_rectangle(
                event.x,
                event.y,
                event.x,
                event.y,
                outline="#00ff66",
                width=3,
            )

        def on_drag(event: tk.Event) -> None:
            if state["start"] is None or state["rect_id"] is None:
                return
            start_x, start_y = state["start"]
            canvas.coords(
                state["rect_id"],
                start_x - screen_left,
                start_y - screen_top,
                event.x,
                event.y,
            )

        def finish_with_rect(rect: Optional[list[int]]) -> None:
            overlay.destroy()
            self.root.deiconify()
            self.root.lift()
            if rect is None:
                self.set_status("已取消题目框选。")
                return
            self.click_config = self._read_click_config_from_ui()
            self.click_config["question_region"] = self._normalize_question_region(
                {"enabled": True, "rect": rect}
            )
            self._refresh_question_region_var()
            self.save_click_config()
            left, top, right, bottom = self.click_config["question_region"]["rect"]
            self.set_status(f"已设置题目识别区域：{left},{top},{right},{bottom}")

        def on_release(event: tk.Event) -> None:
            if state["start"] is None:
                finish_with_rect(None)
                return
            end_x, end_y = to_screen(event)
            start_x, start_y = state["start"]
            left, right = sorted((start_x, end_x))
            top, bottom = sorted((start_y, end_y))
            if right - left < 20 or bottom - top < 20:
                finish_with_rect(None)
                messagebox.showwarning("框选太小", "题目区域太小，请重新框选。")
                return
            finish_with_rect([left, top, right, bottom])

        def on_cancel(_event: Optional[tk.Event] = None) -> None:
            finish_with_rect(None)

        overlay.bind("<Escape>", on_cancel)
        canvas.bind("<ButtonPress-1>", on_press)
        canvas.bind("<B1-Motion>", on_drag)
        canvas.bind("<ButtonRelease-1>", on_release)
        overlay.focus_force()

    def clear_question_region(self) -> None:
        self.click_config = self._read_click_config_from_ui()
        self.click_config["question_region"] = self._normalize_question_region(
            {"enabled": False, "rect": [0, 0, 0, 0]}
        )
        self._refresh_question_region_var()
        self.save_click_config()
        self.set_status("已清除题目识别区域。")

    def set_status(self, text: str) -> None:
        self.status_var.set(text)
        self.root.update_idletasks()

    def _prepare_foreground_target_window(self, action: str = "点击") -> None:
        if not WindowClicker.uses_foreground_window(
            self.click_config.get("window_keyword", DEFAULT_FOREGROUND_WINDOW)
        ):
            return
        try:
            _hwnd, title = WindowClicker.get_foreground_window()
        except Exception:
            return
        if not WindowClicker.is_assistant_window_title(title):
            return

        self.set_status(f"当前前置窗口是助手窗口，正在最小化助手以避免误{action}……")
        self._last_assistant_rect = self._assistant_window_rect()
        try:
            self.root.iconify()
            self.root.update_idletasks()
        except Exception:
            pass
        time.sleep(0.6)
        _hwnd, title = WindowClicker.get_foreground_window()
        if WindowClicker.is_assistant_window_title(title):
            raise RuntimeError(f"当前前置窗口仍是助手窗口，已取消{action}；请切换到目标练习窗口。")

    def _assistant_window_rect(self) -> Optional[tuple[int, int, int, int]]:
        try:
            self.root.update_idletasks()
            left = int(self.root.winfo_rootx())
            top = int(self.root.winfo_rooty())
            width = int(self.root.winfo_width())
            height = int(self.root.winfo_height())
            if width <= 1 or height <= 1:
                return None
            return left, top, left + width, top + height
        except Exception:
            return None

    def _point_in_rect(
        self,
        x: int,
        y: int,
        rect: Optional[tuple[int, int, int, int]],
        margin: int = 8,
    ) -> bool:
        if rect is None:
            return False
        left, top, right, bottom = rect
        return left - margin <= x <= right + margin and top - margin <= y <= bottom + margin

    def _point_in_assistant_window(self, x: int, y: int) -> bool:
        return self._point_in_rect(x, y, self._assistant_window_rect()) or self._point_in_rect(
            x, y, self._last_assistant_rect
        )

    def _load_click_config(self) -> dict[str, Any]:
        config = json.loads(json.dumps(DEFAULT_CLICK_CONFIG))
        if CLICK_CONFIG_PATH.exists():
            try:
                loaded = json.loads(CLICK_CONFIG_PATH.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    config.update(loaded)
                    config["coords"] = {
                        **DEFAULT_CLICK_CONFIG["coords"],
                        **(loaded.get("coords") or {}),
                    }
                    loaded_coords = loaded.get("coords") or {}
                    if "TRUE" not in loaded_coords and "T" in loaded_coords:
                        config["coords"]["TRUE"] = loaded_coords.get("T")
                    if "FALSE" not in loaded_coords and "F" in loaded_coords:
                        config["coords"]["FALSE"] = loaded_coords.get("F")
                        config["coords"]["F"] = DEFAULT_CLICK_CONFIG["coords"]["F"]
                    elif "F" not in loaded_coords:
                        config["coords"]["F"] = DEFAULT_CLICK_CONFIG["coords"]["F"]
                    loaded_region = loaded.get("question_region")
                    if isinstance(loaded_region, dict):
                        config["question_region"] = {
                            **DEFAULT_CLICK_CONFIG["question_region"],
                            **loaded_region,
                        }
                    loaded_memory = loaded.get("position_memory")
                    if isinstance(loaded_memory, dict):
                        config["position_memory"] = {
                            **DEFAULT_CLICK_CONFIG["position_memory"],
                            **loaded_memory,
                        }
            except Exception:
                pass
        if not str(config.get("window_keyword") or "").strip():
            config["window_keyword"] = DEFAULT_FOREGROUND_WINDOW
        region = config.get("question_region")
        if not isinstance(region, dict):
            config["question_region"] = DEFAULT_CLICK_CONFIG["question_region"]
        config["answers_inside_question_region"] = bool(
            config.get(
                "answers_inside_question_region",
                DEFAULT_CLICK_CONFIG["answers_inside_question_region"],
            )
        )
        config["click_submit_after_answer"] = bool(
            config.get(
                "click_submit_after_answer",
                DEFAULT_CLICK_CONFIG["click_submit_after_answer"],
            )
        )
        config["restore_mouse_after_click"] = bool(
            config.get(
                "restore_mouse_after_click",
                DEFAULT_CLICK_CONFIG["restore_mouse_after_click"],
            )
        )
        memory = config.get("position_memory")
        if not isinstance(memory, dict):
            config["position_memory"] = DEFAULT_CLICK_CONFIG["position_memory"]
        config["position_memory"] = self._normalize_position_memory(config["position_memory"])
        return config

    @staticmethod
    def _coord_pair(value: Any) -> tuple[int, int]:
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            try:
                return int(value[0]), int(value[1])
            except Exception:
                return 0, 0
        if isinstance(value, str):
            numbers = re.findall(r"-?\d+", value)
            if len(numbers) >= 2:
                return int(numbers[0]), int(numbers[1])
        return 0, 0

    @staticmethod
    def _normalize_point(value: Any) -> Optional[list[int]]:
        x, y = AssistantApp._coord_pair(value)
        if x > 0 and y > 0:
            return [x, y]
        return None

    def _normalize_position_memory(self, value: Any = None) -> dict[str, Any]:
        raw = value if isinstance(value, dict) else self.click_config.get("position_memory")
        if not isinstance(raw, dict):
            raw = {}
        try:
            radius = int(float(raw.get("search_radius", 180)))
        except Exception:
            radius = 180
        radius = min(600, max(40, radius))

        options: dict[str, list[int]] = {}
        raw_options = raw.get("options") if isinstance(raw.get("options"), dict) else {}
        try:
            schema_version = int(raw.get("schema_version", 1))
        except Exception:
            schema_version = 1
        for label in ("A", "B", "C", "D", "E"):
            point = self._normalize_point(raw_options.get(label))
            if point is not None:
                options[label] = point
        point = self._normalize_point(raw_options.get("F"))
        if point is not None and schema_version >= 2:
            options["F"] = point
        point = self._normalize_point(raw_options.get("TRUE") or raw_options.get("T"))
        if point is not None:
            options["TRUE"] = point
        point = self._normalize_point(raw_options.get("FALSE") or raw_options.get("F"))
        if point is not None:
            options["FALSE"] = point

        buttons: dict[str, list[int]] = {}
        raw_buttons = raw.get("buttons") if isinstance(raw.get("buttons"), dict) else {}
        for key in ("next", "prev", "submit"):
            point = self._normalize_point(raw_buttons.get(key))
            if point is not None:
                buttons[key] = point

        return {
            "schema_version": 2,
            "enabled": bool(raw.get("enabled", True)),
            "search_radius": radius,
            "options": options,
            "buttons": buttons,
        }

    def _read_click_config_from_ui(self) -> dict[str, Any]:
        coords: dict[str, list[int]] = {}
        for label in ("A", "B", "C", "D", "E", "F", "TRUE", "FALSE"):
            x, y = self._coord_pair(self.coord_vars[label].get())
            coords[label] = [x, y]
            self.coord_vars[label].set(f"{x},{y}")
        try:
            loop_interval = max(0.2, float(self.loop_interval_var.get()))
        except Exception:
            loop_interval = DEFAULT_CLICK_CONFIG["loop_interval_seconds"]
            self.loop_interval_var.set(str(loop_interval))
        try:
            loop_max_rounds = max(1, int(float(self.loop_max_rounds_var.get())))
        except Exception:
            loop_max_rounds = DEFAULT_CLICK_CONFIG["loop_max_rounds"]
            self.loop_max_rounds_var.set(str(loop_max_rounds))

        return {
            "enabled": bool(self.click_enabled_var.get()),
            "auto_click_after_query": bool(self.auto_click_var.get()),
            "click_submit_after_answer": bool(self.click_submit_var.get()),
            "click_next_after_answer": bool(self.click_next_var.get()),
            "click_next_after_submit": bool(self.click_next_after_submit_var.get()),
            "restore_mouse_after_click": bool(self.restore_mouse_var.get()),
            "double_check_click_position": bool(self.double_check_position_var.get()),
            "use_ocr_answer_positions": bool(
                self.click_config.get("use_ocr_answer_positions", True)
            ),
            "answers_inside_question_region": bool(self.answers_inside_region_var.get()),
            "window_keyword": self.window_keyword_var.get().strip() or DEFAULT_FOREGROUND_WINDOW,
            "coords": coords,
            "click_delay_seconds": float(
                self.click_config.get(
                    "click_delay_seconds",
                    DEFAULT_CLICK_CONFIG["click_delay_seconds"],
                )
            ),
            "coordinate_check_max_delta_pixels": float(
                self.click_config.get(
                    "coordinate_check_max_delta_pixels",
                    DEFAULT_CLICK_CONFIG["coordinate_check_max_delta_pixels"],
                )
            ),
            "coordinate_check_retry_delay_seconds": float(
                self.click_config.get(
                    "coordinate_check_retry_delay_seconds",
                    DEFAULT_CLICK_CONFIG["coordinate_check_retry_delay_seconds"],
                )
            ),
            "submit_click_delay_seconds": float(
                self.click_config.get(
                    "submit_click_delay_seconds",
                    DEFAULT_CLICK_CONFIG["submit_click_delay_seconds"],
                )
            ),
            "submit_button_texts": self.click_config.get(
                "submit_button_texts",
                DEFAULT_CLICK_CONFIG["submit_button_texts"],
            ),
            "post_submit_page_wait_seconds": float(
                self.click_config.get(
                    "post_submit_page_wait_seconds",
                    DEFAULT_CLICK_CONFIG["post_submit_page_wait_seconds"],
                )
            ),
            "next_click_delay_seconds": float(
                self.click_config.get(
                    "next_click_delay_seconds",
                    DEFAULT_CLICK_CONFIG["next_click_delay_seconds"],
                )
            ),
            "next_button_texts": self.click_config.get(
                "next_button_texts",
                DEFAULT_CLICK_CONFIG["next_button_texts"],
            ),
            "post_click_page_wait_seconds": float(
                self.click_config.get(
                    "post_click_page_wait_seconds",
                    DEFAULT_CLICK_CONFIG["post_click_page_wait_seconds"],
                )
            ),
            "prev_button_texts": self.click_config.get(
                "prev_button_texts",
                DEFAULT_CLICK_CONFIG["prev_button_texts"],
            ),
            "capture_target_window_only": bool(
                self.click_config.get("capture_target_window_only", True)
            ),
            "mouse_restore_position": self.click_config.get(
                "mouse_restore_position",
                DEFAULT_CLICK_CONFIG["mouse_restore_position"],
            ),
            "question_region": self._normalize_question_region(),
            "position_memory": self._normalize_position_memory(),
            "loop_interval_seconds": loop_interval,
            "loop_max_rounds": loop_max_rounds,
            "stop_on_repeated_question": bool(
                self.click_config.get("stop_on_repeated_question", True)
            ),
            "same_question_grace_rounds": int(
                self.click_config.get(
                    "same_question_grace_rounds",
                    DEFAULT_CLICK_CONFIG["same_question_grace_rounds"],
                )
            ),
            "random_fallback_enabled": bool(
                self.click_config.get(
                    "random_fallback_enabled",
                    DEFAULT_CLICK_CONFIG["random_fallback_enabled"],
                )
            ),
            "random_fallback_options": list(
                self.click_config.get(
                    "random_fallback_options",
                    DEFAULT_CLICK_CONFIG["random_fallback_options"],
                )
            ),
            "random_fallback_use_option_bounds": bool(
                self.click_config.get(
                    "random_fallback_use_option_bounds",
                    DEFAULT_CLICK_CONFIG["random_fallback_use_option_bounds"],
                )
            ),
            "multi_random_extra_when_single": bool(
                self.click_config.get(
                    "multi_random_extra_when_single",
                    DEFAULT_CLICK_CONFIG["multi_random_extra_when_single"],
                )
            ),
            "multi_min_selected_answers": max(
                1,
                int(
                    self.click_config.get(
                        "multi_min_selected_answers",
                        DEFAULT_CLICK_CONFIG["multi_min_selected_answers"],
                    )
                ),
            ),
        }

    def _write_click_config_file(self) -> None:
        CLICK_CONFIG_PATH.write_text(
            json.dumps(self.click_config, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def save_click_config(self) -> None:
        self.click_config = self._read_click_config_from_ui()
        self._write_click_config_file()
        self.set_status(f"已保存点击配置：{CLICK_CONFIG_PATH}")

    def capture_coord_later(self, label: str) -> None:
        self.set_status(f"请在 3 秒内把鼠标移动到选项 {label} 的点击位置……")

        def record() -> None:
            try:
                x, y = WindowClicker.get_cursor_position()
                self.coord_vars[label].set(f"{x},{y}")
                self.save_click_config()
                self.set_status(f"已记录选项 {label} 坐标：{x},{y}")
            except Exception as e:
                self.set_status("记录坐标失败。")
                messagebox.showerror("记录坐标失败", str(e))

        self.root.after(3000, record)

    def _ocr_items_fast(self, force_fullscreen: bool = False) -> list[OCRTextItem]:
        if (not force_fullscreen) and self.click_config.get("capture_target_window_only", True):
            self._prepare_foreground_target_window("识别")
            try:
                WindowClicker.activate_target_window(
                    self.click_config.get("window_keyword", DEFAULT_FOREGROUND_WINDOW),
                    delay_seconds=0.05,
                )
                rect = WindowClicker.get_target_window_rect(
                    self.click_config.get("window_keyword", DEFAULT_FOREGROUND_WINDOW)
                )
                if rect is not None:
                    return self.ocr.screenshot_region_to_items(*rect)
            except Exception:
                pass
        return self.ocr.screenshot_to_items()

    def _ocr_text_fast(self, force_fullscreen: bool = False) -> str:
        return OCREngine.items_to_text(self._ocr_items_fast(force_fullscreen=force_fullscreen))

    def _ocr_question_items_fast(self) -> list[OCRTextItem]:
        rect = self._question_region_rect()
        if rect is not None:
            return self.ocr.screenshot_region_to_items(*rect)
        return self._ocr_items_fast()

    def _ocr_question_text_fast(self) -> str:
        return OCREngine.items_to_text(self._ocr_question_items_fast())

    def _position_memory_enabled(self) -> bool:
        return bool(self._normalize_position_memory().get("enabled", True))

    def _position_memory_radius(self) -> int:
        return int(self._normalize_position_memory().get("search_radius", 180))

    def _remembered_point(self, group: str, key: str) -> Optional[tuple[int, int]]:
        memory = self._normalize_position_memory()
        points = memory.get(group)
        if not isinstance(points, dict):
            return None
        point = self._normalize_point(points.get(key))
        if point is None:
            return None
        return int(point[0]), int(point[1])

    def _update_position_memory(self, group: str, key: str, item: OCRTextItem) -> None:
        if not self._position_memory_enabled():
            return
        if group not in {"options", "buttons"}:
            return
        memory = self._normalize_position_memory()
        points = memory.setdefault(group, {})
        if isinstance(points, dict):
            points[key] = [int(item.x), int(item.y)]
        self.click_config["position_memory"] = memory
        try:
            self._write_click_config_file()
        except Exception:
            pass

    def _items_near_point(
        self,
        items: list[OCRTextItem],
        point: Optional[tuple[int, int]],
        radius: Optional[int] = None,
    ) -> list[OCRTextItem]:
        if point is None:
            return list(items)
        px, py = point
        r = int(radius if radius is not None else self._position_memory_radius())
        return [
            item
            for item in items
            if abs(int(item.x) - px) <= r and abs(int(item.y) - py) <= r
        ]

    def _ocr_items_near_point(self, point: tuple[int, int]) -> list[OCRTextItem]:
        px, py = point
        radius = self._position_memory_radius()
        left = px - radius
        top = py - radius
        right = px + radius
        bottom = py + radius

        try:
            target_rect = WindowClicker.get_target_window_rect(
                self.click_config.get("window_keyword", DEFAULT_FOREGROUND_WINDOW)
            )
            if target_rect is not None:
                win_left, win_top, win_right, win_bottom = target_rect
                left = max(left, win_left)
                top = max(top, win_top)
                right = min(right, win_right)
                bottom = min(bottom, win_bottom)
        except Exception:
            pass

        if right - left < 20 or bottom - top < 20:
            left, top, right, bottom = px - radius, py - radius, px + radius, py + radius
        return self.ocr.screenshot_region_to_items(left, top, right, bottom)

    def _ocr_items_from_memory_area(self, group: str, key: str) -> list[OCRTextItem]:
        if not self._position_memory_enabled():
            return []
        point = self._remembered_point(group, key)
        if point is None:
            return []
        return self._ocr_items_near_point(point)

    @staticmethod
    def _extract_answer_label(answer: str) -> Optional[str]:
        labels = AssistantApp._extract_answer_labels("single", answer)
        return labels[0] if labels else None

    @staticmethod
    def _extract_answer_labels(question_type: str, answer: str) -> list[str]:
        qtype = normalize_question_type(question_type)
        text = (answer or "").strip().upper()
        if not text:
            return []

        def finalize_labels(raw: str) -> list[str]:
            compact = re.sub(r"[^ABCDEF]", "", raw.upper())
            labels: list[str] = []
            for label in compact:
                if label not in labels:
                    labels.append(label)
            if qtype == "single":
                return labels[:1]
            if qtype == "multi":
                order = {label: idx for idx, label in enumerate("ABCDEF")}
                return sorted(labels, key=lambda item: order.get(item, 99))
            return labels

        if qtype == "judge":
            normalized = normalize_text(text)
            if any(token in normalized for token in ("正确", "对", "是", "true", "t")):
                return ["TRUE"]
            if any(token in normalized for token in ("错误", "错", "否", "false", "f")):
                return ["FALSE"]
            if text[:1] in {"A", "B"}:
                return [text[:1]]

        explicit_patterns = [
            r"(?:答案|正确答案|参考答案|选项|选择|应选)\s*(?:为|是)?\s*[：:\s]*([ABCDEF](?:\s*[,，、/和与]?\s*[ABCDEF])*)",
            r"(?:^|[，,。．.；;\s])(?:为|是)\s*([ABCDEF](?:\s*[,，、/和与]?\s*[ABCDEF])*)",
            r"^\s*([ABCDEF](?:\s*[,，、/和与]?\s*[ABCDEF])*)\s*(?:$|[。．.、,，:：；;）)])",
        ]
        for pattern in explicit_patterns:
            match = re.search(pattern, text)
            if match:
                return finalize_labels(match.group(1))

        # 只有答案字段本身几乎全是选项字母和分隔符时，才做宽松提取。
        # 不能对整段解释做 re.sub 抽取，否则解释或选项正文中的 A-F 会污染点击目标。
        if re.fullmatch(r"[\sABCDEF,，、/和与]+", text):
            return finalize_labels(text)

        return []

    def _resolve_click_label(self, label: str) -> str:
        return label.upper()

    @staticmethod
    def _strip_leading_choice_marks(text: str) -> str:
        stripped = (text or "").strip()
        # 常见误识别：单选圆圈 + A 被 OCR 成 "OA、..." 或 "0A、..."。
        stripped = re.sub(
            r"^[oO0]\s*(?=[ABCDEFabcdef]\s*[.．、:：)）])",
            "",
            stripped,
        )
        # OCR 可能把单选/复选控件识别成这些符号或“口/回”等近似字形。
        return re.sub(r"^[\s□☐☑☒○●〇◯◎◉◇◆口回]+", "", stripped).strip()

    @staticmethod
    def _has_leading_choice_mark(text: str) -> bool:
        return bool(re.match(r"^\s*[□☐☑☒○●〇◯◎◉◇◆口回]", text or ""))

    @staticmethod
    def _option_text_without_label(text: str) -> str:
        stripped = AssistantApp._strip_leading_choice_marks(text)
        return re.sub(r"^[ABCDEFabcdef]\s*[.．、:：)）]】]?\s*", "", stripped).strip()

    @staticmethod
    def _is_option_label_text(text: str, label: str) -> bool:
        stripped = AssistantApp._strip_leading_choice_marks(text)
        if not stripped:
            return False
        label = label.upper()
        first = stripped[:1].upper()
        if first != label:
            return False
        if len(stripped) == 1:
            return True
        second = stripped[1]
        if second in ".．、:：)）]】 ":
            return True
        return False

    def _option_label_score(self, text: str, label: str) -> int:
        stripped = self._strip_leading_choice_marks(text)
        if not stripped:
            return -1
        label = label.upper()
        upper = stripped.upper()
        if upper == label:
            return 120
        if not upper.startswith(label):
            return -1
        if len(stripped) >= 2:
            second = stripped[1]
            if second in ".．、:：)）]】":
                return 115
            if second == " ":
                return 105
        return -1

    def _item_in_question_region(self, item: OCRTextItem, margin: int = 6) -> bool:
        rect = self._question_region_rect()
        if rect is None:
            return False
        left, top, right, bottom = rect
        return (
            left - margin <= int(item.x) <= right + margin
            and top - margin <= int(item.y) <= bottom + margin
        )

    @staticmethod
    def _feedback_start_y(items: list[OCRTextItem]) -> Optional[int]:
        feedback_words = ("参考答案", "您的答案", "我的答案", "正确答案", "答案解析")
        ys = [
            int(item.y)
            for item in items
            if any(word in normalize_text(item.text) for word in feedback_words)
        ]
        return min(ys) if ys else None

    def _exclude_feedback_items(self, items: list[OCRTextItem]) -> list[OCRTextItem]:
        feedback_y = self._feedback_start_y(items)
        if feedback_y is None:
            return list(items)
        # 页面已出现答题反馈时，反馈行及其下方的 “参考答案: D / 您的答案: A”
        # 不是可点击选项，必须排除，否则重复识别/调试时可能把反馈文字当成候选。
        return [item for item in items if int(item.y) < feedback_y - 4]

    def _option_area_items(self, items: list[OCRTextItem]) -> list[OCRTextItem]:
        rect = self._question_region_rect()
        if rect is None:
            return self._exclude_feedback_items(list(items))
        if self._answers_inside_question_region():
            inside_region = [item for item in items if self._item_in_question_region(item)]
            return self._exclude_feedback_items(inside_region)
        _left, _top, _right, bottom = rect
        below_region = [
            item
            for item in items
            if int(item.y) >= bottom - 4 and not self._item_in_question_region(item)
        ]
        outside_region = [item for item in items if not self._item_in_question_region(item)]
        # 未勾选“答案在框选内”时，不再回退到题目框内部，避免把题干里的“B类/A级”等误当选项。
        candidates = below_region if len(below_region) >= 2 else outside_region
        return self._exclude_feedback_items(candidates)

    def _find_direct_option_item(
        self,
        label: str,
        items: list[OCRTextItem],
        *,
        filter_question_region: bool = True,
    ) -> Optional[OCRTextItem]:
        label = label.upper()
        candidates = self._option_area_items(items) if filter_question_region else list(items)
        best_item: Optional[OCRTextItem] = None
        best_score = -1
        for item in candidates:
            score = self._option_label_score(item.text, label)
            if score < 0:
                continue
            if score > best_score:
                best_score = score
                best_item = item
        if best_item is not None:
            return best_item
        return self._find_embedded_option_item(label, candidates)

    @staticmethod
    def _text_similarity_score(target: str, candidate: str) -> float:
        target = normalize_text(target)
        candidate = normalize_text(candidate)
        if len(target) < 2 or len(candidate) < 2:
            return 0.0
        if target == candidate:
            return 1.0
        if target in candidate or candidate in target:
            return min(len(target), len(candidate)) / max(len(target), len(candidate))
        return difflib.SequenceMatcher(None, target, candidate).ratio()

    def _row_text_items(self, items: list[OCRTextItem]) -> list[OCRTextItem]:
        """Create row-level OCR items to handle split option text.

        Some pages/OCR outputs split one option into several text boxes, for example
        "A." and "选项内容".  A row-level synthetic item lets us match the option
        content even when the option label itself cannot be located reliably.
        """
        row_items: list[OCRTextItem] = []
        for row in self._cluster_items_by_row(items):
            useful = [
                item
                for item in row
                if item.text and not self._is_navigation_like_text(item.text)
            ]
            if not useful:
                continue
            text = " ".join(item.text.strip() for item in useful if item.text.strip())
            xs = [int(item.x) for item in useful]
            ys = [int(item.y) for item in useful]
            bounds = [
                item
                for item in useful
                if item.left is not None
                and item.top is not None
                and item.right is not None
                and item.bottom is not None
            ]
            if bounds:
                left = min(int(item.left) for item in bounds if item.left is not None)
                top = min(int(item.top) for item in bounds if item.top is not None)
                right = max(int(item.right) for item in bounds if item.right is not None)
                bottom = max(int(item.bottom) for item in bounds if item.bottom is not None)
                x = int((left + right) / 2)
                y = int((top + bottom) / 2)
                row_items.append(
                    OCRTextItem(text=text, x=x, y=y, left=left, top=top, right=right, bottom=bottom)
                )
            else:
                row_items.append(
                    OCRTextItem(
                        text=text,
                        x=int(sum(xs) / len(xs)),
                        y=int(sum(ys) / len(ys)),
                    )
                )
        return row_items

    def _find_option_item_by_text(
        self,
        option_text: str,
        items: list[OCRTextItem],
    ) -> Optional[OCRTextItem]:
        target = normalize_text(self._option_text_without_label(option_text))
        if len(target) < 2:
            return None

        option_items = self._option_area_items(items)
        # 第一轮：逐个 OCR 文本框匹配；第二轮：按行合并后匹配。
        # 这样在 A/B/C/D 标签识别不到，或标签和正文被 OCR 拆开时，仍能按选项正文点击。
        candidates = list(option_items) + self._row_text_items(option_items)

        best_item: Optional[OCRTextItem] = None
        best_score = 0.0
        for item in candidates:
            item_text = normalize_text(self._option_text_without_label(item.text))
            if len(item_text) < 2:
                continue
            score = self._text_similarity_score(target, item_text)
            if score > best_score:
                best_score = score
                best_item = item

        # 长文本允许稍低阈值，短文本阈值提高，避免“是/否/对/错”这类短词误点。
        threshold = 0.58 if len(target) >= 8 else 0.72
        if best_item is not None and best_score >= threshold:
            return best_item
        return None

    @staticmethod
    def _answer_label_aliases(label: str) -> list[str]:
        label = label.upper()
        if label in {"TRUE", "T"}:
            return ["对", "正确", "是"]
        if label in {"FALSE"}:
            return ["错", "错误", "否"]
        return [label]

    @staticmethod
    def _suffix_after_answer_label(text: str, label: str) -> Optional[str]:
        raw = (text or "").strip()
        if not raw:
            return None
        aliases = [re.escape(item) for item in AssistantApp._answer_label_aliases(label)]
        if not aliases:
            return None
        # 最新规则：只要 OCR 行里包含目标标签即可匹配，例如 oA、OA、选项A。
        # 若标签后面有内容，优先返回标签后文；否则由调用方用整行文本参与文字匹配。
        pattern = re.compile(
            r"(?:" + "|".join(aliases) + r")"
            r"\s*[.．、:：)）]?\s*(.*)$",
            re.I,
        )
        match = pattern.search(raw)
        if match:
            suffix = AssistantApp._option_text_without_label(match.group(1))
            return suffix.strip()
        return None

    def _find_answer_item_by_label_suffix(
        self,
        label: str,
        option_text: str,
        items: list[OCRTextItem],
    ) -> Optional[OCRTextItem]:
        target = normalize_text(self._option_text_without_label(option_text))
        candidates = self._option_area_items(items)
        best_item: Optional[OCRTextItem] = None
        best_score = 0.0
        for item in candidates:
            suffix = self._suffix_after_answer_label(item.text, label)
            if suffix is None:
                continue
            suffix_norm = normalize_text(suffix)
            item_norm = normalize_text(self._option_text_without_label(item.text))
            if target:
                if suffix_norm == target:
                    score = 1.0
                elif suffix_norm in target or target in suffix_norm:
                    # 长选项常被浏览器截断或 OCR 分段，只要标签后文是正确选项的
                    # 连续片段，就可认为是答案行。
                    score = max(0.72, min(len(suffix_norm), len(target)) / max(len(suffix_norm), len(target)))
                elif item_norm and (target in item_norm or item_norm in target):
                    score = max(0.70, min(len(item_norm), len(target)) / max(len(item_norm), len(target)))
                else:
                    score = max(
                        difflib.SequenceMatcher(None, suffix_norm, target).ratio()
                        if suffix_norm
                        else 0.0,
                        difflib.SequenceMatcher(None, item_norm, target).ratio()
                        if item_norm
                        else 0.0,
                    )
            else:
                # 判断题“对/错”可能没有单独选项文字；能匹配标签本身即可。
                score = 0.75
            if score > best_score:
                best_score = score
                best_item = item
        if best_item is not None and best_score >= 0.58:
            return best_item
        return None

    @staticmethod
    def _cluster_items_by_row(items: list[OCRTextItem], row_tolerance: int = 26) -> list[list[OCRTextItem]]:
        rows: list[list[OCRTextItem]] = []
        for item in sorted(items, key=lambda it: (int(it.y), int(it.x))):
            if not rows:
                rows.append([item])
                continue
            row_y = sum(int(it.y) for it in rows[-1]) / len(rows[-1])
            if abs(int(item.y) - row_y) <= row_tolerance:
                rows[-1].append(item)
            else:
                rows.append([item])
        for row in rows:
            row.sort(key=lambda it: int(it.x))
        return rows

    def _is_navigation_like_text(self, text: str) -> bool:
        norm = normalize_text(text)
        if not norm:
            return False
        navigation_words = list(self.click_config.get("next_button_texts", [])) + list(
            self.click_config.get("prev_button_texts", [])
        )
        return any(normalize_text(word) and normalize_text(word) in norm for word in navigation_words)

    def _find_embedded_option_item(
        self,
        label: str,
        candidates: list[OCRTextItem],
    ) -> Optional[OCRTextItem]:
        """Handle OCR merging a multi-column option row into one text item.

        Example: "A、选项一    B、选项二".  Direct matching can find A only, while B
        is embedded in the same OCR box.  We estimate the embedded label position
        from its character offset within the OCR text box.
        """
        label = label.upper()
        marker_pattern = re.compile(r"(?<![A-Z0-9])([ABCDEF])\s*[.．、:：)）]", re.I)
        option_row_ys = [
            int(item.y)
            for item in candidates
            if any(self._option_label_score(item.text, option_label) >= 0 for option_label in "ABCDEF")
        ]
        min_option_y = min(option_row_ys) if option_row_ys else None
        for item in candidates:
            if min_option_y is not None and int(item.y) < min_option_y - 30:
                continue
            if (
                item.left is None
                or item.top is None
                or item.right is None
                or item.bottom is None
                or item.right <= item.left
                or item.bottom <= item.top
            ):
                continue
            raw = self._strip_leading_choice_marks(item.text)
            matches = list(marker_pattern.finditer(raw))
            if len(matches) < 2:
                continue
            for match in matches:
                if match.group(1).upper() != label:
                    continue
                ratio = max(0.0, min(1.0, match.start(1) / max(1, len(raw))))
                width = int(item.right - item.left)
                estimated_left = int(item.left + width * ratio)
                next_starts = [
                    int(item.left + width * (other.start(1) / max(1, len(raw))))
                    for other in matches
                    if other.start(1) > match.start(1)
                ]
                estimated_right = (
                    min(next_starts) - 6
                    if next_starts
                    else min(int(item.right), estimated_left + max(60, width // len(matches)))
                )
                estimated_right = max(estimated_left + 20, estimated_right)
                return OCRTextItem(
                    text=f"{label}@合并OCR:{item.text}",
                    x=int((estimated_left + estimated_right) / 2),
                    y=int(item.y),
                    left=estimated_left,
                    top=item.top,
                    right=estimated_right,
                    bottom=item.bottom,
                )
        return None

    def _estimate_option_item_from_layout(
        self,
        label: str,
        items: list[OCRTextItem],
    ) -> Optional[OCRTextItem]:
        label = label.upper()
        option_labels = "ABCDEF"
        if label not in set(option_labels):
            return None
        target_index = option_labels.index(label)

        known: dict[str, OCRTextItem] = {}
        for option_label in option_labels:
            direct = self._find_direct_option_item(option_label, items)
            if direct is not None:
                known[option_label] = direct

        if label in known:
            return known[label]

        if len(known) >= 2:
            indexed = sorted((option_labels.index(k), v) for k, v in known.items())
            gaps: list[float] = []
            for (idx_a, item_a), (idx_b, item_b) in zip(indexed, indexed[1:]):
                if idx_b != idx_a:
                    gaps.append((item_b.y - item_a.y) / (idx_b - idx_a))
            if gaps:
                row_gap = sorted(gaps)[len(gaps) // 2]
                nearest_idx, nearest_item = min(
                    indexed, key=lambda pair: abs(pair[0] - target_index)
                )
                estimated_y = int(nearest_item.y + row_gap * (target_index - nearest_idx))
                x_values = [int(item.x) for _idx, item in indexed]
                estimated_x = int(sorted(x_values)[len(x_values) // 2])
                return OCRTextItem(text=f"{label}@选项布局估算", x=estimated_x, y=estimated_y)

        option_area = [
            item
            for item in self._option_area_items(items)
            if not self._is_navigation_like_text(item.text)
            and normalize_text(item.text) not in {"", "提交", "确定", "保存"}
        ]
        rows = self._cluster_items_by_row(option_area)
        rows = [row for row in rows if row]
        if len(rows) >= target_index + 1:
            row = rows[target_index]
            left_item = min(row, key=lambda it: int(it.x))
            return OCRTextItem(
                text=f"{label}@选项行估算:{left_item.text}",
                x=int(left_item.x),
                y=int(left_item.y),
            )
        return None

    @staticmethod
    def _judge_words_for_label(label: str) -> list[str]:
        label = label.upper()
        if label in {"TRUE", "T"}:
            return ["正确", "对", "是", "true", "t"]
        if label in {"FALSE"}:
            return ["错误", "错", "否", "false", "f"]
        return []

    def _find_answer_option_item(
        self,
        label: str,
        question_type: str,
        items: list[OCRTextItem],
    ) -> Optional[OCRTextItem]:
        label = label.upper()
        qtype = normalize_question_type(question_type)

        if qtype == "judge" and label in {"TRUE", "FALSE", "T"}:
            words = [normalize_text(word) for word in self._judge_words_for_label(label)]
            best_item: Optional[OCRTextItem] = None
            best_score = -1
            for item in self._option_area_items(items):
                if (
                    not self._answers_inside_question_region()
                    and self._item_in_question_region(item)
                ):
                    continue
                raw = (item.text or "").strip()
                norm = normalize_text(raw)
                if not norm:
                    continue
                score = -1
                if norm in words:
                    score = 100
                elif any(word and norm.startswith(word) for word in words) and len(norm) <= 8:
                    score = 80
                elif any(word and word in norm for word in words) and len(norm) <= 12:
                    score = 50
                if score > best_score:
                    best_score = score
                    best_item = item
            if best_item is not None:
                return best_item

            fallback = "A" if label in {"TRUE", "T"} else "B"
            return self._find_answer_option_item(fallback, "single", items)

        if label in {"A", "B", "C", "D", "E", "F"}:
            direct = self._find_direct_option_item(label, items)
            if direct is not None:
                return direct
        return None

    @staticmethod
    def _answer_click_point(item: OCRTextItem) -> tuple[int, int]:
        """Click the answer text itself, not the radio/checkbox control."""
        if (
            item.left is not None
            and item.top is not None
            and item.right is not None
            and item.bottom is not None
            and item.right > item.left
            and item.bottom > item.top
        ):
            click_x = int((item.left + item.right) / 2)
            click_y = int((item.top + item.bottom) / 2)
            return click_x, click_y
        return int(item.x), int(item.y)

    @staticmethod
    def _point_distance_sq(a: tuple[int, int], b: tuple[int, int]) -> int:
        dx = int(a[0]) - int(b[0])
        dy = int(a[1]) - int(b[1])
        return dx * dx + dy * dy

    @staticmethod
    def _point_midpoint(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
        return int(round((int(a[0]) + int(b[0])) / 2)), int(round((int(a[1]) + int(b[1])) / 2))

    def _stable_click_target(self, locator, description: str = "") -> Optional[dict[str, Any]]:
        """Calculate click coordinates twice before clicking.

        If the first two coordinates are far apart, calculate once more and use
        the midpoint of the closest pair. The locator receives attempt index 0/1/2
        and should return a dict containing at least x, y, item and source.
        """
        first = locator(0)
        if first is None:
            return None

        if not self.click_config.get("double_check_click_position", True):
            first["position_note"] = "坐标校验未启用"
            return first

        try:
            max_delta = float(
                self.click_config.get(
                    "coordinate_check_max_delta_pixels",
                    DEFAULT_CLICK_CONFIG["coordinate_check_max_delta_pixels"],
                )
            )
        except Exception:
            max_delta = float(DEFAULT_CLICK_CONFIG["coordinate_check_max_delta_pixels"])
        max_delta = max(1.0, max_delta)
        max_delta_sq = int(max_delta * max_delta)

        try:
            retry_delay = float(
                self.click_config.get(
                    "coordinate_check_retry_delay_seconds",
                    DEFAULT_CLICK_CONFIG["coordinate_check_retry_delay_seconds"],
                )
            )
        except Exception:
            retry_delay = float(DEFAULT_CLICK_CONFIG["coordinate_check_retry_delay_seconds"])
        retry_delay = max(0.0, retry_delay)

        if retry_delay > 0:
            time.sleep(retry_delay)
        second = locator(1)
        if second is None:
            first["position_note"] = "坐标双检：第二次定位失败，使用第一次坐标"
            return first

        records = [first, second]
        p1 = (int(first["x"]), int(first["y"]))
        p2 = (int(second["x"]), int(second["y"]))
        dist12_sq = self._point_distance_sq(p1, p2)

        if dist12_sq > max_delta_sq:
            if retry_delay > 0:
                time.sleep(retry_delay)
            third = locator(2)
            if third is not None:
                records.append(third)

        # 从已有定位中选距离最近的两次，使用它们的中点作为最终坐标。
        best_pair = (records[0], records[1] if len(records) > 1 else records[0])
        best_dist_sq = self._point_distance_sq(
            (int(best_pair[0]["x"]), int(best_pair[0]["y"])),
            (int(best_pair[1]["x"]), int(best_pair[1]["y"])),
        )
        for i in range(len(records)):
            for j in range(i + 1, len(records)):
                distance_sq = self._point_distance_sq(
                    (int(records[i]["x"]), int(records[i]["y"])),
                    (int(records[j]["x"]), int(records[j]["y"])),
                )
                if distance_sq < best_dist_sq:
                    best_dist_sq = distance_sq
                    best_pair = (records[i], records[j])

        final_x, final_y = self._point_midpoint(
            (int(best_pair[0]["x"]), int(best_pair[0]["y"])),
            (int(best_pair[1]["x"]), int(best_pair[1]["y"])),
        )
        final = dict(best_pair[1])
        final["x"] = final_x
        final["y"] = final_y
        points_text = ", ".join(
            f"p{idx + 1}=({int(record['x'])},{int(record['y'])})"
            for idx, record in enumerate(records)
        )
        too_far_text = "，触发第三次定位" if dist12_sq > max_delta_sq and len(records) >= 3 else ""
        final["position_note"] = (
            f"坐标双检{too_far_text}：{points_text}，最终=({final_x},{final_y})"
        )
        return final

    def _click_answer_label(
        self,
        label: str,
        question_type: str,
        items: Optional[list[OCRTextItem]],
        option_text: str = "",
    ) -> tuple[str, str]:
        label = label.upper()

        if self.click_config.get("use_ocr_answer_positions", True):
            def locate(attempt: int) -> Optional[dict[str, Any]]:
                # 第一次优先使用调用方已经 OCR 得到的 items；第二/三次重新 OCR，
                # 用于校验页面渲染、鼠标遮挡或 OCR 抖动造成的位置偏移。
                if attempt == 0 and items:
                    search_items = items
                else:
                    search_items = (
                        self._ocr_question_items_fast()
                        if self._answers_inside_question_region()
                        else self._ocr_items_fast()
                    )

                item: Optional[OCRTextItem] = None
                source = ""

                # 1. 优先按 A/B/C/D/E/F 或 判断题“正确/错误”定位。
                item = self._find_answer_option_item(label, question_type, search_items)
                if item is not None:
                    source = "选项字母定位"

                # 2. 若 A/B/C/D 标签定位失败，再尝试“字母 + 选项正文”联合匹配。
                if item is None and option_text:
                    item = self._find_answer_item_by_label_suffix(label, option_text, search_items)
                    if item is not None:
                        source = "字母+正文匹配"

                # 3. 若仍然失败，直接用该选项的文本内容进行模糊匹配并点击。
                if item is None and option_text:
                    item = self._find_option_item_by_text(option_text, search_items)
                    if item is not None:
                        source = "正文内容匹配"

                if item is None:
                    return None
                click_x, click_y = self._answer_click_point(item)
                return {
                    "x": int(click_x),
                    "y": int(click_y),
                    "item": item,
                    "source": source or "OCR",
                }

            target = self._stable_click_target(locate, f"答案 {label}")
            if target is not None:
                click_x, click_y = int(target["x"]), int(target["y"])
                item = target.get("item")
                source = str(target.get("source") or "OCR")
                title = WindowClicker.click(
                    x=click_x,
                    y=click_y,
                    window_keyword=self.click_config["window_keyword"],
                    delay_seconds=self.click_config["click_delay_seconds"],
                )
                center_note = ""
                if isinstance(item, OCRTextItem) and (click_x, click_y) != (int(item.x), int(item.y)):
                    center_note = f"，OCR中心=({item.x},{item.y})"
                position_note = target.get("position_note")
                position_note = f"，{position_note}" if position_note else ""
                item_text = item.text if isinstance(item, OCRTextItem) else "未知"
                return (
                    title,
                    f"{label}@{source}({item_text},点击=({click_x},{click_y}){center_note}{position_note})",
                )

        raise RuntimeError(
            f"未能从当前屏幕 OCR 结果中定位选项 {label}，且没有找到可匹配的选项正文，已取消点击。"
            "请确认目标练习窗口在前台、选项文字可见，或重新框选包含选项的区域。"
        )

    def _button_text_candidates(self, kind: str) -> tuple[list[str], list[str]]:
        kind = "submit" if kind == "submit" else ("prev" if kind == "prev" else "next")

        if kind == "submit":
            source_key = "submit_button_texts"
            default_key = "submit_button_texts"
            negative_keys = ("next_button_texts", "prev_button_texts")
        elif kind == "prev":
            source_key = "prev_button_texts"
            default_key = "prev_button_texts"
            negative_keys = ("next_button_texts", "submit_button_texts")
        else:
            source_key = "next_button_texts"
            default_key = "next_button_texts"
            negative_keys = ("prev_button_texts", "submit_button_texts")

        candidates = [
            normalize_text(text)
            for text in self.click_config.get(
                source_key,
                DEFAULT_CLICK_CONFIG[default_key],
            )
        ]
        negatives: list[str] = []
        for key in negative_keys:
            negatives.extend(
                normalize_text(text)
                for text in self.click_config.get(key, DEFAULT_CLICK_CONFIG.get(key, []))
            )
        if kind != "prev":
            negatives.append(normalize_text("返回"))
        return candidates, negatives

    @staticmethod
    def _find_button_item_in_items(
        items: list[OCRTextItem],
        candidates: list[str],
        negatives: list[str],
    ) -> Optional[OCRTextItem]:
        for item in items:
            norm = normalize_text(item.text)
            if not norm:
                continue
            if any(word and word in norm for word in negatives):
                continue
            if any(candidate and candidate in norm for candidate in candidates):
                return item
        return None

    def _find_navigation_button_item(
        self,
        kind: str = "next",
        items: Optional[list[OCRTextItem]] = None,
    ) -> Optional[OCRTextItem]:
        kind = "submit" if kind == "submit" else ("prev" if kind == "prev" else "next")
        candidates, negatives = self._button_text_candidates(kind)
        remembered_point = self._remembered_point("buttons", kind)

        if remembered_point is not None and items:
            item = self._find_button_item_in_items(
                self._items_near_point(items, remembered_point),
                candidates,
                negatives,
            )
            if item is not None:
                return item

        if remembered_point is not None:
            item = self._find_button_item_in_items(
                self._ocr_items_from_memory_area("buttons", kind),
                candidates,
                negatives,
            )
            if item is not None:
                return item

        if items:
            item = self._find_button_item_in_items(items, candidates, negatives)
            if item is not None:
                return item

        return self._find_button_item_in_items(self._ocr_items_fast(), candidates, negatives)

    def _find_next_button_item(
        self,
        items: Optional[list[OCRTextItem]] = None,
    ) -> Optional[OCRTextItem]:
        return self._find_navigation_button_item("next", items)

    def _find_prev_button_item(
        self,
        items: Optional[list[OCRTextItem]] = None,
    ) -> Optional[OCRTextItem]:
        return self._find_navigation_button_item("prev", items)

    def _find_submit_button_item(
        self,
        items: Optional[list[OCRTextItem]] = None,
    ) -> Optional[OCRTextItem]:
        return self._find_navigation_button_item("submit", items)

    def _click_next_question_button(
        self,
        items: Optional[list[OCRTextItem]] = None,
    ) -> str:
        delay = float(
            self.click_config.get(
                "next_click_delay_seconds",
                DEFAULT_CLICK_CONFIG["next_click_delay_seconds"],
            )
        )
        time.sleep(max(0.0, delay))

        def locate(attempt: int) -> Optional[dict[str, Any]]:
            search_items = items if attempt == 0 else None
            item = self._find_next_button_item(search_items)
            if item is None:
                return None
            return {"x": int(item.x), "y": int(item.y), "item": item, "source": "下一题按钮"}

        target = self._stable_click_target(locate, "下一题按钮")
        if target is None:
            raise RuntimeError("未在屏幕中识别到“下一题/下一页/继续”等按钮。")
        click_x, click_y = int(target["x"]), int(target["y"])
        item = target.get("item")
        title = WindowClicker.click(
            x=click_x,
            y=click_y,
            window_keyword=self.click_config["window_keyword"],
            delay_seconds=0.05,
        )
        memory_item = OCRTextItem(
            text=item.text if isinstance(item, OCRTextItem) else "下一题",
            x=click_x,
            y=click_y,
        )
        self._update_position_memory("buttons", "next", memory_item)
        position_note = target.get("position_note")
        position_note = f"；{position_note}" if position_note else ""
        item_text = item.text if isinstance(item, OCRTextItem) else "下一题"
        return f"{item_text}@({click_x},{click_y}){position_note} / {title}"

    def _click_submit_button(
        self,
        items: Optional[list[OCRTextItem]] = None,
    ) -> str:
        delay = float(
            self.click_config.get(
                "submit_click_delay_seconds",
                DEFAULT_CLICK_CONFIG["submit_click_delay_seconds"],
            )
        )
        time.sleep(max(0.0, delay))

        def locate(attempt: int) -> Optional[dict[str, Any]]:
            search_items = items if attempt == 0 else None
            item = self._find_submit_button_item(search_items)
            if item is None:
                return None
            return {"x": int(item.x), "y": int(item.y), "item": item, "source": "保存按钮"}

        target = self._stable_click_target(locate, "保存按钮")
        if target is None:
            raise RuntimeError("未在屏幕中识别到“保存/提交/确定”等按钮。")
        click_x, click_y = int(target["x"]), int(target["y"])
        item = target.get("item")
        title = WindowClicker.click(
            x=click_x,
            y=click_y,
            window_keyword=self.click_config["window_keyword"],
            delay_seconds=0.05,
        )
        memory_item = OCRTextItem(
            text=item.text if isinstance(item, OCRTextItem) else "保存",
            x=click_x,
            y=click_y,
        )
        self._update_position_memory("buttons", "submit", memory_item)
        position_note = target.get("position_note")
        position_note = f"；{position_note}" if position_note else ""
        item_text = item.text if isinstance(item, OCRTextItem) else "保存"
        return f"{item_text}@({click_x},{click_y}){position_note} / {title}"

    def _wait_for_question_change(
        self,
        old_text: str,
        timeout_seconds: float,
        poll_seconds: float = 0.35,
    ) -> tuple[bool, str, list[OCRTextItem]]:
        """Wait briefly after submit/next click and detect whether the question changed.

        This prevents two common multi-select timing bugs:
        1. clicking "下一题" too soon while the save/submit action is still refreshing;
        2. clicking "下一题" again after "保存" has already advanced to the next question.
        """
        try:
            timeout = max(0.0, float(timeout_seconds))
        except Exception:
            timeout = 0.0
        try:
            poll = max(0.1, float(poll_seconds))
        except Exception:
            poll = 0.35

        old_norm = normalize_text(old_text)
        deadline = time.time() + timeout
        last_raw = ""
        last_items: list[OCRTextItem] = []

        while time.time() < deadline:
            remaining = max(0.0, min(poll, deadline - time.time()))
            if remaining > 0:
                if self.loop_stop_event.wait(remaining):
                    break
            try:
                items = self._ocr_question_items_fast()
                raw = OCREngine.items_to_text(items).strip()
                norm = normalize_text(raw)
                if norm:
                    last_raw = raw
                    last_items = items
                    if old_norm and norm != old_norm:
                        return True, raw, items
            except Exception:
                continue

        return False, last_raw, last_items

    def _mouse_restore_point(self) -> Optional[tuple[int, int]]:
        configured = self.click_config.get(
            "mouse_restore_position",
            DEFAULT_CLICK_CONFIG["mouse_restore_position"],
        )
        x, y = self._coord_pair(configured)
        if x != 0 or y != 0:
            return x, y
        try:
            rect = WindowClicker.get_target_window_rect(
                self.click_config.get("window_keyword", DEFAULT_FOREGROUND_WINDOW)
            )
        except Exception:
            rect = None
        if rect is not None:
            left, top, right, bottom = rect
            return max(left + 10, right - 30), max(top + 10, bottom - 30)
        left, top, right, bottom = self._virtual_screen_rect()
        return max(left + 10, right - 30), max(top + 10, bottom - 30)

    def _restore_mouse_after_click(self) -> None:
        if not self.click_config.get("restore_mouse_after_click", True):
            return
        point = self._mouse_restore_point()
        if point is None:
            return
        try:
            WindowClicker.move_mouse(*point)
        except Exception:
            pass

    def _rescan_and_requery_after_click_failure(
        self,
        result: AnswerResult,
        error_message: str,
    ) -> Optional[tuple[AnswerResult, list[OCRTextItem]]]:
        """After click-position/text matching failure, refresh OCR and ask DeepSeek once more.

        本版本禁用本地题库，不再删除、查询或写入 SQLite。
        任意失败都会被调用方吞掉，并进入随机兜底，避免自动循环中断。
        """
        self.set_status("定位失败，正在重新 OCR 并重新请求 DeepSeek……")

        try:
            retry_items = self._ocr_question_items_fast()
            retry_raw = OCREngine.items_to_text(retry_items).strip()
            if not normalize_text(retry_raw):
                return None

            self.input_text.delete("1.0", tk.END)
            self.input_text.insert(tk.END, retry_raw)
            self.answer_text.insert(
                tk.END,
                f"\n\n定位失败后已重新 OCR，并将重新请求 DeepSeek。原错误：{error_message}\n",
            )

            retry_result = self._query_text(retry_raw)
            self.last_result = retry_result
            self._show_result(retry_result)
            return retry_result, retry_items
        except Exception as e:
            self.answer_text.insert(
                tk.END,
                f"\n\n重新 OCR/请求 DeepSeek 仍失败，将进入随机兜底：{e}\n",
            )
            return None

    def _multi_random_extra_candidates(
        self,
        selected_labels: list[str],
        option_texts: dict[str, str],
        items: Optional[list[OCRTextItem]],
    ) -> list[str]:
        """Return randomized candidate labels to supplement an under-recognized multi-choice answer."""
        selected = {label.upper() for label in selected_labels}

        # Prefer labels that actually appear in the parsed question/options.
        candidates: list[str] = []
        for label in "ABCDEF":
            if label not in selected and (option_texts.get(label) or "").strip():
                candidates.append(label)

        # If the OCR split did not recover option texts, use currently recognizable choices.
        if not candidates:
            try:
                found = self._recognizable_abcd_items(items)
            except Exception:
                found = {}
            for label in "ABCD":
                if label not in selected and label in found:
                    candidates.append(label)

        # Last resort: use the configured random fallback option range.
        if not candidates:
            configured = [
                str(label).upper()
                for label in self.click_config.get(
                    "random_fallback_options",
                    DEFAULT_CLICK_CONFIG["random_fallback_options"],
                )
            ]
            for label in configured or ["A", "B", "C", "D"]:
                if label in {"A", "B", "C", "D", "E", "F"} and label not in selected:
                    candidates.append(label)

        # Deduplicate while preserving order, then shuffle for true random supplementing.
        unique_candidates: list[str] = []
        for label in candidates:
            if label not in unique_candidates:
                unique_candidates.append(label)
        random.shuffle(unique_candidates)
        return unique_candidates

    def _click_multi_random_extra_if_needed(
        self,
        labels: list[str],
        clicked: list[str],
        question_type: str,
        option_items: Optional[list[OCRTextItem]],
        option_texts: dict[str, str],
    ) -> tuple[list[str], str]:
        """If a multi-choice answer contains too few labels, randomly add extra selections.

        This helper never raises. It is only a robustness fallback for practice pages
        that reject multi-choice submissions with fewer than two selected options.
        """
        if normalize_question_type(question_type) != "multi":
            return labels, ""
        if not self.click_config.get("multi_random_extra_when_single", True):
            return labels, ""

        try:
            min_selected = max(
                1,
                int(
                    self.click_config.get(
                        "multi_min_selected_answers",
                        DEFAULT_CLICK_CONFIG["multi_min_selected_answers"],
                    )
                ),
            )
        except Exception:
            min_selected = DEFAULT_CLICK_CONFIG["multi_min_selected_answers"]

        selectable_labels = [label for label in labels if label.upper() in set("ABCDEF")]
        if len(selectable_labels) >= min_selected:
            return labels, ""

        need_count = min_selected - len(selectable_labels)
        candidates = self._multi_random_extra_candidates(labels, option_texts, option_items)
        if not candidates:
            return labels, "；多选补选：未找到可补选项"

        added: list[str] = []
        errors: list[str] = []
        for extra_label in candidates:
            if len(added) >= need_count:
                break
            try:
                _title, clicked_item = self._click_answer_label(
                    label=extra_label,
                    question_type=question_type,
                    items=option_items,
                    option_text=option_texts.get(extra_label.upper(), ""),
                )
                clicked.append(f"补选{clicked_item}")
                labels.append(extra_label)
                added.append(extra_label)
                time.sleep(0.18)
            except Exception as e:
                errors.append(f"{extra_label}:{e}")

        if added:
            return labels, f"；多选仅识别到{len(selectable_labels)}项，已随机补选：{''.join(added)}"
        return labels, f"；多选补选失败：{' | '.join(errors[:2])}"

    def _fallback_random_labels(self, result: AnswerResult) -> tuple[list[str], list[str]]:
        base_labels = [
            str(item).upper()
            for item in self.click_config.get(
                "random_fallback_options",
                DEFAULT_CLICK_CONFIG["random_fallback_options"],
            )
            if str(item).upper() in {"A", "B", "C", "D"}
        ]
        if not base_labels:
            base_labels = ["A", "B", "C", "D"]

        predicted = [
            label.upper()
            for label in self._extract_answer_labels(result.question_type, result.answer)
            if label.upper() in set(base_labels)
        ]
        if predicted:
            # 有明确答案字母时，其他选项属于“肯定错误”，先排除。
            excluded = [label for label in base_labels if label not in set(predicted)]
            return predicted, excluded
        return base_labels, []

    def _configured_option_item(self, label: str) -> Optional[OCRTextItem]:
        label = label.upper()
        coords = self.click_config.get("coords") or {}
        x, y = self._coord_pair(coords.get(label))
        if x != 0 or y != 0:
            return OCRTextItem(text=f"{label}@手动坐标", x=x, y=y)
        memory = self._normalize_position_memory().get("options", {})
        point = memory.get(label)
        if point:
            x, y = self._coord_pair(point)
            if x != 0 or y != 0:
                return OCRTextItem(text=f"{label}@位置记忆", x=x, y=y)
        return None

    def _estimate_option_item_from_ad_bounds(self, label: str) -> Optional[OCRTextItem]:
        """Estimate A-D position by using A as upper bound and D as lower bound."""
        label = label.upper()
        if label not in {"A", "B", "C", "D"}:
            return None

        # If the exact label coordinate exists, use it first.
        exact = self._configured_option_item(label)
        if exact is not None:
            return exact

        coords = self.click_config.get("coords") or {}
        ax, ay = self._coord_pair(coords.get("A"))
        dx, dy = self._coord_pair(coords.get("D"))
        if (ax == 0 and ay == 0) or (dx == 0 and dy == 0):
            return None

        index = {"A": 0, "B": 1, "C": 2, "D": 3}[label]
        ratio = index / 3.0
        x = int(ax + (dx - ax) * ratio)
        y = int(ay + (dy - ay) * ratio)
        return OCRTextItem(text=f"{label}@A-D边界估算", x=x, y=y)

    def _recognizable_abcd_items(self, items: Optional[list[OCRTextItem]]) -> dict[str, OCRTextItem]:
        try:
            search_items = items or self._ocr_items_fast()
        except Exception:
            search_items = items or []

        found: dict[str, OCRTextItem] = {}
        for label in "ABCD":
            try:
                item = self._find_answer_option_item(label, "single", search_items)
            except Exception:
                item = None
            if item is not None:
                found[label] = item

        # If labels themselves are not recognized, use row order as a weak fallback.
        if not found and search_items:
            rows = self._cluster_items_by_row(self._option_area_items(search_items))
            usable_rows = []
            for row in rows:
                useful = [
                    item
                    for item in row
                    if item.text and not self._is_navigation_like_text(item.text)
                ]
                if useful:
                    usable_rows.append(useful)
            for label, row in zip("ABCD", usable_rows[:4]):
                left_item = min(row, key=lambda it: int(it.x))
                found[label] = OCRTextItem(
                    text=f"{label}@可识别行:{left_item.text}",
                    x=int(left_item.x),
                    y=int(left_item.y),
                    left=left_item.left,
                    top=left_item.top,
                    right=left_item.right,
                    bottom=left_item.bottom,
                )
        return found

    def _random_fallback_click(
        self,
        result: AnswerResult,
        items: Optional[list[OCRTextItem]],
        reason: str,
    ) -> tuple[str, str, bool]:
        """Last-resort click strategy that never raises.

        Priority:
        1. Exclude definitely wrong labels when the answer already contains A-D.
        2. Use configured A/D bounds to estimate the target row.
        3. Randomly choose one recognizable A-D option from OCR.
        4. If nothing is clickable, return gracefully without interrupting.
        """
        if not self.click_config.get("random_fallback_enabled", True):
            return "", "随机兜底未启用", False

        allowed_labels, excluded_labels = self._fallback_random_labels(result)
        random.shuffle(allowed_labels)
        excluded_note = f"，已排除：{''.join(excluded_labels)}" if excluded_labels else ""

        last_error = ""
        if self.click_config.get("random_fallback_use_option_bounds", True):
            for label in allowed_labels:
                def locate_bounds(attempt: int, target_label: str = label) -> Optional[dict[str, Any]]:
                    item = self._estimate_option_item_from_ad_bounds(target_label)
                    if item is None:
                        return None
                    return {"x": int(item.x), "y": int(item.y), "item": item, "source": "A-D边界/坐标"}

                try:
                    target = self._stable_click_target(locate_bounds, f"随机兜底 {label}")
                    if target is None:
                        continue
                    click_x, click_y = int(target["x"]), int(target["y"])
                    title = WindowClicker.click(
                        x=click_x,
                        y=click_y,
                        window_keyword=self.click_config["window_keyword"],
                        delay_seconds=self.click_config["click_delay_seconds"],
                    )
                    position_note = target.get("position_note")
                    position_note = f"；{position_note}" if position_note else ""
                    return (
                        title,
                        f"随机兜底:{label}@A-D边界/坐标({click_x},{click_y}){excluded_note}{position_note}；原因：{reason}",
                        True,
                    )
                except Exception as e:
                    last_error = str(e)

        found = self._recognizable_abcd_items(items)
        pool = [(label, item) for label, item in found.items() if label in set(allowed_labels)]
        if not pool:
            # 如果排除法之后没有可点击项，就退到“ABCD 中随机选一个可识别的选项”。
            pool = list(found.items())
        random.shuffle(pool)
        for label, item in pool:
            def locate_recognizable(attempt: int, target_label: str = label, first_item: OCRTextItem = item) -> Optional[dict[str, Any]]:
                if attempt == 0:
                    located = first_item
                else:
                    try:
                        fresh_items = self._ocr_items_fast()
                        located = self._find_answer_option_item(target_label, "single", fresh_items)
                    except Exception:
                        located = None
                    if located is None:
                        located = first_item
                click_x, click_y = self._answer_click_point(located)
                return {"x": int(click_x), "y": int(click_y), "item": located, "source": "可识别选项"}

            try:
                target = self._stable_click_target(locate_recognizable, f"随机可识别选项 {label}")
                if target is None:
                    continue
                click_x, click_y = int(target["x"]), int(target["y"])
                located_item = target.get("item")
                title = WindowClicker.click(
                    x=click_x,
                    y=click_y,
                    window_keyword=self.click_config["window_keyword"],
                    delay_seconds=self.click_config["click_delay_seconds"],
                )
                position_note = target.get("position_note")
                position_note = f"；{position_note}" if position_note else ""
                item_text = located_item.text if isinstance(located_item, OCRTextItem) else item.text
                return (
                    title,
                    f"随机兜底:{label}@可识别选项({item_text},点击=({click_x},{click_y})){excluded_note}{position_note}；原因：{reason}",
                    True,
                )
            except Exception as e:
                last_error = str(e)

        # Final fallback: any configured A-D coordinate, even without A/D bounds.
        coord_pool = []
        for label in allowed_labels + [label for label in "ABCD" if label not in allowed_labels]:
            item = self._configured_option_item(label)
            if item is not None:
                coord_pool.append((label, item))
        random.shuffle(coord_pool)
        for label, item in coord_pool:
            def locate_configured(attempt: int, target_label: str = label, first_item: OCRTextItem = item) -> Optional[dict[str, Any]]:
                located = self._configured_option_item(target_label) or first_item
                return {"x": int(located.x), "y": int(located.y), "item": located, "source": "固定坐标"}

            try:
                target = self._stable_click_target(locate_configured, f"固定坐标兜底 {label}")
                if target is None:
                    continue
                click_x, click_y = int(target["x"]), int(target["y"])
                title = WindowClicker.click(
                    x=click_x,
                    y=click_y,
                    window_keyword=self.click_config["window_keyword"],
                    delay_seconds=self.click_config["click_delay_seconds"],
                )
                position_note = target.get("position_note")
                position_note = f"；{position_note}" if position_note else ""
                return (
                    title,
                    f"随机兜底:{label}@固定坐标({click_x},{click_y}){excluded_note}{position_note}；原因：{reason}",
                    True,
                )
            except Exception as e:
                last_error = str(e)

        return (
            "",
            f"随机兜底未找到可点击 ABCD 选项；原因：{reason}；最后错误：{last_error}",
            False,
        )

    def auto_click_current_answer(self, manual: bool = False) -> None:
        if self.last_result is None:
            messagebox.showwarning("没有答案", "请先查询得到答案。")
            return
        self._try_auto_click(self.last_result, manual=manual)

    def _try_auto_click(
        self,
        result: AnswerResult,
        manual: bool = False,
        foreground_countdown: bool = True,
        pre_ocr_items: Optional[list[OCRTextItem]] = None,
        allow_recovery: bool = True,
    ) -> bool:
        self.click_config = self._read_click_config_from_ui()

        if not self.click_config["enabled"]:
            if manual:
                messagebox.showwarning("未启用", "请先勾选“启用自动点击”，并确认这是你的练习窗口。")
            return False

        labels = self._extract_answer_labels(result.question_type, result.answer)
        option_texts = split_question_and_options(result.question)
        if (
            not any(option_texts.get(label) for label in "ABCDEF")
            and result.matched_question
        ):
            # 兼容旧结果对象：如果外部传入 matched_question，则可回退拿到选项正文。
            # DeepSeek-only 正常流程下 matched_question 为空，不会走到这里。
            matched_option_texts = split_question_and_options(result.matched_question)
            if any(matched_option_texts.get(label) for label in "ABCDEF"):
                option_texts = matched_option_texts

        try:
            clicked: list[str] = []
            title = ""
            submitted = False
            answer_clicked = False

            if (
                foreground_countdown
                and WindowClicker.uses_foreground_window(self.click_config.get("window_keyword", ""))
            ):
                self.set_status("将在 3 秒后点击；若助手在前台会自动最小化，请将目标练习窗口切到前台。")
                self._prepare_foreground_target_window("点击")
                time.sleep(3.0)
            self._prepare_foreground_target_window("点击")
            WindowClicker.activate_target_window(
                self.click_config["window_keyword"],
                delay_seconds=0.15,
            )

            option_items: list[OCRTextItem] = []
            if self.click_config.get("use_ocr_answer_positions", True):
                self.set_status("正在 OCR 识别答案文字位置……")
                option_items = pre_ocr_items if pre_ocr_items is not None else []

            if labels:
                for label in labels:
                    title, clicked_item = self._click_answer_label(
                        label=label,
                        question_type=result.question_type,
                        items=option_items,
                        option_text=option_texts.get(label.upper(), ""),
                    )
                    clicked.append(clicked_item)
                    time.sleep(0.18)

                labels, multi_extra_note = self._click_multi_random_extra_if_needed(
                    labels=labels,
                    clicked=clicked,
                    question_type=result.question_type,
                    option_items=option_items or pre_ocr_items,
                    option_texts=option_texts,
                )
                clicked_text = "/".join(clicked) + multi_extra_note
                answer_clicked = True
            else:
                # 答案本身无法解析为 A/B/C/D 时，不结束流程，直接进入随机兜底。
                title, fallback_text, fallback_clicked = self._random_fallback_click(
                    result,
                    option_items or pre_ocr_items,
                    f"无法从答案“{result.answer}”中解析出选项字母",
                )
                answer_clicked = fallback_clicked
                clicked_text = fallback_text

            question_changed_after_submit = False
            changed_after_submit_text = ""
            if (
                self.click_config.get("click_submit_after_answer", True)
                and normalize_question_type(result.question_type) == "multi"
            ):
                try:
                    submit_info = self._click_submit_button(option_items or pre_ocr_items)
                    submitted = True
                    clicked_text = f"{clicked_text}；保存：{submit_info}"

                    # 保存/提交后先等待页面稳定。若已经进入下一题，就不能再点“下一题”，
                    # 否则会直接跳过下一题。
                    wait_seconds = float(
                        self.click_config.get(
                            "post_submit_page_wait_seconds",
                            DEFAULT_CLICK_CONFIG["post_submit_page_wait_seconds"],
                        )
                    )
                    question_changed_after_submit, changed_after_submit_text, _changed_items = (
                        self._wait_for_question_change(result.question, wait_seconds)
                        if wait_seconds > 0
                        else (False, "", [])
                    )
                    if question_changed_after_submit:
                        clicked_text = f"{clicked_text}；保存后已进入下一题，跳过额外下一题点击"
                except Exception as e:
                    clicked_text = f"{clicked_text}；保存未点击：{e}"

            if self.click_config.get("click_next_after_answer", True) and answer_clicked:
                if submitted and question_changed_after_submit:
                    # 保存按钮已经完成跳题，不能继续点下一题。
                    pass
                elif submitted and not self.click_config.get("click_next_after_submit", True):
                    clicked_text = f"{clicked_text}；已设置为保存后不额外点击下一题"
                else:
                    # 保存后页面可能刷新，所以不要复用保存前的 OCR items。
                    next_items = None if submitted else (option_items or pre_ocr_items)
                    try:
                        next_info = self._click_next_question_button(next_items)
                        clicked_text = f"{clicked_text}；下一题：{next_info}"
                    except Exception as e:
                        clicked_text = f"{clicked_text}；下一题未点击：{e}"
            self._restore_mouse_after_click()
            self.set_status(f"已在窗口“{title or '未知窗口'}”点击：{clicked_text}。")
            if manual:
                messagebox.showinfo("已点击", f"已在窗口“{title or '未知窗口'}”点击：{clicked_text}。")
            return True
        except Exception as e:
            self._restore_mouse_after_click()
            message = str(e)

            # 第一次失败：重新 OCR + 重新请求 DeepSeek，再重试一次。
            if allow_recovery:
                recovered = self._rescan_and_requery_after_click_failure(result, message)
                if recovered is not None:
                    retry_result, retry_items = recovered
                    retried = self._try_auto_click(
                        retry_result,
                        manual=manual,
                        foreground_countdown=False,
                        pre_ocr_items=retry_items,
                        allow_recovery=False,
                    )
                    if retried:
                        return True

            # 重新识别仍失败：进入排除法/随机兜底，保证本轮不会因为定位失败而抛出中断。
            try:
                fallback_items = pre_ocr_items
                if fallback_items is None:
                    try:
                        fallback_items = self._ocr_items_fast()
                    except Exception:
                        fallback_items = []
                title, fallback_text, fallback_clicked = self._random_fallback_click(
                    result,
                    fallback_items,
                    message,
                )
                clicked_text = fallback_text

                submitted = False
                question_changed_after_submit = False
                if (
                    fallback_clicked
                    and self.click_config.get("click_submit_after_answer", True)
                    and normalize_question_type(result.question_type) == "multi"
                ):
                    try:
                        submit_info = self._click_submit_button(None)
                        submitted = True
                        clicked_text = f"{clicked_text}；保存：{submit_info}"
                        wait_seconds = float(
                            self.click_config.get(
                                "post_submit_page_wait_seconds",
                                DEFAULT_CLICK_CONFIG["post_submit_page_wait_seconds"],
                            )
                        )
                        question_changed_after_submit, _changed_raw, _changed_items = (
                            self._wait_for_question_change(result.question, wait_seconds)
                            if wait_seconds > 0
                            else (False, "", [])
                        )
                        if question_changed_after_submit:
                            clicked_text = f"{clicked_text}；保存后已进入下一题，跳过额外下一题点击"
                    except Exception as submit_error:
                        clicked_text = f"{clicked_text}；保存未点击：{submit_error}"

                if fallback_clicked and self.click_config.get("click_next_after_answer", True):
                    if submitted and question_changed_after_submit:
                        pass
                    elif submitted and not self.click_config.get("click_next_after_submit", True):
                        clicked_text = f"{clicked_text}；已设置为保存后不额外点击下一题"
                    else:
                        try:
                            next_info = self._click_next_question_button(None if submitted else fallback_items)
                            clicked_text = f"{clicked_text}；下一题：{next_info}"
                        except Exception as next_error:
                            clicked_text = f"{clicked_text}；下一题未点击：{next_error}"
                self._restore_mouse_after_click()
                self.set_status(f"已执行失败兜底：{clicked_text}")
                if manual:
                    messagebox.showinfo("已执行兜底", clicked_text)
                # 即使最终没有可点击项，也返回 True，避免循环因异常直接中断。
                return True
            except Exception as fallback_error:
                self._restore_mouse_after_click()
                final_message = f"自动点击失败，但已吞掉异常以避免中断：{message}；兜底错误：{fallback_error}"
                if manual:
                    messagebox.showwarning("自动点击兜底失败", final_message)
                else:
                    self.answer_text.insert(tk.END, f"\n\n{final_message}")
                    self.set_status(final_message)
                return True

    def import_sample(self) -> None:
        messagebox.showinfo("本地题库已禁用", "当前版本已禁用本地题库，示例 CSV 不再导入。")

    def import_csv(self) -> None:
        messagebox.showinfo("本地题库已禁用", "当前版本已禁用本地题库，所有问题都会直接请求 DeepSeek。")

    def capture_ocr_async(self) -> None:
        self._run_async(self._capture_ocr)

    def _capture_ocr(self) -> None:
        self.set_status("正在截屏并 OCR，请稍候……")
        try:
            text = self._ocr_question_text_fast()
            self.input_text.delete("1.0", tk.END)
            self.input_text.insert(tk.END, text)
            self.set_status("OCR 完成。可以点击“查询答案”。")
        except Exception as e:
            self.set_status("OCR 失败。")
            messagebox.showerror(
                "OCR 失败",
                f"{e}\n\n如果尚未安装 OCR 依赖，请先 pip install -r requirements.txt；也可以手动粘贴题目文本查询。",
            )

    def query_async(self) -> None:
        self._run_async(self._query)

    def _save_answer_result_to_bank(self, result: AnswerResult, source: str = "deepseek") -> bool:
        """本版本禁用本地题库，保留函数只为兼容旧按钮/旧调用。"""
        return False

    def _query_text(self, raw: str) -> AnswerResult:
        if not self.deepseek.is_configured():
            raise RuntimeError("DeepSeek 未配置。请复制 .env.example 为 .env，并填写 DEEPSEEK_API_KEY。")
        self.set_status("正在请求 DeepSeek……")
        result = self.deepseek.ask(raw)
        result.source = "deepseek"
        result.matched_question = ""
        self.set_status("DeepSeek 已返回。")
        return result

    def _query(self) -> None:
        raw = self.input_text.get("1.0", tk.END).strip()
        if not raw:
            messagebox.showwarning("缺少题目", "请先 OCR 或粘贴题目文本。")
            return

        self.answer_text.delete("1.0", tk.END)
        self.set_status("正在请求 DeepSeek……")

        try:
            result = self._query_text(raw)
        except Exception as e:
            self.set_status("查询失败。")
            self.answer_text.insert(tk.END, f"{e}\n")
            return

        self.last_result = result
        self._show_result(result)
        self.set_status(f"完成。来源：{result.source}")
        if self.auto_click_var.get():
            self._try_auto_click(result, manual=False)

    def start_loop_async(self) -> None:
        if self.loop_running:
            messagebox.showinfo("循环运行中", "自动循环已经在运行。")
            return
        self.click_config = self._read_click_config_from_ui()
        if not self.click_config["enabled"]:
            messagebox.showwarning("未启用", "请先勾选“启用自动点击”，并确认这是你的练习窗口。")
            return
        self.save_click_config()
        self.loop_stop_event.clear()
        try:
            self.root.iconify()
        except Exception:
            pass
        self._run_async(self._auto_loop)

    def stop_loop(self) -> None:
        self.loop_stop_event.set()
        self.set_status("已请求停止循环，会在当前步骤结束后停止。")

    def _auto_loop(self) -> None:
        self.loop_running = True
        last_norm = ""
        same_question_seen = 0
        max_rounds = int(self.click_config.get("loop_max_rounds", 50))
        interval = float(self.click_config.get("loop_interval_seconds", 1.2))
        post_click_wait = float(
            self.click_config.get(
                "post_click_page_wait_seconds",
                DEFAULT_CLICK_CONFIG["post_click_page_wait_seconds"],
            )
        )
        same_grace_rounds = int(
            self.click_config.get(
                "same_question_grace_rounds",
                DEFAULT_CLICK_CONFIG["same_question_grace_rounds"],
            )
        )
        try:
            if WindowClicker.uses_foreground_window(self.click_config.get("window_keyword", "")):
                self.set_status("循环将在 3 秒后开始；请将目标练习窗口切到前台。")
                if self.loop_stop_event.wait(3.0):
                    self.set_status("循环已停止。")
                    return
            for round_index in range(1, max_rounds + 1):
                if self.loop_stop_event.is_set():
                    self.set_status("循环已停止。")
                    break

                self.set_status(f"循环第 {round_index}/{max_rounds} 轮：正在识别题目区域……")
                question_items = self._ocr_question_items_fast()
                using_question_region = self._question_region_rect() is not None
                ocr_items = question_items
                raw = OCREngine.items_to_text(ocr_items).strip()
                norm = normalize_text(raw)
                if not norm:
                    self.set_status("循环结束：未识别到题目。")
                    break

                if (
                    self.click_config.get("stop_on_repeated_question", True)
                    and last_norm
                    and norm == last_norm
                ):
                    same_question_seen += 1
                    if same_question_seen <= same_grace_rounds:
                        self.set_status(
                            f"题目暂未变化，等待页面刷新 {same_question_seen}/{same_grace_rounds}……"
                        )
                        if self.loop_stop_event.wait(interval):
                            self.set_status("循环已停止。")
                            break
                        continue
                    self.set_status("循环结束：题目连续未变化，已停止以避免重复点击。")
                    break
                same_question_seen = 0

                self.input_text.delete("1.0", tk.END)
                self.input_text.insert(tk.END, raw)

                self.answer_text.delete("1.0", tk.END)
                self.set_status(f"循环第 {round_index}/{max_rounds} 轮：正在请求 DeepSeek……")
                try:
                    result = self._query_text(raw)
                except Exception as e:
                    # DeepSeek 识别失败时也不终止循环，交给随机兜底点击。
                    result = AnswerResult(
                        source="query_failed_fallback",
                        question=raw,
                        answer="",
                        confidence=0.0,
                        explanation=f"查询失败，进入随机兜底：{e}",
                        question_type=infer_question_type(raw, ""),
                    )
                    self.answer_text.insert(tk.END, result.explanation + "\n")
                self.last_result = result
                self._show_result(result)

                labels = self._extract_answer_labels(result.question_type, result.answer)
                if not labels:
                    self.set_status("答案无法转换成可点击选项，将进入随机兜底。")

                clicked = self._try_auto_click(
                    result,
                    manual=False,
                    foreground_countdown=False,
                    pre_ocr_items=(
                        question_items
                        if using_question_region
                        and self._answers_inside_question_region()
                        else (None if using_question_region else ocr_items)
                    ),
                )
                if not clicked:
                    self.set_status("本轮未完成点击，但已避免中断；将继续下一轮或等待页面变化。")
                elif post_click_wait > 0:
                    # 给保存/下一题后的页面刷新留出稳定时间，避免下一轮 OCR 仍扫到旧题。
                    if self.loop_stop_event.wait(post_click_wait):
                        self.set_status("循环已停止。")
                        break

                last_norm = norm
                if self.loop_stop_event.wait(interval):
                    self.set_status("循环已停止。")
                    break
            else:
                self.set_status(f"循环结束：已达到最多 {max_rounds} 轮。")
        except Exception as e:
            self.set_status(f"循环失败：{e}")
            messagebox.showerror("循环失败", str(e))
        finally:
            self.loop_running = False

    def _show_result(self, result: AnswerResult) -> None:
        lines = [
            f"来源：{result.source}",
            f"题型：{result.question_type}",
            f"答案：{result.answer}",
            f"置信度：{result.confidence:.2f}",
            "",
            "解释：",
            result.explanation or "无",
        ]
        # DeepSeek-only 版本不显示本地题库匹配信息。
        self.answer_text.delete("1.0", tk.END)
        self.answer_text.insert(tk.END, "\n".join(lines))

    def save_current_result(self) -> None:
        messagebox.showinfo("本地题库已禁用", "当前版本不再保存答案到本地题库，所有问题都会直接请求 DeepSeek。")

    def _run_async(self, fn) -> None:
        def runner() -> None:
            try:
                fn()
            except Exception as e:
                self.set_status("执行失败。")
                messagebox.showerror("执行失败", str(e))

        threading.Thread(target=runner, daemon=True).start()


def main() -> None:
    WindowClicker.enable_dpi_awareness()
    root = tk.Tk()
    AssistantApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
