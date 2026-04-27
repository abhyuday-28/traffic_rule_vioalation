from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageTk

from traffic_system.emailer import send_violation_email, send_violation_telegram
from traffic_system.pipeline import AnalysisResult, EngineConfig, TrafficViolationEngine, ViolationRecord
from traffic_system.settings import OUTPUT_ROOT, SAMPLE_IMAGES_DIR, SAMPLE_VIDEOS_DIR, SUPPORTED_IMAGE_TYPES, SUPPORTED_VIDEO_TYPES, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID


class TrafficApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Traffic Rule Violation System")
        self.root.geometry("1600x900")
        self.root.minsize(1500, 860)
        self.root.configure(bg="#edf2f4")

        self.engine = TrafficViolationEngine()
        self.capture: cv2.VideoCapture | None = None
        self.current_source_path: str | None = None
        self.frame_index = 0
        self.current_image = None
        self.violations: list[ViolationRecord] = []
        self.video_running = False

        self.source_type = tk.StringVar(value="image")
        self.detect_helmet = tk.BooleanVar(value=True)
        self.detect_triple = tk.BooleanVar(value=True)
        self.detect_red = tk.BooleanVar(value=True)
        self.stop_line_ratio = tk.DoubleVar(value=0.68)

        self.smtp_host = tk.StringVar(value="smtp.gmail.com")
        self.smtp_port = tk.StringVar(value="587")
        self.smtp_user = tk.StringVar()
        self.smtp_password = tk.StringVar()
        self.sender_email = tk.StringVar()
        self.authority_email = tk.StringVar()
        self.status_text = tk.StringVar(value="Ready.")
        self.messages_text = tk.StringVar(value="\n".join(self.engine.capability_messages()))
        self.details_text = tk.StringVar(value="No violation selected.")

        self._build_ui()
        self._set_status("Application started. Select an image, video, or live camera source.")

    def _build_ui(self) -> None:
        top_bar = tk.Frame(self.root, bg="#12343b", height=72)
        top_bar.pack(fill="x")

        tk.Label(
            top_bar,
            text="Traffic Rule Violation System",
            font=("Segoe UI", 22, "bold"),
            bg="#12343b",
            fg="#f8f5f1",
        ).pack(side="left", padx=20, pady=16)

        tk.Label(
            top_bar,
            text="Image, video, or live camera analysis with evidence capture and email reporting",
            font=("Segoe UI", 10),
            bg="#12343b",
            fg="#d7e3e5",
        ).pack(side="left", padx=12, pady=20)

        body = ttk.Panedwindow(self.root, orient="horizontal")
        body.pack(fill="both", expand=True, padx=14, pady=14)

        controls_outer = tk.Frame(body, bg="#ffffff", bd=1, relief="solid", width=360)
        controls_outer.pack_propagate(False)

        viewer = tk.Frame(body, bg="#ffffff", bd=1, relief="solid", width=760)
        viewer.pack_propagate(False)

        results = tk.Frame(body, bg="#ffffff", bd=1, relief="solid", width=480)
        results.pack_propagate(False)

        body.add(controls_outer, weight=0)
        body.add(viewer, weight=1)
        body.add(results, weight=0)

        controls = self._build_scrollable_controls(controls_outer)
        self._build_controls(controls)
        self._build_viewer(viewer)
        self._build_results(results)

        tk.Label(
            self.root,
            textvariable=self.status_text,
            anchor="w",
            font=("Segoe UI", 10),
            bg="#1f2933",
            fg="#f8f5f1",
            padx=14,
            pady=8,
        ).pack(fill="x")

    def _build_scrollable_controls(self, parent: tk.Frame) -> tk.Frame:
        parent.configure(width=360)
        canvas = tk.Canvas(parent, bg="#ffffff", highlightthickness=0, width=360)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scroll_frame = tk.Frame(canvas, bg="#ffffff")

        scroll_frame.bind(
            "<Configure>",
            lambda _event: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas_window = canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        def resize_inner(event):
            canvas.itemconfigure(canvas_window, width=event.width)

        canvas.bind("<Configure>", resize_inner)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        def on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind_all("<MouseWheel>", on_mousewheel)
        return scroll_frame

    def _build_controls(self, parent: tk.Frame) -> None:
        parent.configure(width=360)
        tk.Label(parent, text="Controls", font=("Segoe UI", 16, "bold"), bg="#ffffff", fg="#1b263b").pack(
            anchor="w", padx=16, pady=(16, 8)
        )

        source_box = tk.LabelFrame(parent, text="Input Source", bg="#ffffff", fg="#1b263b", padx=10, pady=10)
        source_box.pack(fill="x", padx=14, pady=8)

        for value, text in (("image", "Image"), ("video", "Video"), ("live", "Live Camera")):
            ttk.Radiobutton(source_box, text=text, variable=self.source_type, value=value).pack(anchor="w", pady=2)

        ttk.Button(source_box, text="Choose File", command=self._choose_source).pack(fill="x", pady=(10, 6))

        mode_box = tk.LabelFrame(parent, text="Violation Modes", bg="#ffffff", fg="#1b263b", padx=10, pady=10)
        mode_box.pack(fill="x", padx=14, pady=8)
        ttk.Checkbutton(mode_box, text="No Helmet", variable=self.detect_helmet).pack(anchor="w", pady=2)
        ttk.Checkbutton(mode_box, text="Triple Riding", variable=self.detect_triple).pack(anchor="w", pady=2)
        ttk.Checkbutton(mode_box, text="Red Light Jumping", variable=self.detect_red).pack(anchor="w", pady=2)

        stop_line_box = tk.LabelFrame(parent, text="Red Light Settings", bg="#ffffff", fg="#1b263b", padx=10, pady=10)
        stop_line_box.pack(fill="x", padx=14, pady=8)
        tk.Label(stop_line_box, text="Stop line position", bg="#ffffff", fg="#334e68", font=("Segoe UI", 10)).pack(anchor="w")
        tk.Scale(
            stop_line_box,
            from_=0.35,
            to=0.9,
            resolution=0.01,
            orient="horizontal",
            variable=self.stop_line_ratio,
            bg="#ffffff",
            length=260,
        ).pack(fill="x")

        action_box = tk.LabelFrame(parent, text="Processing", bg="#ffffff", fg="#1b263b", padx=10, pady=10)
        action_box.pack(fill="x", padx=14, pady=8)
        ttk.Button(action_box, text="Start Analysis", command=self._start_processing).pack(fill="x", pady=(0, 6))
        ttk.Button(action_box, text="Stop", command=self._stop_processing).pack(fill="x", pady=(0, 6))
        ttk.Button(action_box, text="Export Log Excel", command=self._export_log).pack(fill="x")

        email_box = tk.LabelFrame(parent, text="Email Reporting", bg="#ffffff", fg="#1b263b", padx=10, pady=10)
        email_box.pack(fill="x", padx=14, pady=8)
        self._add_labeled_entry(email_box, "Authority email", self.authority_email)
        self._add_labeled_entry(email_box, "Sender email", self.sender_email)
        self._add_labeled_entry(email_box, "SMTP host", self.smtp_host)
        self._add_labeled_entry(email_box, "SMTP port", self.smtp_port)
        self._add_labeled_entry(email_box, "SMTP username", self.smtp_user)
        self._add_labeled_entry(email_box, "SMTP password", self.smtp_password, show="*")
        ttk.Button(email_box, text="Email Selected Violation", command=self._email_selected).pack(fill="x", pady=(10, 0))

        telegram_box = tk.LabelFrame(parent, text="Telegram Reporting", bg="#ffffff", fg="#1b263b", padx=10, pady=10)
        telegram_box.pack(fill="x", padx=14, pady=8)
        tk.Label(
            telegram_box,
            text=f"Configured Telegram chat ID: {TELEGRAM_CHAT_ID}",
            bg="#ffffff",
            fg="#64748b",
            wraplength=280,
            justify="left",
            font=("Segoe UI", 9),
        ).pack(anchor="w", pady=(6, 0))
        ttk.Button(telegram_box, text="Telegram Selected Violation", command=self._telegram_selected).pack(fill="x", pady=(10, 0))

        capabilities = tk.LabelFrame(parent, text="Capabilities", bg="#ffffff", fg="#1b263b", padx=10, pady=10)
        capabilities.pack(fill="both", expand=True, padx=14, pady=(8, 14))
        tk.Label(
            capabilities,
            textvariable=self.messages_text,
            justify="left",
            anchor="nw",
            bg="#ffffff",
            fg="#7b2d26",
            wraplength=290,
            font=("Segoe UI", 9),
        ).pack(fill="both", expand=True)

    def _build_viewer(self, parent: tk.Frame) -> None:
        tk.Label(parent, text="Analysis Viewer", font=("Segoe UI", 16, "bold"), bg="#ffffff", fg="#1b263b").pack(
            anchor="w", padx=16, pady=(16, 8)
        )
        self.viewer_label = tk.Label(parent, bg="#d9e2ec", width=920, height=540)
        self.viewer_label.pack(fill="both", expand=True, padx=16, pady=(0, 12))

    def _build_results(self, parent: tk.Frame) -> None:
        parent.configure(width=430)
        tk.Label(parent, text="Violations", font=("Segoe UI", 16, "bold"), bg="#ffffff", fg="#1b263b").pack(
            anchor="w", padx=16, pady=(16, 8)
        )

        columns = ("time", "type", "plate")
        self.tree = ttk.Treeview(parent, columns=columns, show="headings", height=14)
        self.tree.heading("time", text="Time")
        self.tree.heading("type", text="Violation")
        self.tree.heading("plate", text="Plate")
        self.tree.column("time", width=120, anchor="w")
        self.tree.column("type", width=170, anchor="w")
        self.tree.column("plate", width=120, anchor="w")
        self.tree.pack(fill="x", padx=14)
        self.tree.bind("<<TreeviewSelect>>", self._on_select_violation)

        details_box = tk.LabelFrame(parent, text="Details", bg="#ffffff", fg="#1b263b", padx=10, pady=10)
        details_box.pack(fill="both", expand=True, padx=14, pady=12)
        tk.Label(
            details_box,
            textvariable=self.details_text,
            justify="left",
            anchor="nw",
            bg="#ffffff",
            fg="#243b53",
            wraplength=370,
            font=("Segoe UI", 10),
        ).pack(fill="both", expand=True)
        ttk.Button(parent, text="Show Number Plate", command=self._show_selected_plate).pack(fill="x", padx=14, pady=(0, 14))

    def _add_labeled_entry(self, parent: tk.Widget, label_text: str, variable: tk.StringVar, show: str | None = None) -> None:
        tk.Label(parent, text=label_text, bg="#ffffff", fg="#334e68", font=("Segoe UI", 9)).pack(anchor="w", pady=(6, 0))
        ttk.Entry(parent, textvariable=variable, show=show or "").pack(fill="x")

    def _choose_source(self) -> None:
        source_kind = self.source_type.get()
        if source_kind == "image":
            path = filedialog.askopenfilename(filetypes=SUPPORTED_IMAGE_TYPES, initialdir=str(SAMPLE_IMAGES_DIR))
        elif source_kind == "video":
            path = filedialog.askopenfilename(filetypes=SUPPORTED_VIDEO_TYPES, initialdir=str(SAMPLE_VIDEOS_DIR))
        else:
            path = "0"

        if path:
            self.current_source_path = path
            self._set_status(f"Selected source: {path}")

    def _start_processing(self) -> None:
        self._stop_processing(clear_status=False)
        self.violations.clear()
        self.tree.delete(*self.tree.get_children())
        self.details_text.set("No violation selected.")
        self.frame_index = 0

        source_kind = self.source_type.get()
        if source_kind == "image":
            self._process_image()
            return

        if source_kind == "live":
            source = 0
        else:
            if not self.current_source_path:
                messagebox.showerror("Select a video", "Choose a video file before starting analysis.")
                return
            source = self.current_source_path

        self.capture = cv2.VideoCapture(source)
        if not self.capture.isOpened():
            messagebox.showerror("Cannot open source", f"Unable to open source: {source}")
            self.capture = None
            return

        self.video_running = True
        self._set_status("Video analysis started.")
        self._process_video_frame()

    def _process_image(self) -> None:
        if not self.current_source_path:
            messagebox.showerror("Select an image", "Choose an image file before starting analysis.")
            return

        frame = self._load_image(self.current_source_path)
        if frame is None:
            messagebox.showerror(
                "Cannot read image",
                "Failed to read the selected image.\n\n"
                f"Path: {self.current_source_path}\n\n"
                "Try a JPG, PNG, BMP, or WEBP image, and avoid incomplete downloads.",
            )
            return

        result = self.engine.analyze_frame(frame, self.frame_index, self._current_config())
        self._handle_result(result)
        self._show_frame(result.annotated_frame)
        self._set_status("Image analysis complete.")

    def _process_video_frame(self) -> None:
        if not self.video_running or self.capture is None:
            return

        ok, frame = self.capture.read()
        if not ok or frame is None:
            self._stop_processing(clear_status=False)
            self._set_status("Video processing finished.")
            return

        if self.frame_index % 4 == 0:
            result = self.engine.analyze_frame(frame, self.frame_index, self._current_config())
            self._handle_result(result)
            display_frame = result.annotated_frame
        else:
            display_frame = frame.copy()

        self._show_frame(display_frame)
        self.frame_index += 1
        self.root.after(30, self._process_video_frame)

    def _handle_result(self, result: AnalysisResult) -> None:
        self.messages_text.set("\n".join(result.messages))
        for violation in result.violations:
            self.violations.append(violation)
            self.tree.insert("", "end", values=(violation.created_at.split()[1], violation.violation_type, violation.plate_text))

    def _show_frame(self, frame) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        image.thumbnail((900, 640))
        self.current_image = ImageTk.PhotoImage(image=image)
        self.viewer_label.configure(image=self.current_image)

    def _load_image(self, image_path: str):
        path = Path(image_path)
        if not path.exists() or not path.is_file():
            return None

        try:
            data = np.fromfile(str(path), dtype=np.uint8)
            if data.size:
                decoded = cv2.imdecode(data, cv2.IMREAD_COLOR)
                if decoded is not None:
                    return decoded
        except Exception:
            pass

        try:
            with Image.open(path) as image:
                rgb_image = image.convert("RGB")
                return cv2.cvtColor(np.array(rgb_image), cv2.COLOR_RGB2BGR)
        except Exception:
            return None

    def _current_config(self) -> EngineConfig:
        return EngineConfig(
            detect_no_helmet=self.detect_helmet.get(),
            detect_triple_riding=self.detect_triple.get(),
            detect_red_light=self.detect_red.get(),
            stop_line_ratio=self.stop_line_ratio.get(),
        )

    def _on_select_violation(self, _event) -> None:
        selection = self.tree.selection()
        if not selection:
            return

        index = self.tree.index(selection[0])
        if index >= len(self.violations):
            return

        violation = self.violations[index]
        lines = [
            f"Violation: {violation.violation_type}",
            f"Time: {violation.created_at}",
            f"Plate: {violation.plate_text}",
            f"Confidence: {violation.confidence:.2f}",
            f"Evidence: {violation.evidence_path}",
            f"Plate image: {violation.plate_path or 'Not available'}",
            f"Notes: {violation.notes}",
        ]
        self.details_text.set("\n".join(lines))

    def _email_selected(self) -> None:
        selection = self.tree.selection()
        if not selection:
            messagebox.showerror("Select a violation", "Choose a violation from the list before emailing it.")
            return

        if not self.authority_email.get() or not self.sender_email.get():
            messagebox.showerror("Missing email details", "Provide the authority email and sender email first.")
            return

        index = self.tree.index(selection[0])
        violation = self.violations[index]

        try:
            port = int(self.smtp_port.get())
            body = (
                "Traffic rule violation detected.\n\n"
                f"Violation type: {violation.violation_type}\n"
                f"Time: {violation.created_at}\n"
                f"Detected plate: {violation.plate_text}\n"
                f"Notes: {violation.notes}\n"
                f"Evidence path: {violation.evidence_path}\n"
            )
            send_violation_email(
                smtp_host=self.smtp_host.get().strip(),
                smtp_port=port,
                username=self.smtp_user.get().strip(),
                password=self.smtp_password.get(),
                sender=self.sender_email.get().strip(),
                recipient=self.authority_email.get().strip(),
                subject=f"Traffic Violation Alert - {violation.violation_type}",
                body=body,
                attachment_path=violation.evidence_path,
            )
        except Exception as exc:
            messagebox.showerror("Email failed", str(exc))
            return

        messagebox.showinfo("Email sent", "The selected violation was emailed successfully.")

    def _show_selected_plate(self) -> None:
        selection = self.tree.selection()
        if not selection:
            messagebox.showerror("Select a violation", "Choose a violation from the list first.")
            return

        index = self.tree.index(selection[0])
        if index >= len(self.violations):
            return

        violation = self.violations[index]
        if not violation.plate_path:
            messagebox.showinfo("No plate image", "No extracted number plate image is available for this violation.")
            return

        frame = self._load_image(str(violation.plate_path))
        if frame is None:
            messagebox.showerror("Cannot open plate image", f"Failed to load:\n{violation.plate_path}")
            return

        self._show_frame(frame)
        self._set_status(f"Showing plate image: {violation.plate_path.name}")

    def _telegram_selected(self) -> None:
        selection = self.tree.selection()
        if not selection:
            messagebox.showerror("Select a violation", "Choose a violation from the list before sending Telegram.")
            return

        if not TELEGRAM_BOT_TOKEN.strip() or not TELEGRAM_CHAT_ID.strip():
            messagebox.showerror("Missing Telegram details", "Telegram bot token or chat ID is not configured.")
            return

        index = self.tree.index(selection[0])
        violation = self.violations[index]
        message_text = (
            "Traffic rule violation detected.\n\n"
            f"Violation type: {violation.violation_type}\n"
            f"Time: {violation.created_at}\n"
            f"Detected plate: {violation.plate_text}\n"
            f"Notes: {violation.notes}\n"
            f"Evidence path: {violation.evidence_path}\n"
        )

        try:
            payload = send_violation_telegram(
                bot_token=TELEGRAM_BOT_TOKEN,
                chat_id=TELEGRAM_CHAT_ID,
                message_text=message_text,
                photo_path=violation.evidence_path,
            )
        except Exception as exc:
            messagebox.showerror("Telegram failed", str(exc))
            return

        result = payload.get("result", {})
        messagebox.showinfo("Telegram sent", f"Violation sent successfully.\nMessage ID: {result.get('message_id', 'unknown')}")

    def _export_log(self) -> None:
        if not self.violations:
            messagebox.showinfo("No data", "No violations have been recorded yet.")
            return

        rows = []
        for violation in self.violations:
            row = asdict(violation)
            row["evidence_path"] = str(violation.evidence_path)
            row["plate_path"] = str(violation.plate_path) if violation.plate_path else ""
            rows.append(row)

        export_path = OUTPUT_ROOT / "violation_log.xlsx"
        pd.DataFrame(rows).to_excel(export_path, index=False, engine="xlsxwriter")
        messagebox.showinfo("Export complete", f"Log exported to:\n{export_path}")

    def _stop_processing(self, clear_status: bool = True) -> None:
        self.video_running = False
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        if clear_status:
            self._set_status("Processing stopped.")

    def _set_status(self, text: str) -> None:
        self.status_text.set(text)


def launch_app() -> None:
    root = tk.Tk()
    try:
        TrafficApp(root)
    except Exception as exc:
        messagebox.showerror("Startup failed", f"{exc}\n\n{traceback.format_exc()}")
        raise
    root.mainloop()
