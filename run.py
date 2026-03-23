"""
CSAI412 - Phishing Website Detection - One-Click Runner
========================================================
A tkinter GUI that runs all model training scripts sequentially,
showing progress and status in a dark-themed window.

Usage: python run.py
"""

import os
import sys
import subprocess
import threading
import platform
import tkinter as tk
from tkinter import ttk

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(PROJECT_DIR, "figures")

# Detect the Python executable
# On macOS/Linux, prefer python3; on Windows, use python
if platform.system() == "Windows":
    PYTHON = "python"
else:
    # Try python3 first, fall back to python
    try:
        result = subprocess.run(
            ["python3", "--version"],
            capture_output=True, text=True, timeout=5
        )
        PYTHON = "python3" if result.returncode == 0 else "python"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        PYTHON = "python"

# Steps to run -- (display_name, script_path_relative_to_project)
STEPS = [
    ("Load Data",               os.path.join("src", "data_loader.py")),
    ("Run EDA",                 os.path.join("src", "eda.py")),
    ("Train Logistic Regression", os.path.join("src", "models", "logistic_regression.py")),
    ("Train KNN",               os.path.join("src", "models", "knn.py")),
    ("Train SVM Linear",        os.path.join("src", "models", "svm_linear.py")),
    ("Train SVM RBF",           os.path.join("src", "models", "svm_rbf.py")),
    ("Train Decision Tree",     os.path.join("src", "models", "decision_tree.py")),
    ("Train MLP",               os.path.join("src", "models", "mlp.py")),
    ("Train K-Means + PCA",     os.path.join("src", "models", "kmeans_pca.py")),
    ("Run Comparison",          os.path.join("src", "comparison.py")),
]


# ---------------------------------------------------------------------------
# Dark theme colors
# ---------------------------------------------------------------------------
BG_COLOR = "#1e1e2e"
BG_SECONDARY = "#2a2a3d"
FG_COLOR = "#cdd6f4"
FG_DIM = "#6c7086"
ACCENT_COLOR = "#89b4fa"
SUCCESS_COLOR = "#a6e3a1"
ERROR_COLOR = "#f38ba8"
BUTTON_BG = "#45475a"
BUTTON_FG = "#cdd6f4"
BUTTON_ACTIVE_BG = "#585b70"
PROGRESS_TROUGH = "#313244"
PROGRESS_BAR = "#89b4fa"


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
class PhishingRunnerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CSAI412 -- Phishing Website Detection")
        self.root.geometry("720x560")
        self.root.resizable(True, True)
        self.root.configure(bg=BG_COLOR)

        self.running = False
        self.completed = False

        self._build_ui()

    # ---- UI construction --------------------------------------------------
    def _build_ui(self):
        # Title
        title_frame = tk.Frame(self.root, bg=BG_COLOR)
        title_frame.pack(fill=tk.X, padx=20, pady=(20, 5))

        tk.Label(
            title_frame,
            text="CSAI412 -- Phishing Website Detection",
            font=("Helvetica", 18, "bold"),
            bg=BG_COLOR,
            fg=ACCENT_COLOR,
        ).pack()

        tk.Label(
            title_frame,
            text="Machine Learning Group Project  |  One-Click Model Runner",
            font=("Helvetica", 11),
            bg=BG_COLOR,
            fg=FG_DIM,
        ).pack(pady=(2, 0))

        # Separator
        sep = tk.Frame(self.root, height=1, bg=BUTTON_BG)
        sep.pack(fill=tk.X, padx=20, pady=10)

        # Run button
        btn_frame = tk.Frame(self.root, bg=BG_COLOR)
        btn_frame.pack(fill=tk.X, padx=20, pady=(0, 10))

        self.run_button = tk.Button(
            btn_frame,
            text="Run All Models",
            font=("Helvetica", 15, "bold"),
            bg=ACCENT_COLOR,
            fg="#1e1e2e",
            activebackground="#b4d0fb",
            activeforeground="#1e1e2e",
            relief=tk.FLAT,
            padx=30,
            pady=10,
            cursor="hand2",
            command=self._on_run_clicked,
        )
        self.run_button.pack(expand=True)

        # Progress bar -- using ttk with custom style
        style = ttk.Style()
        style.theme_use("default")
        style.configure(
            "Dark.Horizontal.TProgressbar",
            troughcolor=PROGRESS_TROUGH,
            background=PROGRESS_BAR,
            thickness=22,
        )

        prog_frame = tk.Frame(self.root, bg=BG_COLOR)
        prog_frame.pack(fill=tk.X, padx=20, pady=(5, 2))

        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_bar = ttk.Progressbar(
            prog_frame,
            variable=self.progress_var,
            maximum=100,
            mode="determinate",
            style="Dark.Horizontal.TProgressbar",
        )
        self.progress_bar.pack(fill=tk.X)

        # Progress percentage label
        self.pct_label = tk.Label(
            self.root,
            text="0 %",
            font=("Helvetica", 10),
            bg=BG_COLOR,
            fg=FG_DIM,
        )
        self.pct_label.pack(pady=(0, 5))

        # Status label
        self.status_label = tk.Label(
            self.root,
            text="Ready. Click 'Run All Models' to begin.",
            font=("Helvetica", 11),
            bg=BG_COLOR,
            fg=FG_COLOR,
            wraplength=660,
            justify=tk.LEFT,
        )
        self.status_label.pack(fill=tk.X, padx=20, pady=(0, 8))

        # Log area
        log_frame = tk.Frame(self.root, bg=BG_SECONDARY, bd=1, relief=tk.SUNKEN)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 10))

        self.log_text = tk.Text(
            log_frame,
            bg=BG_SECONDARY,
            fg=FG_COLOR,
            font=("Courier", 10),
            relief=tk.FLAT,
            wrap=tk.WORD,
            state=tk.DISABLED,
            insertbackground=FG_COLOR,
            selectbackground=ACCENT_COLOR,
            selectforeground="#1e1e2e",
            padx=8,
            pady=8,
        )
        scrollbar = tk.Scrollbar(
            log_frame, command=self.log_text.yview,
            bg=BG_SECONDARY, troughcolor=BG_SECONDARY,
        )
        self.log_text.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Configure text tags for coloring
        self.log_text.tag_configure("success", foreground=SUCCESS_COLOR)
        self.log_text.tag_configure("error", foreground=ERROR_COLOR)
        self.log_text.tag_configure("info", foreground=ACCENT_COLOR)
        self.log_text.tag_configure("dim", foreground=FG_DIM)

        # Bottom buttons (hidden until done)
        self.bottom_frame = tk.Frame(self.root, bg=BG_COLOR)

        self.open_folder_button = tk.Button(
            self.bottom_frame,
            text="Open Figures Folder",
            font=("Helvetica", 12, "bold"),
            bg=SUCCESS_COLOR,
            fg="#1e1e2e",
            activebackground="#c6f3c1",
            activeforeground="#1e1e2e",
            relief=tk.FLAT,
            padx=20,
            pady=8,
            cursor="hand2",
            command=self._open_figures_folder,
        )

    # ---- Logging ----------------------------------------------------------
    def _log(self, message, tag=None):
        """Append a message to the log text widget."""
        self.log_text.configure(state=tk.NORMAL)
        if tag:
            self.log_text.insert(tk.END, message + "\n", tag)
        else:
            self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def _set_status(self, text, color=None):
        """Update the status label."""
        self.status_label.configure(text=text, fg=color or FG_COLOR)

    def _set_progress(self, value):
        """Update the progress bar and percentage label."""
        self.progress_var.set(value)
        self.pct_label.configure(text=f"{int(value)} %")

    # ---- Button handlers --------------------------------------------------
    def _on_run_clicked(self):
        if self.running:
            return
        self.running = True
        self.completed = False

        # Reset UI
        self.run_button.configure(state=tk.DISABLED, bg=BUTTON_BG)
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.delete("1.0", tk.END)
        self.log_text.configure(state=tk.DISABLED)
        self._set_progress(0)
        self.bottom_frame.pack_forget()

        # Run in background thread
        thread = threading.Thread(target=self._run_all_steps, daemon=True)
        thread.start()

    def _open_figures_folder(self):
        """Open the figures directory in the system file manager."""
        figures_path = FIGURES_DIR
        if not os.path.isdir(figures_path):
            os.makedirs(figures_path, exist_ok=True)

        system = platform.system()
        try:
            if system == "Darwin":
                subprocess.Popen(["open", figures_path])
            elif system == "Windows":
                os.startfile(figures_path)
            else:
                subprocess.Popen(["xdg-open", figures_path])
        except Exception as e:
            self.root.after(0, lambda: self._log(f"Could not open folder: {e}", "error"))

    # ---- Pipeline execution -----------------------------------------------
    def _run_all_steps(self):
        """Execute all steps sequentially in a background thread."""
        total_steps = len(STEPS)
        failed_steps = []

        self.root.after(0, lambda: self._log(
            f"Starting pipeline with {total_steps} steps...\n"
            f"Python: {PYTHON}\n"
            f"Project: {PROJECT_DIR}\n",
            "info",
        ))

        for i, (name, script) in enumerate(STEPS):
            step_num = i + 1
            script_path = os.path.join(PROJECT_DIR, script)

            # Update status
            self.root.after(0, lambda n=name, s=step_num: (
                self._set_status(f"[{s}/{total_steps}] Running: {n} ..."),
                self._log(f"[{s}/{total_steps}] {n}", "info"),
            ))

            # Check script exists
            if not os.path.isfile(script_path):
                self.root.after(0, lambda n=name, sp=script_path: (
                    self._log(f"  SKIPPED -- file not found: {sp}", "error"),
                ))
                failed_steps.append(name)
                progress = (step_num / total_steps) * 100
                self.root.after(0, lambda p=progress: self._set_progress(p))
                continue

            # Run the script
            try:
                result = subprocess.run(
                    [PYTHON, script_path],
                    capture_output=True,
                    text=True,
                    cwd=PROJECT_DIR,
                    timeout=600,  # 10 minute timeout per step
                )

                if result.returncode == 0:
                    # Count lines of output for summary
                    stdout_lines = result.stdout.strip().split("\n") if result.stdout.strip() else []
                    self.root.after(0, lambda n=name, lc=len(stdout_lines): (
                        self._log(f"  DONE ({lc} lines output)", "success"),
                    ))
                else:
                    failed_steps.append(name)
                    # Show last few lines of stderr
                    stderr_tail = result.stderr.strip().split("\n")[-5:] if result.stderr else []
                    error_msg = "\n    ".join(stderr_tail) if stderr_tail else "(no error output)"
                    self.root.after(0, lambda n=name, em=error_msg: (
                        self._log(f"  FAILED (exit code {result.returncode})", "error"),
                        self._log(f"    {em}", "dim"),
                    ))

            except subprocess.TimeoutExpired:
                failed_steps.append(name)
                self.root.after(0, lambda n=name: (
                    self._log(f"  TIMEOUT -- {n} took more than 10 minutes", "error"),
                ))
            except Exception as e:
                failed_steps.append(name)
                self.root.after(0, lambda n=name, err=str(e): (
                    self._log(f"  ERROR -- {n}: {err}", "error"),
                ))

            # Update progress
            progress = (step_num / total_steps) * 100
            self.root.after(0, lambda p=progress: self._set_progress(p))

        # Finish
        self.root.after(0, lambda: self._on_pipeline_done(failed_steps))

    def _on_pipeline_done(self, failed_steps):
        """Called on the main thread when the pipeline finishes."""
        self.running = False
        self.completed = True

        self._log("")  # blank line

        if not failed_steps:
            self._set_status(
                "Done! All 10 steps completed successfully. Open figures/ to see results.",
                SUCCESS_COLOR,
            )
            self._log("All steps completed successfully!", "success")
        else:
            failed_list = ", ".join(failed_steps)
            self._set_status(
                f"Done with {len(failed_steps)} failure(s): {failed_list}",
                ERROR_COLOR,
            )
            self._log(f"Completed with {len(failed_steps)} failure(s): {failed_list}", "error")
            self._log("Successfully completed steps still generated their outputs.", "dim")

        # Count generated figures
        if os.path.isdir(FIGURES_DIR):
            fig_files = [f for f in os.listdir(FIGURES_DIR) if f.endswith(".png")]
            self._log(f"\nFigures generated: {len(fig_files)} PNG files in figures/", "info")

        # Re-enable run button
        self.run_button.configure(state=tk.NORMAL, bg=ACCENT_COLOR, text="Run Again")

        # Show "Open Figures Folder" button
        self.open_folder_button.pack(expand=True, pady=5)
        self.bottom_frame.pack(fill=tk.X, padx=20, pady=(0, 15))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    root = tk.Tk()

    # Try to set a dark title bar on macOS
    try:
        root.tk.call("tk", "windowingsystem")
    except tk.TclError:
        pass

    app = PhishingRunnerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
