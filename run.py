"""
CSAI412 — Phishing Website Detection
One-click runner with Jupyter Notebook and Google Colab options.
"""

import subprocess
import sys
import os
import threading
import webbrowser
import tkinter as tk
from tkinter import messagebox

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
NOTEBOOK = os.path.join(PROJECT_DIR, "notebook.ipynb")
PYTHON = "python3" if sys.platform != "win32" else "python"

# GitHub repo for Colab link
GITHUB_REPO = "nazimhnc/csai412-phishing-ml"
COLAB_URL = f"https://colab.research.google.com/github/{GITHUB_REPO}/blob/main/notebook.ipynb"


def ensure_jupyter():
    try:
        subprocess.run([PYTHON, "-m", "jupyter", "--version"],
                       capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        try:
            update_status("Installing Jupyter...")
            subprocess.run([PYTHON, "-m", "pip", "install", "jupyter", "notebook"],
                           capture_output=True, check=True)
            return True
        except Exception as e:
            messagebox.showerror("Error", f"Failed to install Jupyter:\n{e}\n\nTry: pip install jupyter")
            return False


def update_status(text):
    root.after(0, lambda: status_label.config(text=text))


def run_jupyter():
    jupyter_btn.config(state="disabled")
    update_status("Checking Jupyter...")
    root.update()

    if not ensure_jupyter():
        jupyter_btn.config(state="normal")
        return

    if not os.path.exists(NOTEBOOK):
        messagebox.showerror("Error", f"Notebook not found:\n{NOTEBOOK}")
        jupyter_btn.config(state="normal")
        return

    def execute_and_open():
        update_status("Executing all cells (this takes a few minutes)...")

        try:
            subprocess.run(
                [PYTHON, "-m", "jupyter", "nbconvert",
                 "--to", "notebook", "--execute", "--inplace",
                 "--ExecutePreprocessor.timeout=600",
                 NOTEBOOK],
                capture_output=True, text=True, cwd=PROJECT_DIR,
                timeout=900
            )
        except Exception:
            pass

        update_status("Opening notebook in browser...")
        subprocess.Popen(
            [PYTHON, "-m", "jupyter", "notebook", NOTEBOOK],
            cwd=PROJECT_DIR
        )
        update_status("Notebook is open in your browser!")
        root.after(0, lambda: jupyter_btn.config(state="normal"))

    threading.Thread(target=execute_and_open, daemon=True).start()


def run_colab():
    update_status("Opening Google Colab in browser...")
    webbrowser.open(COLAB_URL)
    update_status("Colab opened! Click 'Runtime > Run all' in Colab.")


# ─── GUI ───

root = tk.Tk()
root.title("CSAI412 — Phishing Website Detection")
root.geometry("520x400")
root.resizable(False, False)

BG = "#1e1e2e"
FG = "#cdd6f4"
ACCENT = "#89b4fa"
COLAB_COLOR = "#f9ab00"
BTN_FG = "#1e1e2e"

root.configure(bg=BG)

# Title
tk.Label(root, text="CSAI412", font=("Helvetica", 36, "bold"),
         bg=BG, fg=ACCENT).pack(pady=(25, 0))

tk.Label(root, text="Phishing Website Detection",
         font=("Helvetica", 15), bg=BG, fg=FG).pack(pady=(0, 2))

tk.Label(root, text="Machine Learning Group Project",
         font=("Helvetica", 11), bg=BG, fg="#6c7086").pack(pady=(0, 25))

# Buttons frame
btn_frame = tk.Frame(root, bg=BG)
btn_frame.pack(pady=(0, 10))

# Jupyter button
jupyter_btn = tk.Button(btn_frame, text="Run in Jupyter",
                        font=("Helvetica", 14, "bold"),
                        bg=ACCENT, fg=BTN_FG, activebackground="#b4befe",
                        relief="flat", padx=25, pady=10, cursor="hand2",
                        command=run_jupyter)
jupyter_btn.grid(row=0, column=0, padx=8)

# Colab button
colab_btn = tk.Button(btn_frame, text="Open in Colab",
                      font=("Helvetica", 14, "bold"),
                      bg=COLAB_COLOR, fg=BTN_FG, activebackground="#fdd663",
                      relief="flat", padx=25, pady=10, cursor="hand2",
                      command=run_colab)
colab_btn.grid(row=0, column=1, padx=8)

# Status
status_label = tk.Label(root, text="Choose how to run the project",
                        font=("Helvetica", 11), bg=BG, fg="#a6adc8", wraplength=480)
status_label.pack(pady=(15, 8))

# Info
tk.Label(root, text="Jupyter: Runs locally — executes all cells, opens in browser\n"
                     "Colab: Runs in Google's cloud — needs internet, no install",
         font=("Helvetica", 10), bg=BG, fg="#585b70", wraplength=480,
         justify="center").pack(pady=(0, 10))

# Footer
tk.Label(root, text="Nazim Ahmed  ·  Danniyaal Ahmed  ·  Mohamed Talha",
         font=("Helvetica", 9), bg=BG, fg="#45475a").pack(side="bottom", pady=10)

root.mainloop()
