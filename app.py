import os
import sys
import threading

if sys.stdout is None or sys.stderr is None:
    _null = open(os.devnull, "w")
    sys.stdout = sys.stdout or _null
    sys.stderr = sys.stderr or _null

import customtkinter as ctk

from qa import (
    LLM_MODEL,
    load_model,
    load_collection,
    answer,
    answer_plain,
)

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("League of Legends — RAG QA")
        self.geometry("1100x720")
        self.model = None
        self.collection = None

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        top = ctk.CTkFrame(self)
        top.grid(row=0, column=0, padx=16, pady=(16, 8), sticky="ew")
        top.grid_columnconfigure(0, weight=1)

        self.entry = ctk.CTkEntry(
            top, placeholder_text="what does Aatrox's ultimate do?", height=44
        )
        self.entry.grid(row=0, column=0, padx=(0, 8), sticky="ew")
        self.entry.bind("<Return>", lambda event: self.on_send())

        self.send_button = ctk.CTkButton(
            top, text="Send question", width=150, height=44, command=self.on_send
        )
        self.send_button.grid(row=0, column=1)

        self.status = ctk.CTkLabel(
            self, text="Loading model and vector store...", anchor="w"
        )
        self.status.grid(row=1, column=0, padx=16, pady=(0, 8), sticky="ew")

        panes = ctk.CTkFrame(self, fg_color="transparent")
        panes.grid(row=2, column=0, padx=16, pady=(0, 16), sticky="nsew")
        panes.grid_columnconfigure(0, weight=1, uniform="cols")
        panes.grid_columnconfigure(1, weight=1, uniform="cols")
        panes.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(
            panes, text="RAG answer", font=ctk.CTkFont(size=16, weight="bold")
        ).grid(row=0, column=0, padx=(0, 8), pady=4, sticky="w")
        ctk.CTkLabel(
            panes,
            text=f"Plain model — no RAG ({LLM_MODEL})",
            font=ctk.CTkFont(size=16, weight="bold"),
        ).grid(row=0, column=1, padx=(8, 0), pady=4, sticky="w")

        self.rag_box = ctk.CTkTextbox(panes, wrap="word", font=ctk.CTkFont(size=14))
        self.rag_box.grid(row=1, column=0, padx=(0, 8), sticky="nsew")

        self.plain_box = ctk.CTkTextbox(panes, wrap="word", font=ctk.CTkFont(size=14))
        self.plain_box.grid(row=1, column=1, padx=(8, 0), sticky="nsew")

        self.set_busy(True)
        threading.Thread(target=self.load_resources, daemon=True).start()

    def load_resources(self):
        model = load_model()
        collection = load_collection()
        self.model = model
        self.collection = collection
        count = collection.count()
        self.after(0, lambda: self.on_ready(count))

    def on_ready(self, count):
        self.status.configure(text=f"Ready — {count} chunks loaded. Ask a question.")
        self.set_busy(False)

    def set_busy(self, busy):
        state = "disabled" if busy else "normal"
        self.send_button.configure(state=state)
        self.entry.configure(state=state)

    def write_box(self, box, text):
        box.configure(state="normal")
        box.delete("1.0", "end")
        box.insert("1.0", text)

    def on_send(self):
        question = self.entry.get().strip()
        if not question or self.collection is None:
            return
        self.set_busy(True)
        self.status.configure(text="Thinking...")
        self.write_box(self.rag_box, "")
        self.write_box(self.plain_box, "")
        threading.Thread(
            target=self.run_query, args=(question,), daemon=True
        ).start()

    def run_query(self, question):
        results = {}

        def run_rag():
            try:
                reply, sources = answer(self.collection, self.model, question)
                if sources:
                    reply = reply + "\n\nSources:\n" + "\n".join(
                        f"  - {source}" for source in sources
                    )
                results["rag"] = reply
            except Exception as error:
                results["rag"] = f"Error: {error}"

        def run_plain():
            try:
                results["plain"] = answer_plain(question)
            except Exception as error:
                results["plain"] = f"Error: {error}"

        rag_thread = threading.Thread(target=run_rag)
        plain_thread = threading.Thread(target=run_plain)
        rag_thread.start()
        plain_thread.start()
        rag_thread.join()
        plain_thread.join()
        self.after(0, lambda: self.show_results(results["rag"], results["plain"]))

    def show_results(self, reply, plain):
        self.write_box(self.rag_box, reply)
        self.write_box(self.plain_box, plain)
        self.status.configure(text="Ready — ask a question.")
        self.set_busy(False)


if __name__ == "__main__":
    App().mainloop()
