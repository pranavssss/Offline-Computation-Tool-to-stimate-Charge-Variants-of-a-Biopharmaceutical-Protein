#!/usr/bin/env python3
"""
Charge Variants Tool — Tkinter (themed, offline, no external deps)
Adds: Approximation Method = Moment Matching. Compute → runs MM and renders PMF/Table/Summary.
"""

from __future__ import annotations

import csv
import sys
import math
import tkinter as tk
from typing import Dict, List, Tuple
from tkinter import ttk, filedialog, messagebox

APP_TITLE = "Charge Variants Tool (UI Preview – Offline)"
TOL = 1e-9

DEFAULT_CHARGES_5 = [-2, -1, 0, 1, 2]

PAD_OUTER_X = 12
PAD_OUTER_Y = 10
GUTTER = 14

BTN_W_NARROW = 10
BTN_W_NORMAL = 16
BTN_W_WIDE = 22

THEMES: Dict[str, Dict[str, str]] = {
    "Dark": {
        "bg": "#0b0e11",
        "panel": "#14181d",
        "panel_alt": "#1b2128",
        "text": "#e6edf3",
        "muted": "#9aa7b3",
        "line": "#232a32",
        "accent": "#2b90d9",
        "accent_alt": "#4cb0ff",
        "warn": "#e99f3e",
        "ok": "#40c463",
        "canvas_bg": "#0f1318",
        "bar": "#4a90e2",
    },
    "Light": {
        "bg": "#f6f7f9",
        "panel": "#ffffff",
        "panel_alt": "#f0f3f7",
        "text": "#121417",
        "muted": "#5a6b7b",
        "line": "#dfe5ec",
        "accent": "#2b90d9",
        "accent_alt": "#4cb0ff",
        "warn": "#b86b00",
        "ok": "#198754",
        "canvas_bg": "#fafafa",
        "bar": "#2b90d9",
    },
}

class App(tk.Tk):
    # ----- Lifecycle ---------------------------------------------------------
    def __init__(self) -> None:
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1150x720")
        self.minsize(1000, 640)
        self.current_theme = "Dark"

        # state
        self.charge_cols: List[int] = DEFAULT_CHARGES_5[:]   # editable via Charge Count
        self.result_rows: List[Tuple[int, float]] = []       # (charge, prob)
        self.results_available: bool = False

        # build
        self._init_scaling()
        self.style = ttk.Style(self)
        self._apply_theme(self.current_theme, first_time=True)
        self._build_menu()
        self._build_bindings()
        self._build_panes()
        self._status("Ready (offline)")

    # ----- Display / Theming -------------------------------------------------
    def _init_scaling(self) -> None:
        try:
            self.call("tk", "scaling", 1.2)
        except Exception:
            pass

    def _apply_theme(self, name: str, first_time: bool = False) -> None:
        pal = THEMES.get(name, THEMES["Dark"])
        self.configure(bg=pal["bg"])

        s = self.style
        if first_time:
            _ = s.theme_use()

        s.configure("TFrame", background=pal["panel"])
        s.configure("TLabelframe", background=pal["panel"], bordercolor=pal["line"])
        s.configure("TLabelframe.Label", background=pal["panel"], foreground=pal["text"], padding=2)
        s.configure("TLabel", background=pal["panel"], foreground=pal["text"], padding=2)
        s.configure("ToolLabel.TLabel", foreground=pal["muted"], background=pal["panel"])
        s.configure("Title.TLabel", font=("Helvetica", 12, "bold"), foreground=pal["text"], background=pal["panel"])
        s.configure("Header.TLabel", font=("Helvetica", 13, "bold"), foreground=pal["text"], background=pal["panel"])

        s.configure("TButton", background=pal["accent"], foreground="#ffffff", padding=(10, 6), borderwidth=0)
        s.map("TButton",
              background=[("active", pal["accent_alt"])],
              relief=[("pressed", "sunken"), ("!pressed", "flat")])

        field_style = {
            "fieldbackground": pal["panel"],
            "foreground": pal["text"],
            "bordercolor": pal["line"],
            "lightcolor": pal["line"],
            "darkcolor": pal["line"],
            "insertcolor": pal["text"],
        }
        s.configure("TEntry", **field_style, padding=6)
        s.configure("TCombobox", **field_style, padding=4)

        s.configure("TNotebook", background=pal["panel"], borderwidth=0)
        s.configure("TNotebook.Tab", background=pal["panel"], foreground=pal["muted"], padding=(14, 8))
        s.map("TNotebook.Tab",
              background=[("selected", pal["panel_alt"])],
              foreground=[("selected", pal["text"])])

        s.configure("Treeview", background=pal["panel"], fieldbackground=pal["panel"],
                    foreground=pal["text"], bordercolor=pal["line"], rowheight=26)
        s.configure("Treeview.Heading", background=pal["panel_alt"], foreground=pal["text"], relief="flat")
        s.map("Treeview", background=[("selected", pal["accent"])], foreground=[("selected", "#ffffff")])

        s.configure("TSeparator", background=pal["line"])
        s.configure("Section.TLabelframe.Label", foreground=pal["muted"], background=pal["panel"])
        s.configure("Section.TLabelframe", background=pal["panel"], bordercolor=pal["line"])

        self.palette = pal

    def set_theme(self, name: str) -> None:
        self.current_theme = name
        self._apply_theme(name)
        self.canvas.configure(bg=self.palette["canvas_bg"])
        self._draw_chart()
        self.status_bar.configure(style="Status.TFrame")
        self.status_label.configure(style="Status.TLabel")

    # ----- Menus / Shortcuts -------------------------------------------------
    def _build_menu(self) -> None:
        m = tk.Menu(self)

        filem = tk.Menu(m, tearoff=0)
        filem.add_command(label="New", accelerator="Cmd+N", command=self.action_new)
        filem.add_command(label="Open CSV…", accelerator="Cmd+O", command=self.action_open_csv)
        filem.add_command(label="Save Input CSV…", accelerator="Cmd+S", command=self.action_save_input_csv)
        filem.add_separator()
        filem.add_command(label="Quit", accelerator="Cmd+Q", command=self.destroy)
        m.add_cascade(label="File", menu=filem)

        editm = tk.Menu(m, tearoff=0)
        editm.add_command(label="Add Row", accelerator="Cmd+Shift+A", command=self.action_add_row)
        editm.add_command(label="Delete Selected", accelerator="Delete", command=self.action_delete_selected)
        editm.add_separator()
        editm.add_command(label="Validate All", command=self.action_validate_all)
        m.add_cascade(label="Edit", menu=editm)

        viewm = tk.Menu(m, tearoff=0)
        themem = tk.Menu(viewm, tearoff=0)
        themem.add_radiobutton(label="Dark", command=lambda: self.set_theme("Dark"))
        themem.add_radiobutton(label="Light", command=lambda: self.set_theme("Light"))
        viewm.add_cascade(label="Theme", menu=themem)
        m.add_cascade(label="View", menu=viewm)

        helpm = tk.Menu(m, tearoff=0)
        helpm.add_command(label="About", command=lambda: messagebox.showinfo(
            "About", "Charge Variants Tool (UI Preview)\nFully offline.\nMoment Matching included."))
        m.add_cascade(label="Help", menu=helpm)

        self.config(menu=m)

    def _build_bindings(self) -> None:
        self.bind_all("<Command-n>", lambda e: self.action_new())
        self.bind_all("<Command-o>", lambda e: self.action_open_csv())
        self.bind_all("<Command-s>", lambda e: self.action_save_input_csv())
        self.bind_all("<Command-q>", lambda e: self.destroy())
        self.bind_all("<Command-r>", lambda e: self.action_compute())
        self.bind_all("<Delete>", lambda e: self.action_delete_selected())

    # ----- Layout (Panes) ----------------------------------------------------
    def _build_panes(self) -> None:
        paned = ttk.Panedwindow(self, orient="horizontal")
        paned.pack(fill="both", expand=True, padx=PAD_OUTER_X, pady=PAD_OUTER_Y)

        # Left pane: controls + table
        left = ttk.Frame(paned)
        paned.add(left, weight=3)
        self._build_left_controls(left)
        self._build_ptm_table(left)

        # Right pane: compute + tabs
        right = ttk.Frame(paned)
        paned.add(right, weight=2)
        self._build_compute(right)
        self._build_tabs(right)

        # Footer status
        self.status_var = tk.StringVar(value="")
        self.status_bar = ttk.Frame(self, padding=(10, 2), style="Status.TFrame")
        self.status_bar.pack(fill="x", side="bottom")
        self.style.configure("Status.TFrame", background=self.palette["panel_alt"])
        self.style.configure("Status.TLabel", background=self.palette["panel_alt"],
                             foreground=self.palette["muted"], padding=6)
        self.status_label = ttk.Label(self.status_bar, textvariable=self.status_var, style="Status.TLabel")
        self.status_label.pack(side="left")

        # seed right-side visuals
        self._populate_placeholder_results()

    # ----- Left Pane ---------------------------------------------------------
    def _build_left_controls(self, parent: ttk.Frame) -> None:
        controls_left = ttk.Frame(parent)
        controls_left.pack(fill="x", pady=(0, 8))

        # Rows
        sec_rows = ttk.Labelframe(controls_left, text="Rows", style="Section.TLabelframe", padding=(12, 10))
        sec_rows.pack(side="left", padx=(0, GUTTER))
        ttk.Button(sec_rows, text="Add Row", width=BTN_W_NORMAL, command=self.action_add_row
                   ).pack(fill="x", pady=(0, 8))
        ttk.Button(sec_rows, text="Delete", width=BTN_W_NORMAL, command=self.action_delete_selected
                   ).pack(fill="x")

        # Charge Count
        sec_cc = ttk.Labelframe(controls_left, text="Charge Count", style="Section.TLabelframe", padding=(12, 10))
        sec_cc.pack(side="left", padx=(0, GUTTER))
        self.charge_count_var = tk.StringVar(value=str(len(self.charge_cols)))
        cc_entry = ttk.Entry(sec_cc, textvariable=self.charge_count_var, width=10)
        cc_entry.pack(fill="x", pady=(0, 8))
        cc_entry.bind("<Return>", lambda _e: self.action_apply_charge_count())
        ttk.Button(sec_cc, text="Apply", width=BTN_W_NORMAL, command=self.action_apply_charge_count
                   ).pack(fill="x")

        # Files
        sec_files = ttk.Labelframe(controls_left, text="Files", style="Section.TLabelframe", padding=(12, 10))
        sec_files.pack(side="left", padx=(0, GUTTER))
        ttk.Button(sec_files, text="Open CSV…", width=BTN_W_NORMAL, command=self.action_open_csv
                   ).pack(fill="x", pady=(0, 8))
        ttk.Button(sec_files, text="Save CSV…", width=BTN_W_NORMAL, command=self.action_save_input_csv
                   ).pack(fill="x")

    def _build_ptm_table(self, parent: ttk.Frame) -> None:
        columns = ["Site", "Copies"] + [str(c) for c in self.charge_cols] + ["Sum"]
        self.tree = ttk.Treeview(parent, columns=columns, show="headings", height=18)
        for col in columns:
            self.tree.heading(col, text=col)
            anchor = "w" if col == "Site" else "center"
            width = 160 if col == "Site" else 84
            self.tree.column(col, width=width, anchor=anchor, stretch=False)
        self.tree.pack(fill="both", expand=True)

        pal = self.palette
        self.tree.tag_configure("odd", background=self._blend(pal["panel"], pal["line"], 0.15))
        self.tree.tag_configure("even", background=pal["panel"])

        self._insert_empty_row()
        self.tree.bind("<Double-1>", self._on_edit_cell)

    # ----- Right Pane --------------------------------------------------------
    def _build_compute(self, parent: ttk.Frame) -> None:
        sec = ttk.Labelframe(parent, text="Compute", style="Section.TLabelframe", padding=(10, 8, 10, 10))
        sec.pack(fill="x", pady=(0, 8))

        for c in range(6):
            sec.columnconfigure(c, weight=1 if c in (1, 3, 5) else 0)

        # Row 0 — Range + Method
        ttk.Label(sec, text="Range", style="ToolLabel.TLabel").grid(row=0, column=0, sticky="w", padx=(0, 6), pady=4)
        self.qmin_var = tk.StringVar(value="-2")
        self.qmax_var = tk.StringVar(value="2")
        ttk.Entry(sec, textvariable=self.qmin_var, width=8).grid(row=0, column=1, sticky="w", padx=(0, 6), pady=4)
        ttk.Label(sec, text="to", style="ToolLabel.TLabel").grid(row=0, column=2, sticky="w", padx=(0, 6), pady=4)
        ttk.Entry(sec, textvariable=self.qmax_var, width=8).grid(row=0, column=3, sticky="w", padx=(0, 12), pady=4)

        ttk.Label(sec, text="Approximation Method", style="ToolLabel.TLabel").grid(row=0, column=4, sticky="w", padx=(0, 6), pady=4)
        self.method_var = tk.StringVar(value="Convolution")
        ttk.Combobox(
            sec,
            textvariable=self.method_var,
            values=["Convolution", "Moment Matching", "Placeholder Bell"],
            width=18,
            state="readonly"
        ).grid(row=0, column=5, sticky="ew", pady=4)

        # Row 1 — Compute button (right-aligned)
        ttk.Button(sec, text="Compute Distribution", width=BTN_W_WIDE,
                   command=self.action_compute).grid(row=1, column=5, sticky="e", pady=(6, 2))

    def _build_tabs(self, parent: ttk.Frame) -> None:
        self.tabs = ttk.Notebook(parent)
        self.tabs.pack(fill="both", expand=True)

        self.tab_chart = ttk.Frame(self.tabs)
        self.tabs.add(self.tab_chart, text="Chart")
        self.canvas = tk.Canvas(self.tab_chart, bg=self.palette["canvas_bg"], highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=8, pady=8)
        self.canvas.bind("<Configure>", lambda _e: self._draw_chart())

        self.tab_table = ttk.Frame(self.tabs)
        self.tabs.add(self.tab_table, text="Table")
        self.table_tree = ttk.Treeview(self.tab_table, columns=["Charge", "Probability"], show="headings", height=12)
        self.table_tree.heading("Charge", text="Charge")
        self.table_tree.heading("Probability", text="Probability")
        self.table_tree.column("Charge", width=120, anchor="center")
        self.table_tree.column("Probability", width=220, anchor="e")
        self.table_tree.pack(fill="both", expand=True, padx=8, pady=8)

        self.tab_summary = ttk.Frame(self.tabs)
        self.tabs.add(self.tab_summary, text="Summary")
        self.summary_txt = tk.Text(self.tab_summary, height=10, bd=0, highlightthickness=0,
                                   bg=self.palette["panel"], fg=self.palette["text"])
        self.summary_txt.pack(fill="both", expand=True, padx=8, pady=8)

    # ======= Helpers =======
    def _blend(self, c1: str, c2: str, t: float) -> str:
        def _hex_to_rgb(h: str):
            h = h.lstrip('#'); return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
        def _rgb_to_hex(r: int, g: int, b: int) -> str:
            return f"#{r:02x}{g:02x}{b:02x}"
        r1,g1,b1 = _hex_to_rgb(c1); r2,g2,b2 = _hex_to_rgb(c2)
        r = int(r1 + (r2-r1)*t); g = int(g1 + (g2-g1)*t); b = int(b1 + (b2-b1)*t)
        return _rgb_to_hex(r,g,b)

    def _status(self, msg: str) -> None:
        self.status_var.set(msg); self.update_idletasks()

    def _insert_empty_row(self) -> None:
        vals = ["", "1"] + ["0" for _ in self.charge_cols] + ["0.000"]
        tag = "odd" if len(self.tree.get_children()) % 2 == 0 else "even"
        self.tree.insert("", "end", values=vals, tags=(tag,))

    def _collect_table_rows(self) -> List[Dict[str, float]]:
        rows: List[Dict[str, float]] = []
        for item in self.tree.get_children():
            vals = list(self.tree.item(item, "values"))
            try:
                copies = int(vals[1]) if str(vals[1]).strip() != "" else 1
            except Exception:
                copies = 1
            probs: List[float] = []
            for i, _c in enumerate(self.charge_cols, start=2):
                try:
                    p = float(vals[i])
                except Exception:
                    p = 0.0
                probs.append(max(0.0, p))
            s = sum(probs)
            vals[-1] = f"{s:.3f}"
            self.tree.item(item, values=vals)

            row = {"Site": vals[0], "Copies": copies}
            for i, _c in enumerate(self.charge_cols, start=2):
                row[str(self.charge_cols[i - 2])] = probs[i - 2]
            row["Sum"] = s
            rows.append(row)
        return rows

    def _validate_rows(self) -> Tuple[int, int]:
        rows = self._collect_table_rows()
        ok = 0
        for r in rows:
            keys = [k for k in r.keys() if k not in ("Site", "Copies", "Sum")]
            probs = [max(0.0, float(r[k])) for k in keys]
            s = sum(probs)
            if s > 0:
                probs = [p / s for p in probs]
            if abs(sum(probs) - 1.0) <= 1e-3 and r["Copies"] >= 1:
                ok += 1
        return ok, len(rows)

    # ======= Convolution Algorithm (Exact) =======
    def _convolve(self, p: Dict[int, float], q: Dict[int, float]) -> Dict[int, float]:
        """Discrete convolution of two PMFs p, q given as {charge: prob} dicts."""
        from collections import defaultdict
        r = defaultdict(float)
        for a, pa in p.items():
            for b, pb in q.items():
                r[a + b] += pa * pb
        return dict(r)

    def _power_pmf(self, p: Dict[int, float], n: int) -> Dict[int, float]:
        """n-fold self-convolution of PMF p using exponentiation-by-squaring."""
        if n == 0:
            return {0: 1.0}
        result = {0: 1.0}
        base = p.copy()
        k = n
        while k > 0:
            if k & 1:
                result = self._convolve(result, base)
            base = self._convolve(base, base)
            k //= 2
        return result

    def _convolution_exact(self, qmin: int, qmax: int) -> Tuple[List[Tuple[int, float]], float, float, float, float]:
        """
        Exact convolution-based charge distribution computation.
        Returns: (pmf list[(q,p)], mu, var, sigma, coverage_in_window)
        """
        rows = self._collect_table_rows()
        charges = list(self.charge_cols)
        
        # Start with identity (spike at 0)
        total_pmf = {0: 1.0}
        
        # Convolve each site's contribution
        for r in rows:
            # Gather probabilities in charge order
            probs = [max(0.0, float(r.get(str(c), 0.0))) for c in charges]
            s = sum(probs)
            if s <= 0.0:
                # Default to spike at 0 if present, else uniform
                probs = [1.0 if c == 0 else 0.0 for c in charges]
                s = 1.0
            probs = [p / s for p in probs]
            
            # Build base PMF for this site
            base_pmf = {c: p for c, p in zip(charges, probs)}
            
            # Raise to power (copies)
            copies = max(1, int(r.get("Copies", 1)))
            site_pmf = self._power_pmf(base_pmf, copies)
            
            # Convolve into total
            total_pmf = self._convolve(total_pmf, site_pmf)
        
        # Normalize defensively
        total_sum = sum(total_pmf.values())
        if total_sum > 0 and abs(total_sum - 1.0) > 1e-12:
            total_pmf = {k: v / total_sum for k, v in total_pmf.items()}
        
        # Compute statistics from full PMF
        all_charges = sorted(total_pmf.keys())
        mu = sum(q * total_pmf[q] for q in all_charges)
        var = sum((q - mu) ** 2 * total_pmf[q] for q in all_charges)
        sigma = math.sqrt(var) if var > 0 else 0.0
        
        # Apply windowing
        if qmin >= qmax:
            # Auto-determine window
            if len(all_charges) > 0:
                qmin = min(all_charges)
                qmax = max(all_charges)
            else:
                qmin, qmax = -2, 2
        
        # Window the PMF and compute coverage
        windowed_pmf: List[Tuple[int, float]] = []
        coverage = 0.0
        
        for q in range(qmin, qmax + 1):
            prob = total_pmf.get(q, 0.0)
            # Aggregate tails into edge bins
            if q == qmin:
                # Add all probability from below qmin
                prob += sum(total_pmf.get(c, 0.0) for c in all_charges if c < qmin)
            if q == qmax:
                # Add all probability from above qmax
                prob += sum(total_pmf.get(c, 0.0) for c in all_charges if c > qmax)
            windowed_pmf.append((q, prob))
            coverage += prob
        
        # Renormalize windowed PMF
        if coverage > 0 and abs(coverage - 1.0) > 1e-12:
            windowed_pmf = [(q, p / coverage) for q, p in windowed_pmf]
            # Note: coverage before renormalization represents actual mass in window
        
        return windowed_pmf, mu, var, sigma, coverage

    # ======= Moment Matching core =======
    def _std_normal_cdf(self, x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    def _moment_matching(self, qmin: int, qmax: int) -> Tuple[List[Tuple[int, float]], float, float, float, float]:
        """
        Returns: (pmf list[(q,p)], mu, var, sigma, coverage_in_window)
        Uses current table: 'Copies' and columns for each charge in self.charge_cols.
        """
        rows = self._collect_table_rows()
        charges = list(self.charge_cols)

        # per-row moments
        mu_total = 0.0
        var_total = 0.0
        for r in rows:
            # gather probs in charge order
            probs = [max(0.0, float(r.get(str(c), 0.0))) for c in charges]
            s = sum(probs)
            if s <= 0.0:
                # default to spike at 0 if present, else uniform zero (variance 0)
                probs = [1.0 if c == 0 else 0.0 for c in charges]
                s = 1.0
            probs = [p / s for p in probs]

            mu_row = sum(c * p for c, p in zip(charges, probs))
            var_row = sum(((c - mu_row) ** 2) * p for c, p in zip(charges, probs))

            copies = max(1, int(r.get("Copies", 1)))
            mu_total += mu_row * copies
            var_total += var_row * copies

        mu = mu_total
        var = max(0.0, var_total)
        sigma = math.sqrt(var) if var > 0 else 0.0

        # default window if user provided nonsense
        if qmin >= qmax:
            if sigma == 0.0:
                c = int(round(mu)); qmin, qmax = c - 2, c + 2
            else:
                qmin = math.floor(mu - 4 * sigma)
                qmax = math.ceil(mu + 4 * sigma)

        # discretize Normal(μ,σ²) to integer PMF
        pmf: List[Tuple[int, float]] = []
        if sigma == 0.0:
            spike = int(round(mu))
            for q in range(qmin, qmax + 1):
                pmf.append((q, 1.0 if q == spike else 0.0))
            coverage = 1.0 if qmin <= spike <= qmax else 0.0
        else:
            raw = []
            for q in range(qmin, qmax + 1):
                a = (q - 0.5 - mu) / sigma
                b = (q + 0.5 - mu) / sigma
                p = max(0.0, self._std_normal_cdf(b) - self._std_normal_cdf(a))
                raw.append(p)
            s = sum(raw) or 1.0
            pmf = [(qmin + i, raw[i] / s) for i in range(len(raw))]

            aW = (qmin - 0.5 - mu) / sigma
            bW = (qmax + 0.5 - mu) / sigma
            coverage = max(0.0, min(1.0, self._std_normal_cdf(bW) - self._std_normal_cdf(aW)))

        return pmf, mu, var, sigma, coverage

    # ======= Actions =======
    def action_new(self) -> None:
        for it in self.tree.get_children():
            self.tree.delete(it)
        self._insert_empty_row()
        self.results_available = False
        self._clear_output_tabs()
        self._populate_placeholder_results()
        self._status("New table created")

    def action_add_row(self) -> None:
        self._insert_empty_row()
        self._status("Row added")

    def action_delete_selected(self) -> None:
        sel = self.tree.selection()
        if not sel:
            messagebox.showwarning("Delete", "Select at least one row.")
            return
        for it in sel:
            self.tree.delete(it)
        self._status(f"Deleted {len(sel)} row(s)")

    def action_apply_charge_count(self) -> None:
        raw = self.charge_count_var.get().strip()
        try:
            n = int(raw)
        except Exception:
            messagebox.showwarning("Charge count", "Enter a valid integer (e.g., 5 or 11).")
            return
        n = max(1, min(101, n))
        if n % 2 == 0:
            n -= 1
            self._status(f"Charge count must be odd; using {n}.")
        k = (n - 1) // 2
        charge_list = list(range(-k, k + 1))
        self.charge_count_var.set(str(len(charge_list)))
        self.qmin_var.set(str(-k))
        self.qmax_var.set(str(+k))
        self.action_set_charges(charge_list)

    def action_set_charges(self, charge_list: List[int]) -> None:
        old_rows = [self.tree.item(it, "values") for it in self.tree.get_children()]
        self.charge_cols = list(charge_list)

        columns = ["Site", "Copies"] + [str(c) for c in self.charge_cols] + ["Sum"]
        self.tree.config(columns=columns)
        for col in columns:
            self.tree.heading(col, text=col)
            anchor = "w" if col == "Site" else "center"
            width = 160 if col == "Site" else 84
            self.tree.column(col, width=width, anchor=anchor, stretch=False)

        for it in self.tree.get_children():
            self.tree.delete(it)

        if old_rows:
            for idx, row in enumerate(old_rows):
                site, copies = row[0], row[1]
                vals = [site, copies] + ["0"] * len(self.charge_cols) + ["0.000"]
                tag = "odd" if idx % 2 == 0 else "even"
                self.tree.insert("", "end", values=vals, tags=(tag,))
        else:
            self._insert_empty_row()

        self.results_available = False
        self._clear_output_tabs()
        self._populate_placeholder_results()
        self._status(f"Charge range set to {self.charge_cols[0]} … {self.charge_cols[-1]}")

    def action_validate_all(self) -> None:
        ok, total = self._validate_rows()
        if total == 0:
            self._status("No rows to validate")
            return
        msg = f"Validation: {ok}/{total} rows OK (Sum≈1.000 and Copies≥1)"
        self._status(msg)
        messagebox.showinfo("Validate All", msg)

    def action_open_csv(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not path:
            return
        try:
            with open(path, "r", newline="") as f:
                reader = csv.DictReader(f)
                headers = [h.strip() for h in (reader.fieldnames or [])]
                if not headers or "Site" not in headers or "Copies" not in headers:
                    raise ValueError("CSV must include 'Site' and 'Copies' columns.")

                charges: List[int] = []
                for h in headers:
                    if h.lower() in ("site", "copies", "sum"):
                        continue
                    try:
                        charges.append(int(h))
                    except ValueError:
                        pass
                if not charges:
                    raise ValueError("No numeric charge columns found in CSV header.")
                charges.sort()
                self.charge_cols = charges
                self.charge_count_var.set(str(len(charges)))
                self.qmin_var.set(str(min(charges)))
                self.qmax_var.set(str(max(charges)))

                columns = ["Site", "Copies"] + [str(c) for c in self.charge_cols] + ["Sum"]
                self.tree.config(columns=columns)
                for col in columns:
                    self.tree.heading(col, text=col)
                    anchor = "w" if col == "Site" else "center"
                    width = 160 if col == "Site" else 84
                    self.tree.column(col, width=width, anchor=anchor, stretch=False)

                for it in self.tree.get_children():
                    self.tree.delete(it)

                for idx, row in enumerate(reader):
                    vals = [row.get("Site", ""), row.get("Copies", "1")]
                    for c in self.charge_cols:
                        vals.append(row.get(str(c), "0"))
                    try:
                        s = sum(float(row.get(str(c), "0") or "0") for c in self.charge_cols)
                    except Exception:
                        s = 0.0
                    vals.append(f"{s:.3f}")
                    tag = "odd" if idx % 2 == 0 else "even"
                    self.tree.insert("", "end", values=vals, tags=(tag,))

            self.results_available = False
            self._clear_output_tabs()
            self._populate_placeholder_results()
            self._status(f"Opened: {path}")
        except Exception as e:
            messagebox.showerror("Open CSV", f"Failed to open:\n{e}")

    def action_save_input_csv(self) -> None:
        path = filedialog.asksaveasfilename(defaultextension=".csv",
                                            filetypes=[("CSV files", "*.csv")])
        if not path:
            return
        try:
            cols = ["Site", "Copies"] + [str(c) for c in self.charge_cols]
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(cols)
                for item in self.tree.get_children():
                    vals = list(self.tree.item(item, "values"))
                    out = vals[:-1]  # drop Sum
                    w.writerow(out)
            self._status(f"Saved: {path}")
            messagebox.showinfo("Save Input", "Input CSV saved.")
        except Exception as e:
            messagebox.showerror("Save Input", f"Failed to save:\n{e}")

    def action_compute(self) -> None:
        # Read desired range
        try:
            qmin = int(self.qmin_var.get())
            qmax = int(self.qmax_var.get())
        except Exception:
            messagebox.showwarning("Range", "Enter valid integers for Min/Max charge.")
            return

        method = self.method_var.get().strip()

        if method == "Convolution":
            # ---- Run exact discrete convolution using the table ----
            pmf, mu, var, sigma, coverage = self._convolution_exact(qmin, qmax)

            # store + draw
            self.result_rows = pmf
            self.results_available = True

            # fill results table
            for it in self.table_tree.get_children():
                self.table_tree.delete(it)
            for q, p in self.result_rows:
                self.table_tree.insert("", "end", values=(q, f"{p:.12f}"))

            # summary text
            self.summary_txt.delete("1.0", tk.END)
            self.summary_txt.insert("1.0", "Summary (Convolution - Exact)\n\n")
            self.summary_txt.insert("end", f"- Detected charges: {self.charge_cols[0]} … {self.charge_cols[-1]}\n")
            self.summary_txt.insert("end", f"- Rows (PTM sites): {len(self.tree.get_children())}\n")
            self.summary_txt.insert("end", f"- μ = {mu:.6f}\n- σ² = {var:.6f}\n- σ = {sigma:.6f}\n")
            self.summary_txt.insert("end", f"- Window: [{qmin}, {qmax}]\n- Mass in window: {coverage*100:.2f}%\n")

            # chart
            self._draw_chart()
            self._status("Computed using Convolution (exact).")

        elif method == "Moment Matching":
            # ---- Run Gaussian moment-matching using the table ----
            pmf, mu, var, sigma, coverage = self._moment_matching(qmin, qmax)

            # store + draw
            self.result_rows = pmf
            self.results_available = True

            # fill results table
            for it in self.table_tree.get_children():
                self.table_tree.delete(it)
            for q, p in self.result_rows:
                self.table_tree.insert("", "end", values=(q, f"{p:.12f}"))

            # summary text
            self.summary_txt.delete("1.0", tk.END)
            self.summary_txt.insert("1.0", "Summary (Moment Matching)\n\n")
            self.summary_txt.insert("end", f"- Detected charges: {self.charge_cols[0]} … {self.charge_cols[-1]}\n")
            self.summary_txt.insert("end", f"- Rows (PTM sites): {len(self.tree.get_children())}\n")
            self.summary_txt.insert("end", f"- μ = {mu:.6f}\n- σ² = {var:.6f}\n- σ = {sigma:.6f}\n")
            self.summary_txt.insert("end", f"- Window: [{qmin}, {qmax}]\n- Mass in window: {coverage*100:.2f}%\n")

            # chart
            self._draw_chart()
            self._status("Computed using Moment Matching.")
        else:
            # Fallback placeholder (old behavior)
            if qmin >= qmax:
                messagebox.showwarning("Range", "Min charge must be less than Max charge.")
                return
            charges = list(range(qmin, qmax + 1))
            probs = [math.exp(-(q**2) / 4.0) for q in charges]
            s = sum(probs) or 1.0
            probs = [p / s for p in probs]

            self.result_rows = list(zip(charges, probs))
            self.results_available = True

            for it in self.table_tree.get_children():
                self.table_tree.delete(it)
            for q, p in self.result_rows:
                self.table_tree.insert("", "end", values=(q, f"{p:.12f}"))

            mean = sum(q * p for q, p in self.result_rows)
            var = sum((q - mean) ** 2 * p for q, p in self.result_rows)
            self.summary_txt.delete("1.0", tk.END)
            self.summary_txt.insert("1.0", "Summary (Placeholder)\n\n")
            self.summary_txt.insert("end", f"- Charges shown: [{qmin}, {qmax}]\n")
            self.summary_txt.insert("end", f"- Method: {method}\n")
            self.summary_txt.insert("end", f"- Mean: {mean:.6f}\n- Variance: {var:.6f}\n")

            self._draw_chart()
            self._status("Computed (placeholder).")

    # ----- Chart drawing -----------------------------------------------------
    def _draw_chart(self) -> None:
        pal = self.palette
        self.canvas.delete("all")
        self.canvas.configure(bg=pal["canvas_bg"])
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        if w < 50 or h < 50:
            return

        pad = 44
        x0, y0 = pad, h - pad
        x1, y1 = w - pad, pad
        self.canvas.create_rectangle(x0, y1, x1, y0, outline=pal["line"], width=1)

        if not self.results_available or not self.result_rows:
            msg = "Press 'Compute Distribution' to see chart."
            self.canvas.create_text((w // 2, h // 2), text=msg, fill=self.palette["muted"])
            return

        charges = [q for q, _ in self.result_rows]
        probs = [p for _, p in self.result_rows]
        qmin, qmax = min(charges), max(charges)
        pmax = max(probs) or 1.0
        span = max(1, qmax - qmin + 1)

        for frac in (0.25, 0.5, 0.75):
            y = y0 - int(frac * (y0 - y1 - 10))
            self.canvas.create_line(x0, y, x1, y, fill=self._blend(pal["line"], pal["panel"], 0.5))

        bar_gap = 6
        bar_w = max(2, (x1 - x0 - (span + 1) * bar_gap) // span)

        for i, (_q, p) in enumerate(self.result_rows):
            x_left = x0 + bar_gap + i * (bar_w + bar_gap)
            x_right = x_left + bar_w
            bh = int((p / pmax) * (y0 - y1 - 10))
            y_top = y0 - bh
            self.canvas.create_rectangle(x_left, y_top, x_right, y0, fill=pal["bar"], outline="")

        self.canvas.create_text(x0, y0 + 14, text=str(qmin), anchor="w", fill=pal["muted"])
        if qmin < 0 < qmax:
            try:
                idx0 = charges.index(0)
                x0bar = x0 + bar_gap + idx0 * (bar_w + bar_gap) + bar_w // 2
                self.canvas.create_text(x0bar, y0 + 14, text="0", fill=pal["muted"])
            except ValueError:
                pass
        self.canvas.create_text(x1, y0 + 14, text=str(qmax), anchor="e", fill=pal["muted"])
        self.canvas.create_text(x0 - 6, y1, text=f"{pmax:.2f}", anchor="e", fill=pal["muted"])

    # ----- Output & Editing --------------------------------------------------
    def _clear_output_tabs(self) -> None:
        for it in self.table_tree.get_children():
            self.table_tree.delete(it)
        self.summary_txt.delete("1.0", tk.END)
        self.results_available = False
        self.result_rows = []

    def _populate_placeholder_results(self) -> None:
        self.result_rows = [(-1, 0.2), (0, 0.6), (1, 0.2)]
        self.results_available = True
        for it in self.table_tree.get_children():
            self.table_tree.delete(it)
        for q, p in self.result_rows:
            self.table_tree.insert("", "end", values=(q, f"{p:.12f}"))
        self.summary_txt.delete("1.0", tk.END)
        self.summary_txt.insert(
            "1.0",
            "Summary (placeholder)\n\n- Charges shown: [-1, 1]\n- Method: Placeholder\n"
            "- Mean: 0.000000\n- Variance: 0.400000\n"
        )
        self._draw_chart()

    def _on_edit_cell(self, event) -> None:
        if self.tree.identify("region", event.x, event.y) != "cell":
            return
        rowid = self.tree.identify_row(event.y)
        colid = self.tree.identify_column(event.x)
        if not rowid or not colid:
            return
        col_index = int(colid.replace("#", "")) - 1
        colname = self.tree["columns"][col_index]
        if colname == "Sum":
            return

        x, y, w, h = self.tree.bbox(rowid, colid)
        value = self.tree.set(rowid, colname)

        entry = tk.Entry(self.tree, bd=0, highlightthickness=1,
                         highlightbackground=self.palette["line"],
                         highlightcolor=self.palette["accent"],
                         bg=self.palette["panel"], fg=self.palette["text"])
        entry.place(x=x, y=y, width=w, height=h)
        entry.insert(0, value)
        entry.focus()

        def commit(_=None):
            new_val = entry.get().strip()
            self.tree.set(rowid, colname, new_val)
            entry.destroy()
            self._collect_table_rows()

        entry.bind("<Return>", commit)
        entry.bind("<Escape>", lambda _e: entry.destroy())
        entry.bind("<FocusOut>", lambda _e: entry.destroy())


# ========= Entrypoint =========
if __name__ == "__main__":
    try:
        App().mainloop()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
