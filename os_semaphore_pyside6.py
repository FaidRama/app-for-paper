# os_semaphore_sim_pyside6_batchtab.py
import sys
import os
import json
import random
import time
import math
import simpy
import numpy as np
import pandas as pd
from collections import deque

from PySide6 import QtCore, QtWidgets, QtGui
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# =========================
# Simulation primitives (same as yours)
# =========================
class TraceLog:
    def __init__(self):
        self.events = []
    def log(self, env, event_type, pid, info=""):
        self.events.append({"time": float(env.now), "event": event_type, "pid": pid, "info": info})

class FIFOSemaphore:
    def __init__(self, env, initial=1, logger=None, ctx_switch=0.0):
        self.env = env
        self.count = initial
        self.queue = deque()
        self.logger = logger
        self.op_count = 0
        self.wait_count = 0
        self.ctx_switch = ctx_switch
        self.ql_samples = []

    def _sample(self):
        self.ql_samples.append((float(self.env.now), len(self.queue)))

    def acquire(self, pid):
        self._sample()
        if self.count > 0 and len(self.queue) == 0:
            self.count -= 1
            self.op_count += 1
            if self.logger: self.logger.log(self.env, "sem-acquire", pid, "immediate")
            return None
        else:
            ev = self.env.event()
            self.queue.append((ev, pid, float(self.env.now)))
            self.wait_count += 1
            if self.logger: self.logger.log(self.env, "sem-block", pid, f"pos={len(self.queue)}")
            return ev

    def release(self, pid):
        self.op_count += 1
        if self.queue:
            ev, pid_next, tarr = self.queue.popleft()
            if self.ctx_switch and self.ctx_switch > 0:
                def delayed(env, ev):
                    yield env.timeout(self.ctx_switch)
                    ev.succeed()
                self.env.process(delayed(self.env, ev))
            else:
                ev.succeed()
            if self.logger: self.logger.log(self.env, "sem-release-pass", pid, f"to {pid_next}")
        else:
            self.count += 1
            if self.logger: self.logger.log(self.env, "sem-release-inc", pid, f"count={self.count}")

class CPU:
    def __init__(self, env, logger, preemptive=False, cores=1):
        self.env = env
        self.logger = logger
        self.cores = cores
        self.preemptive = preemptive
        if preemptive:
            self.resource = simpy.PreemptiveResource(env, capacity=cores)
        else:
            self.resource = simpy.Resource(env, capacity=cores)
        self.available = list(range(cores))
        self.core_last = {i: 0.0 for i in range(cores)}
        self.core_busy = {i: 0.0 for i in range(cores)}
        self.allocated = {}

    def allocate(self, pid):
        if not self.available:
            core = None
        else:
            core = self.available.pop(0)
            self.allocated[core] = pid
            self.core_last[core] = float(self.env.now)
            if self.logger: self.logger.log(self.env, "cpu-alloc", pid, f"core={core}")
        return core

    def release(self, core, pid):
        if core is None:
            return
        used = float(self.env.now - self.core_last.get(core, self.env.now))
        if used < 0: used = 0.0
        self.core_busy[core] += used
        if core in self.allocated:
            del self.allocated[core]
        if core not in self.available:
            self.available.append(core)
            self.available.sort()
        if self.logger: self.logger.log(self.env, "cpu-release", pid, f"core={core},used={used:.6f}")

    def total_busy(self):
        return sum(self.core_busy.values())

    def total_idle(self, total_time):
        return max(0.0, self.cores * total_time - self.total_busy())

class Mutex:
    def __init__(self, env, logger=None, ctx_switch=0.0):
        self.env = env
        self.busy = False
        self.owner = None
        self.queue = deque()
        self.logger = logger
        self.ctx_switch = ctx_switch
        self.op_count = 0
        self.wait_count = 0
        self.ql_samples = []
    
    def _sample(self):
        self.ql_samples.append((float(self.env.now), len(self.queue)))
    
    def acquire(self, pid):
        self._sample()
        if not self.busy and len(self.queue) == 0:
            self.busy = True
            self.owner = pid
            self.op_count += 1
            if self.logger: self.logger.log(self.env, "mtx-acquire", pid, "immediate")
            return None
        
        #must wait
        ev = self.env.event()
        self.queue.append((ev, pid))
        self.wait_count += 1
        if self.logger: self.logger.log(self.env, "mtx-block", pid, f"pos=={len(self.queue)}")
        return ev
    
    def release(self, pid):
        #optional safety: only owner release
        if pid != self.owner:
            if self.logger: self.logger.log(self.env, "mtx-invalid-release", pid, "not owner")
            return
        self.op_count += 1

        if self.queue:
            ev, nxt = self.queue.popleft()
            self.owner = nxt
            if self.ctx_switch > 0:
                def delayed(env, ev):
                    yield env.timeout(self.ctx_switch)
                    ev.succeed()
                self.env.process(delayed(self.owner, ev))
            else:
                ev.succeed()
            if self.logger: self.logger.log(self.env, "mtx-release-pass", pid, f"{nxt}")
        else:
            self.busy = False
            self.owner = None
            if self.logger: self.logger.log(self.env, "mtx-release-free", pid)

def safe_timeout(env, d):
    return env.timeout(d)

# -----------------------
# Single-run simulation (same signature as before)
# -----------------------
def run_simulation_single(n_process=100, mean_arrival=0.5, mean_cs=3.0, semaphore_count=1,
                          scheduler="NONE", quantum=1.0, cpu_cores=1, max_cs_per_proc=1,
                          starv_mult=3.0, ctx_switch=0.0, seed=None, use_mutex=False):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    env = simpy.Environment()
    logger = TraceLog()
    if use_mutex:
        sem = Mutex(env, logger=logger, ctx_switch=ctx_switch)
    else:
        sem = FIFOSemaphore(env, initial=semaphore_count, logger=logger, ctx_switch=ctx_switch)
    cpu = CPU(env, logger, preemptive=(scheduler=="PRIO"), cores=cpu_cores)

    wait_times = []
    timeline = []   # list of (pid, start, end)
    usage = []
    completed = 0
    interrupt_count = 0

    def process_behavior(pid, runs, priority):
        nonlocal interrupt_count, completed
        for _ in range(runs):

            try:
            # -----------------------------
            # 1) Acquire semaphore
            # -----------------------------
                start_wait = float(env.now)
                acq = sem.acquire(pid)
                if acq is not None:
                    try:
                        yield acq
                    except simpy.Interrupt:
                        interrupt_count += 1
                    # proses kena preempt / dibatalkan → skip iterasi ini
                        continue

                wait = float(env.now - start_wait)
                wait_times.append(wait)
                logger.log(env, "enter_cs", pid, f"wait={wait:.6f}")

            # -----------------------------
            # 2) Critical Section length
            # -----------------------------
                cs_len = random.expovariate(1.0 / mean_cs)
                cs_start = float(env.now)

            # =========================================================
            # 3) CPU scheduler — semua yield dibungkus interrupt-safe
            # =========================================================

            # --- NO SCHEDULER
                if scheduler == "NONE":
                    try:
                        req = cpu.resource.request()
                        yield req
                        core = cpu.allocate(pid)
                        yield safe_timeout(env, cs_len)
                    except simpy.Interrupt:
                    # lepas CPU bila ada interrupt
                        interrupt_count += 1
                        try: cpu.release(core, pid)
                        except: pass
                        try: cpu.resource.release(req)
                        except: pass
                        sem.release(pid)
                        continue
                    cpu.release(core, pid)
                    cpu.resource.release(req)

            # --- ROUND ROBIN
                elif scheduler == "RR":
                    remaining = cs_len
                    q = quantum
                    while remaining > 0:
                        try:
                            req = cpu.resource.request()
                            yield req
                            core = cpu.allocate(pid)
                            run = min(remaining, q)
                            yield safe_timeout(env, run)
                            remaining -= run
                        except simpy.Interrupt:
                            interrupt_count += 1
                            try: cpu.release(core, pid)
                            except: pass
                            try: cpu.resource.release(req)
                            except: pass
                            sem.release(pid)
                            continue
                        cpu.release(core, pid)
                        cpu.resource.release(req)

            # --- PRIORITY SCHEDULER
                elif scheduler == "PRIO":
                    try:
                        req = cpu.resource.request(priority=priority)
                        yield req
                        core = cpu.allocate(pid)
                        yield safe_timeout(env, cs_len)
                    except simpy.Interrupt:
                        interrupt_count += 1
                        try: cpu.release(core, pid)
                        except: pass
                        try: cpu.resource.release(req)
                        except: pass
                        sem.release(pid)
                        continue
                    cpu.release(core, pid)
                    cpu.resource.release(req)

            # --- MLFQ
                elif scheduler == "MLFQ":
                    remaining = cs_len
                    quanta = [quantum, quantum * 2, quantum * 4]
                    level = 0
                    while remaining > 0:
                        try:
                            req = cpu.resource.request()
                            yield req
                            core = cpu.allocate(pid)
                            q = quanta[level]
                            run = min(remaining, q)
                            yield safe_timeout(env, run)
                            remaining -= run
                            if remaining > 0 and level < 2:
                                level += 1
                        except simpy.Interrupt:
                            interrupt_count += 1
                            try: cpu.release(core, pid)
                            except: pass
                            try: cpu.resource.release(req)
                            except: pass
                            sem.release(pid)
                            continue
                        cpu.release(core, pid)
                        cpu.resource.release(req)

            # -----------------------------
            # 4) Finish CS
            # -----------------------------
                cs_end = float(env.now)
                usage.append(cs_end - cs_start)
                timeline.append((pid, cs_start, cs_end))
                logger.log(env, "exit_cs", pid, f"dur={(cs_end - cs_start):.6f}")

                sem.release(pid)
                completed += 1

            # sedikit jeda agar tidak starving
                yield env.timeout(0.001)

            except simpy.Interrupt:
                interrupt_count += 1
            # jika ada interrupt dari luar loop
                continue


    def generator():
        for i in range(n_process):
            ia = random.expovariate(1.0 / mean_arrival)
            yield env.timeout(ia)
            pid = f"P{i+1}"
            pr = random.randint(0, 10)
            env.process(process_behavior(pid, max_cs_per_proc, pr))

    env.process(generator())
    env.run()

    total_time = float(env.now)
    avg_wait = float(np.mean(wait_times)) if wait_times else 0.0
    std_wait = float(np.std(wait_times)) if wait_times else 0.0
    throughput = completed / total_time if total_time > 0 else 0.0
    total_use = cpu.total_busy()
    cpu_util = total_use / (total_time * cpu_cores) if total_time > 0 else 0.0
    cpu_idle = cpu.total_idle(total_time)
    starv_thr = avg_wait * starv_mult if avg_wait > 0 else 10.0
    starved = sum(1 for w in wait_times if w > starv_thr)
    starvation_rate = starved / len(wait_times) if wait_times else 0.0
    qlen_samples = sem.ql_samples
    avg_q = float(np.mean([q for _, q in qlen_samples])) if qlen_samples else 0.0

    result = {
        "avg_wait": avg_wait,
        "std_wait": std_wait,
        "throughput": throughput,
        "cpu_util": min(cpu_util, 1.0),
        "cpu_idle": cpu_idle,
        "total_time": total_time,
        "starvation_rate": starvation_rate,
        "starved_count": int(starved),
        "wait_times": wait_times,
        "timeline": timeline,
        "trace": logger.events,
        "avg_queue_len": avg_q,
        "wait_count": sem.wait_count,
        "semaphore_ops": sem.op_count,
        "interrupt_count": interrupt_count
    }
    return result

# =========================
# GUI: Canvas + App
# =========================
class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=6, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi, tight_layout=True)
        super().__init__(fig)
        self.axes = fig.subplots()

# -------------------------
# Batch worker (QThread) - runs batch in background
# -------------------------
class BatchWorker(QtCore.QObject):
    progress = QtCore.Signal(int)
    finished = QtCore.Signal(object)  # will emit DataFrame

    def __init__(self, scenarios, schedulers, base_cfg, repeats, seed_base=42):
        super().__init__()
        self.scenarios = scenarios
        self.schedulers = schedulers
        self.base_cfg = base_cfg
        self.repeats = repeats
        self.seed_base = seed_base
        self._is_running = True

    @QtCore.Slot()
    def run(self):
        records = []
        total_tasks = len(self.scenarios) * len(self.schedulers) * self.repeats
        done = 0
        for sname, scfg in self.scenarios.items():
            for sched in self.schedulers:
                for rep in range(self.repeats):
                    if not self._is_running:
                        break
                    cfg = dict(self.base_cfg)
                    cfg.update(scfg)
                    # run single sim
                    seed = self.seed_base + done
                    res = run_simulation_single(
                        n_process=cfg["n_process"],
                        mean_arrival=cfg["mean_arrival"],
                        mean_cs=cfg["mean_cs"],
                        semaphore_count=cfg["semaphore_count"],
                        scheduler=sched,
                        quantum=cfg.get("quantum", 1.0),
                        cpu_cores=cfg.get("cpu_cores", 1),
                        max_cs_per_proc=cfg.get("max_cs_per_process", 1),
                        starv_mult=cfg.get("starv_mult", 3.0),
                        ctx_switch=cfg.get("ctx_switch", 0.0),
                        seed=seed
                    )
                    rec = {
                        "scenario": sname,
                        "scheduler": sched,
                        "repeat": rep + 1,
                        "n_process": cfg["n_process"],
                        "mean_arrival": cfg["mean_arrival"],
                        "mean_cs": cfg["mean_cs"],
                        "semaphore_count": cfg["semaphore_count"],
                        "avg_wait": res["avg_wait"],
                        "std_wait": res["std_wait"],
                        "throughput": res["throughput"],
                        "cpu_util": res["cpu_util"],
                        "cpu_idle": res["cpu_idle"],
                        "starvation_rate": res["starvation_rate"],
                        "starved_count": res["starved_count"],
                        "total_time": res["total_time"],
                        "avg_queue_len": res.get("avg_queue_len", None),
                        "wait_count": res.get("wait_count", None),
                        "semaphore_ops": res.get("semaphore_ops", None),
                        "wait_times": json.dumps(res["wait_times"]),
                        "timeline": json.dumps(res["timeline"]),
                        "trace": json.dumps(res["trace"])
                    }
                    records.append(rec)
                    done += 1
                    pct = int(done / total_tasks * 100)
                    self.progress.emit(pct)
        df = pd.DataFrame.from_records(records)
        self.finished.emit(df)

    def stop(self):
        self._is_running = False

# -------------------------
# Main Window
# -------------------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OS Semaphore Simulator (PySide6) - with Batch Tab")
        self.resize(1200, 760)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        hl = QtWidgets.QHBoxLayout(central)

        # Left controls
        left = QtWidgets.QFrame()
        left.setFixedWidth(320)
        left_l = QtWidgets.QVBoxLayout(left)
        hl.addWidget(left)

        # parameter widgets (as before)
        self.spin_n = QtWidgets.QSpinBox(); self.spin_n.setRange(1,2000); self.spin_n.setValue(200)
        self.spin_arr = QtWidgets.QDoubleSpinBox(); self.spin_arr.setRange(0.01, 10.0); self.spin_arr.setSingleStep(0.01); self.spin_arr.setValue(0.5)
        self.spin_cs = QtWidgets.QDoubleSpinBox(); self.spin_cs.setRange(0.01, 50.0); self.spin_cs.setSingleStep(0.1); self.spin_cs.setValue(3.0)
        self.spin_sem = QtWidgets.QSpinBox(); self.spin_sem.setRange(1,16); self.spin_sem.setValue(1)
        self.combo_sched = QtWidgets.QComboBox(); self.combo_sched.addItems(["NONE","RR","PRIO","MLFQ"])
        self.spin_quant = QtWidgets.QDoubleSpinBox(); self.spin_quant.setRange(0.1, 10.0); self.spin_quant.setSingleStep(0.1); self.spin_quant.setValue(1.0)
        self.spin_cores = QtWidgets.QSpinBox(); self.spin_cores.setRange(1,8); self.spin_cores.setValue(1)
        self.spin_runs = QtWidgets.QSpinBox(); self.spin_runs.setRange(1,10); self.spin_runs.setValue(1)
        self.spin_starv = QtWidgets.QDoubleSpinBox(); self.spin_starv.setRange(1.0, 20.0); self.spin_starv.setSingleStep(0.5); self.spin_starv.setValue(3.0)
        self.spin_ctx = QtWidgets.QDoubleSpinBox(); self.spin_ctx.setRange(0.0, 0.01); self.spin_ctx.setSingleStep(0.0005); self.spin_ctx.setValue(0.0)

        def add_row(label, widget):
            lb = QtWidgets.QLabel(label)
            hl2 = QtWidgets.QHBoxLayout()
            hl2.addWidget(lb)
            hl2.addWidget(widget)
            left_l.addLayout(hl2)

        add_row("Processes", self.spin_n)
        add_row("Mean inter-arrival", self.spin_arr)
        add_row("Mean CS length", self.spin_cs)
        add_row("Semaphore count", self.spin_sem)
        add_row("Scheduler", self.combo_sched)
        add_row("Quantum", self.spin_quant)
        add_row("CPU cores", self.spin_cores)
        add_row("CS runs/process", self.spin_runs)
        add_row("Starvation multiplier", self.spin_starv)
        add_row("Semaphore ctx-switch", self.spin_ctx)

        self.btn_run = QtWidgets.QPushButton("Run Simulation")
        left_l.addWidget(self.btn_run)
        self.btn_run.clicked.connect(self.on_run)

        left_l.addSpacing(8)
        self.metrics_box = QtWidgets.QGroupBox("Metrics")
        ml = QtWidgets.QVBoxLayout(self.metrics_box)
        self.lbl_metrics = QtWidgets.QLabel("No run yet.")
        self.lbl_metrics.setWordWrap(True)
        ml.addWidget(self.lbl_metrics)
        left_l.addWidget(self.metrics_box)
        left_l.addStretch()

        # Right: tabbed visual + batch tab
        self.tabs = QtWidgets.QTabWidget()
        hl.addWidget(self.tabs, 1)

        self.output_box = QtWidgets.QTextEdit()
        self.output_box.setReadOnly(True)
        left_l.addWidget(self.output_box)

        self.chk_mutex = QtWidgets.QCheckBox("Use Mutex")
        left_l.addWidget(self.chk_mutex)
        self.chk_compare = QtWidgets.QCheckBox("Compare: Mutex vs Semaphore")
        left_l.addWidget(self.chk_compare)
        left_l.addStretch()

        # Tab 1: Gantt
        self.tab_gantt = QtWidgets.QWidget(); v1 = QtWidgets.QVBoxLayout(self.tab_gantt)
        self.canvas_gantt = MplCanvas(self, width=8, height=5, dpi=100)
        v1.addWidget(self.canvas_gantt)
        self.tabs.addTab(self.tab_gantt, "Gantt (CS segments)")

        # Tab 2: Wait Histogram
        self.tab_hist = QtWidgets.QWidget(); v2 = QtWidgets.QVBoxLayout(self.tab_hist)
        self.canvas_hist = MplCanvas(self, width=8, height=4, dpi=100)
        v2.addWidget(self.canvas_hist)
        self.tabs.addTab(self.tab_hist, "Wait Time Histogram")

        # Tab 3: CPU Util
        self.tab_cpu = QtWidgets.QWidget(); v3 = QtWidgets.QVBoxLayout(self.tab_cpu)
        self.canvas_cpu = MplCanvas(self, width=8, height=3, dpi=100)
        v3.addWidget(self.canvas_cpu)
        self.tabs.addTab(self.tab_cpu, "CPU Utilization")

        # Tab 4: Batch Experiments (NEW)
        self.tab_batch = QtWidgets.QWidget(); vb = QtWidgets.QVBoxLayout(self.tab_batch)
        # Batch controls
        hb_controls = QtWidgets.QHBoxLayout()
        self.btn_run_batch = QtWidgets.QPushButton("Run Batch (Low/Med/High × RR/PRIO/MLFQ)")
        self.btn_save_batch = QtWidgets.QPushButton("Save Batch CSV")
        self.btn_save_batch.setEnabled(False)
        hb_controls.addWidget(self.btn_run_batch)
        hb_controls.addWidget(self.btn_save_batch)
        vb.addLayout(hb_controls)
        # progress bar
        self.batch_progress = QtWidgets.QProgressBar()
        vb.addWidget(self.batch_progress)
        # results preview (text)
        self.batch_txt = QtWidgets.QTextEdit()
        self.batch_txt.setReadOnly(True)
        vb.addWidget(self.batch_txt)
        self.tabs.addTab(self.tab_batch, "Batch Experiments")



        # signals
        self.btn_run_batch.clicked.connect(self.on_run_batch)
        self.btn_save_batch.clicked.connect(self.on_save_batch)

        # internal storage
        self.last_result = None
        self.last_batch_df = None
        self.worker_thread = None
        self.batch_worker = None

        # status bar
        self.status = QtWidgets.QStatusBar()
        self.setStatusBar(self.status)

    # -------------------------
    # Single-run handler
    # -------------------------
    @QtCore.Slot()
    def on_run(self):
        self.status.showMessage("Running simulation...")
        QtWidgets.QApplication.processEvents()

        n = int(self.spin_n.value())
        arr = float(self.spin_arr.value())
        cs = float(self.spin_cs.value())
        semc = int(self.spin_sem.value())
        sched = str(self.combo_sched.currentText())
        quant = float(self.spin_quant.value())
        cores = int(self.spin_cores.value())
        runs = int(self.spin_runs.value())
        starv = float(self.spin_starv.value())
        ctx = float(self.spin_ctx.value())
        use_mutex = self.chk_mutex.isChecked()

        if self.chk_compare.isChecked():
            res_sem = run_simulation_single(n_process=n, mean_arrival=arr, mean_cs=cs,
                                        semaphore_count=semc, scheduler=sched, quantum=quant,
                                        cpu_cores=cores, max_cs_per_proc=runs, starv_mult=starv,
                                        ctx_switch=ctx, use_mutex=False,seed=42)
            res_mtx = run_simulation_single(n_process=n, mean_arrival=arr, mean_cs=cs,
                                        semaphore_count=semc, scheduler=sched, quantum=quant,
                                        cpu_cores=cores, max_cs_per_proc=runs, starv_mult=starv,
                                        ctx_switch=ctx, use_mutex=True,seed=42)
            self.show_compare_results(res_sem, res_mtx)
            res = res_sem
        else:
            res = run_simulation_single(n_process=n, mean_arrival=arr, mean_cs=cs,
                                    semaphore_count=semc, scheduler=sched, quantum=quant,
                                    cpu_cores=cores, max_cs_per_proc=runs, starv_mult=starv,
                                    ctx_switch=ctx, use_mutex=use_mutex,seed=42)
            


        self.last_result = res
        txt = (f"Avg wait: {res['avg_wait']:.4f}\nStd wait: {res['std_wait']:.4f}\n"
               f"Throughput: {res['throughput']:.4f} proc/time\nCPU util: {res['cpu_util']*100:.2f}%\n"
               f"CPU idle (sum cores): {res['cpu_idle']:.4f}\nStarvation rate: {res['starvation_rate']*100:.2f}% "
               f"({res['starved_count']} events)\nAvg queue len (sample): {res.get('avg_queue_len', 0):.3f}\n"
               f"interrupt count: {res['interrupt_count']}\n")
        self.lbl_metrics.setText(txt)
        self.status.showMessage("Simulation finished.", 5000)

        # draw gantt
        self._draw_gantt(res["timeline"])
        # draw hist
        self._draw_hist(res["wait_times"])
        # draw cpu
        self._draw_cpu(res["cpu_util"])

    # helpers to draw
    def _draw_gantt(self, timeline):
        self.canvas_gantt.axes.clear()
        if not timeline:
            self.canvas_gantt.axes.text(0.5, 0.5, "No timeline", ha='center')
            self.canvas_gantt.draw()
            return
        # order pids nicely
        pids = sorted(list({pid for pid, s, e in timeline}), key=lambda x: int(x[1:]) if x.startswith('P') and x[1:].isdigit() else x)
        pid_to_y = {pid: i for i, pid in enumerate(pids)}
        cmap = plt.get_cmap("tab20")
        colors = {}
        for (pid, s, e) in timeline:
            y = pid_to_y[pid]
            if pid not in colors:
                colors[pid] = cmap(hash(pid) % 20)
            self.canvas_gantt.axes.barh(y, e - s, left=s, height=0.6, color=colors[pid])
        self.canvas_gantt.axes.set_yticks(list(pid_to_y.values()))
        self.canvas_gantt.axes.set_yticklabels(list(pid_to_y.keys()))
        self.canvas_gantt.axes.set_xlabel("Time")
        self.canvas_gantt.axes.set_title("Gantt-like timeline (critical sections)")
        self.canvas_gantt.draw()

    def _draw_hist(self, waits):
        self.canvas_hist.axes.clear()
        if waits:
            self.canvas_hist.axes.hist(waits, bins=30)
            self.canvas_hist.axes.set_xlabel("Wait time")
            self.canvas_hist.axes.set_ylabel("Frequency")
            self.canvas_hist.axes.set_title("Wait time distribution")
        else:
            self.canvas_hist.axes.text(0.5, 0.5, "No wait data", ha='center')
        self.canvas_hist.draw()

    def _draw_cpu(self, util):
        self.canvas_cpu.axes.clear()
        self.canvas_cpu.axes.barh([0], [util], height=0.4)
        self.canvas_cpu.axes.set_xlim(0, 1)
        self.canvas_cpu.axes.set_yticks([])
        self.canvas_cpu.axes.set_xlabel("CPU Utilization (fraction of total capacity)")
        self.canvas_cpu.axes.set_title(f"CPU Utilization = {util*100:.2f}%")
        self.canvas_cpu.draw()

    def show_compare_results(self, sem, mtx):
        metrics = [
            ("Avg Wait", "avg_wait"),
            ("Std Wait", "std_wait"),
            ("Throughput", "throughput"),
            ("CPU Util", "cpu_util"),
            ("CPU Idle", "cpu_idle"),
            ("Total Time", "total_time"),
            ("Starvation", "starvation_rate"),
            ("Starved", "starved_count"),
            ("Avg Queue", "avg_queue_len"),
            ("Wait Count", "wait_count"),
            ("Semaph Ops", "semaphore_ops"),
            ("Interrupt", "interrupt_count"),
        ]

    # header
        out = [f"{'Metric':<10}\t{'Semaphore':<9}\t{'Mutex':<9}", "-"*30]

        for label, key in metrics:
            sval = sem.get(key, 0)
            mval = mtx.get(key, 0)
         # format angka
            sval_str = f"{sval:.4f}" if isinstance(sval, float) else str(sval)
            mval_str = f"{mval:.4f}" if isinstance(mval, float) else str(mval)
            out.append(f"{label:<10}\t{sval_str:<9}\t{mval_str:<9}")

    # pakai monospace agar tab rapi
        self.output_box.setFont(QtGui.QFont("Courier New"))
        self.output_box.setPlainText("\n".join(out))


    # -------------------------
    # Batch handling (Tab)
    # -------------------------
    @QtCore.Slot()
    def on_run_batch(self):
        # disable button
        self.btn_run_batch.setEnabled(False)
        self.btn_save_batch.setEnabled(False)
        self.batch_progress.setValue(0)
        self.batch_txt.clear()
        self.status.showMessage("Starting batch experiments...")

        # default scenarios (as agreed)
        scenarios = {
            "Low": {"n_process": 100, "mean_arrival": 1.0, "mean_cs": 1.0, "semaphore_count": 4},
            "Medium": {"n_process": 500, "mean_arrival": 0.5, "mean_cs": 3.0, "semaphore_count": 2},
            "High": {"n_process": 1000, "mean_arrival": 0.2, "mean_cs": 6.0, "semaphore_count": 1}
        }
        schedulers = ["RR", "PRIO", "MLFQ"]
        repeats = 20  # chosen for paper-quality stats

        base_cfg = {
            "quantum": float(self.spin_quant.value()),
            "cpu_cores": int(self.spin_cores.value()),
            "max_cs_per_process": int(self.spin_runs.value()),
            "starv_mult": float(self.spin_starv.value()),
            "ctx_switch": float(self.spin_ctx.value()),
            "use_mutex": False
        }

        # create worker & thread
        self.worker_thread = QtCore.QThread()
        self.batch_worker = BatchWorker(scenarios, schedulers, base_cfg, repeats, seed_base=1000)
        self.batch_worker.moveToThread(self.worker_thread)
        self.batch_worker.progress.connect(self.batch_progress.setValue)
        self.batch_worker.finished.connect(self._on_batch_finished)
        self.worker_thread.started.connect(self.batch_worker.run)
        self.worker_thread.start()

    @QtCore.Slot(object)
    def _on_batch_finished(self, df):
        # store df
        self.last_batch_df = df
        # render summary to text preview (grouped mean/std)
        if df is None or df.empty:
            self.batch_txt.setPlainText("No results")
        else:
            # compute summary table
            pivot = df.groupby(["scenario", "scheduler"]).agg(
                avg_wait_mean = ("avg_wait", "mean"),
                avg_wait_std  = ("avg_wait", "std"),
                throughput_mean = ("throughput", "mean"),
                cpu_util_mean = ("cpu_util", "mean")
            ).reset_index()
            txt = pivot.to_string(index=False, float_format="%.4f")
            self.batch_txt.setPlainText(txt)
            # enable save
            self.btn_save_batch.setEnabled(True)
        self.batch_progress.setValue(100)
        self.btn_run_batch.setEnabled(True)
        self.status.showMessage("Batch finished.", 8000)
        # cleanup thread
        try:
            self.worker_thread.quit()
            self.worker_thread.wait(1000)
        except Exception:
            pass

    @QtCore.Slot()
    def on_save_batch(self):
        if self.last_batch_df is None:
            QtWidgets.QMessageBox.information(self, "No data", "Run batch first.")
            return
        # ask path
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save batch CSV", "batch_results.csv", "CSV files (*.csv)")
        if not path:
            return
        # write CSV (json fields retained)
        self.last_batch_df.to_csv(path, index=False)
        QtWidgets.QMessageBox.information(self, "Saved", f"Saved batch CSV to:\n{path}")

# =========================
# run app
# =========================
def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
