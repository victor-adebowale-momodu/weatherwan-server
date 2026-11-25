import os
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)
rng = np.random.default_rng(42)

# ==================== 1. 375-PARAM MIXED-INPUT MLP ====================
def relu(x):
    return np.maximum(0.0, x)

def sigmoid(x):
    x = np.clip(x, -15.0, 15.0)
    return 1.0 / (1.0 + np.exp(-x))

class MLP375:
    def __init__(self, weights=None):
        if weights is None:
            self.w = rng.normal(0.0, 0.1, size=375).astype(np.float32)
        else:
            self.w = weights.astype(np.float32).copy()

    def unpack(self, vec):
        W_dyn  = vec[0:72].reshape(6, 12)
        b_dyn  = vec[72:84]
        W_stat = vec[84:102].reshape(3, 6)
        b_stat = vec[102:108]
        W_fus  = vec[108:324].reshape(18, 12)
        b_fus  = vec[324:336]
        W_head = vec[336:372].reshape(12, 3)
        b_head = vec[372:375]
        return W_dyn, b_dyn, W_stat, b_stat, W_fus, b_fus, W_head, b_head

    def forward(self, X_dyn, X_stat):
        W_dyn, b_dyn, W_stat, b_stat, W_fus, b_fus, W_head, b_head = self.unpack(self.w)
        h_dyn  = relu(X_dyn @ W_dyn + b_dyn)
        h_stat = relu(X_stat @ W_stat + b_stat)
        h_cat  = np.concatenate([h_dyn, h_stat], axis=1)
        h_fus  = relu(h_cat @ W_fus + b_fus)
        out    = sigmoid(h_fus @ W_head + b_head)
        return out, (X_dyn, X_stat, h_dyn, h_stat, h_cat, h_fus, out)

    def loss_only(self, X_dyn, X_stat, Y):
        out, _ = self.forward(X_dyn, X_stat)
        eps = 1e-6
        mse = 0.5 * np.mean((out[:, :2] - Y[:, :2]) ** 2)
        bce = -np.mean(Y[:, 2] * np.log(out[:, 2] + eps) + (1 - Y[:, 2]) * np.log(1 - out[:, 2] + eps))
        return float(mse + bce)

    def train_epoch(self, X_dyn, X_stat, Y, w_global, lr=0.08, mu=0.01):
        N = len(X_dyn)
        out, (X_dyn, X_stat, h_dyn, h_stat, h_cat, h_fus, _) = self.forward(X_dyn, X_stat)
        W_dyn, b_dyn, W_stat, b_stat, W_fus, b_fus, W_head, b_head = self.unpack(self.w)

        d_out = np.zeros_like(out)
        d_out[:, 0] = (out[:, 0] - Y[:, 0]) * out[:, 0] * (1.0 - out[:, 0]) / N
        d_out[:, 1] = (out[:, 1] - Y[:, 1]) * out[:, 1] * (1.0 - out[:, 1]) / N
        d_out[:, 2] = (out[:, 2] - Y[:, 2]) / N

        dW_head = h_fus.T @ d_out
        db_head = d_out.sum(axis=0)
        d_fus   = (d_out @ W_head.T) * (h_fus > 0)

        dW_fus  = h_cat.T @ d_fus
        db_fus  = d_fus.sum(axis=0)
        d_cat   = d_fus @ W_fus.T

        d_hdyn  = d_cat[:, :12] * (h_dyn > 0)
        d_hstat = d_cat[:, 12:] * (h_stat > 0)

        dW_dyn  = X_dyn.T @ d_hdyn
        db_dyn  = d_hdyn.sum(axis=0)
        dW_stat = X_stat.T @ d_hstat
        db_stat = d_hstat.sum(axis=0)

        grads = np.concatenate([
            dW_dyn.ravel(), db_dyn,
            dW_stat.ravel(), db_stat,
            dW_fus.ravel(), db_fus,
            dW_head.ravel(), db_head
        ]).astype(np.float32)

        grads = np.clip(grads + mu * (self.w - w_global), -5.0, 5.0)
        self.w -= lr * grads
        return self.loss_only(X_dyn, X_stat, Y)

# ==================== 2. 30-DAY WEATHERWAN SIMULATION TRACES ====================
DAYS = 30
DT_MIN = 5
STEPS = DAYS * 24 * (60 // DT_MIN)  # 8640 steps

V_UC_MIN = 2.5
V_UC_MAX = 5.75

NODE_PROFILES = [
    {"name": "WhiteHouse_3S2P_1500F", "stat": [0.52, 0.50, 0.40], "shade": 1.00, "uc_F": 1500.0, "bat_Wh": 57.7},
    {"name": "Science_Shaded_1500F",  "stat": [0.52, 0.50, 0.40], "shade": 0.65, "uc_F": 1500.0, "bat_Wh": 57.7},
    {"name": "Secretariat_3000F",     "stat": [0.52, 1.00, 0.40], "shade": 1.00, "uc_F": 3000.0, "bat_Wh": 57.7},
    {"name": "AgedBat_1500F",         "stat": [0.35, 0.50, 0.35], "shade": 0.85, "uc_F": 1500.0, "bat_Wh": 38.0},
]

def generate_weather(node_idx):
    t_hr = np.arange(STEPS) * (DT_MIN / 60.0)
    hour = t_hr % 24.0
    day  = (t_hr // 24.0).astype(int)

    cloud = np.ones(STEPS)
    for d0, d1, factor in [(6, 8, 0.25), (15, 18, 0.20), (24, 26, 0.35)]:
        cloud[(day >= d0) & (day < d1)] = factor

    shade = NODE_PROFILES[node_idx]["shade"]
    solar_curve = np.maximum(0.0, np.sin((hour - 6.0) * np.pi / 12.0))
    irr = np.clip(820.0 * solar_curve * cloud * shade + rng.normal(0, 15, STEPS), 0.0, 1000.0)

    temp = 27.5 - 5.5 * np.cos(2 * np.pi * (hour - 2.0) / 24.0) - 3.0 * (1.0 - cloud) + rng.normal(0, 0.15, STEPS)
    hum  = np.clip(62.0 + 20.0 * np.cos(2 * np.pi * (hour - 2.0) / 24.0) + 10.0 * (1.0 - cloud) + rng.normal(0, 0.4, STEPS), 20.0, 98.0)
    pres = 1012.0 + 1.5 * np.sin(4 * np.pi * hour / 24.0) - 2.5 * (1.0 - cloud) + rng.normal(0, 0.08, STEPS)
    return hour, irr, temp, hum, pres, cloud

datasets = []
for idx, prof in enumerate(NODE_PROFILES):
    hour, irr, temp, hum, pres, cloud = generate_weather(idx)
    p_pv = 40.0 * (irr / 1000.0) * 0.85
    vol  = np.clip(np.abs(np.diff(temp, prepend=temp[0])) / 2.0 + np.abs(np.diff(hum, prepend=hum[0])) / 6.0, 0.0, 1.0)
    soc  = np.clip(0.65 + 0.20 * np.sin(np.arange(STEPS) * 2 * np.pi / 288) - 0.25 * (1 - cloud), 0.15, 0.92)
    v_uc = np.clip(4.4 + 1.2 * np.sin(np.arange(STEPS) * 2 * np.pi / 288) - 1.1 * (1 - cloud), 2.3, V_UC_MAX)

    X_dyn = np.column_stack([
        soc, v_uc / V_UC_MAX, p_pv / 40.0, vol,
        0.5 * (1.0 + np.sin(2 * np.pi * hour / 24.0)),
        0.5 * (1.0 + np.cos(2 * np.pi * hour / 24.0)),
    ]).astype(np.float32)
    X_stat = np.tile(prof["stat"], (STEPS, 1)).astype(np.float32)

    e_score = 0.6 * soc + 0.4 * (p_pv / 40.0)
    Y = np.column_stack([
        np.clip(1.0 - e_score, 0.05, 0.95),
        np.clip(0.75 - 0.60 * e_score, 0.10, 0.85),
        ((v_uc < 0.85 * V_UC_MAX) & (soc > 0.35)).astype(np.float32),
    ]).astype(np.float32)
    datasets.append((X_dyn, X_stat, Y, hour, irr, temp, hum, pres, cloud))

# ==================== 3. TRAIN FL & CENTRALIZED MODELS ====================
ROUNDS, EPOCHS = 30, 5
global_model  = MLP375()
central_model = MLP375(global_model.w)

X_d_all = np.vstack([d[0][:6000] for d in datasets])
X_s_all = np.vstack([d[1][:6000] for d in datasets])
Y_all   = np.vstack([d[2][:6000] for d in datasets])
X_d_val = np.vstack([d[0][6000:] for d in datasets])
X_s_val = np.vstack([d[1][6000:] for d in datasets])
Y_val   = np.vstack([d[2][6000:] for d in datasets])

fl_val_losses, cent_val_losses = [], []
for r in range(ROUNDS):
    local_weights, local_sizes = [], []
    for X_d, X_s, Y, *_ in datasets:
        client = MLP375(global_model.w)
        n_tr = 6000
        for _ in range(EPOCHS):
            client.train_epoch(X_d[:n_tr], X_s[:n_tr], Y[:n_tr], global_model.w, lr=0.08, mu=0.01)
        local_weights.append(client.w)
        local_sizes.append(n_tr)

    total_n = sum(local_sizes)
    global_model.w = sum((n / total_n) * w for n, w in zip(local_sizes, local_weights))
    fl_val_losses.append(global_model.loss_only(X_d_val, X_s_val, Y_val))

    for _ in range(EPOCHS):
        central_model.train_epoch(X_d_all, X_s_all, Y_all, central_model.w, lr=0.08, mu=0.0)
    cent_val_losses.append(central_model.loss_only(X_d_val, X_s_val, Y_val))

# ==================== 4. SIMULATE 30 DAYS ON NODE 1 ====================
def simulate_node(strategy="fl"):
    prof = NODE_PROFILES[0]
    X_d, X_s, _, hour, irr, temp, hum, pres, cloud = datasets[0]
    bat_wh = prof["bat_Wh"]
    e_bat = 0.75 * bat_wh * 3600.0
    e_bat_max = bat_wh * 3600.0
    c_uc = prof["uc_F"]
    v_uc = 5.1

    last_t, last_h, last_p, last_i = temp[0], hum[0], pres[0], irr[0]
    mins_since = 999
    tx_events, uc_covered, brownouts = 0, 0, 0
    peak_bat_tx_ma, in_safe_soc = 2.0, 0
    e_leak_total, active_energy_j = 0.0, 0.0

    hist_soc, hist_vuc, hist_pv, hist_tx = [], [], [], []
    model = global_model if strategy == "fl" else central_model

    for t in range(STEPS):
        soc = e_bat / e_bat_max
        if 0.20 <= soc <= 0.90:
            in_safe_soc += 1

        p_pv = 40.0 * (irr[t] / 1000.0) * 0.85
        e_base = 0.075
        e_bat = max(0.0, e_bat - e_base)
        active_energy_j += e_base

        if strategy == "static":
            pi_chg = 0.0
            do_tx = (mins_since >= 15)
        else:
            x_d = np.array([[soc, v_uc / V_UC_MAX, p_pv / 40.0, X_d[t, 3], X_d[t, 4], X_d[t, 5]]], dtype=np.float32)
            out, _ = model.forward(x_d, X_s[t:t+1])
            dt_tx  = 15.0 + out[0, 0] * 105.0
            delta  = 0.05 + out[0, 1] * 0.95
            pi_chg = out[0, 2]
            dev = max(abs(temp[t]-last_t)/5.0, abs(hum[t]-last_h)/20.0, abs(pres[t]-last_p)/5.0, abs(irr[t]-last_i)/300.0)
            do_tx = (dev >= delta) or (mins_since >= dt_tx)

        if p_pv > 0.3:
            e_in = p_pv * 300.0 * 0.88
            charge_uc = (irr[t] < 220.0) or (pi_chg > 0.5)
            if charge_uc and v_uc < V_UC_MAX:
                e_now = 0.5 * c_uc * (v_uc ** 2)
                e_cap = 0.5 * c_uc * (V_UC_MAX ** 2)
                add = min(e_in, e_cap - e_now)
                v_uc = np.sqrt(2.0 * (e_now + add) / c_uc)
                if e_in - add > 0:
                    e_bat = min(e_bat_max * 0.90, e_bat + (e_in - add))
            else:
                e_bat = min(e_bat_max * 0.90, e_bat + e_in)

        i_leak = 0.0005 * (v_uc / V_UC_MAX)
        v_new = max(0.0, v_uc - (i_leak * 300.0) / c_uc)
        e_leak_total += 0.5 * c_uc * (v_uc**2 - v_new**2)
        v_uc = v_new

        did_tx = 0
        if do_tx:
            e_tx = 2.1
            if soc <= 0.05 and v_uc < V_UC_MIN:
                brownouts += 1
            else:
                tx_events += 1
                did_tx = 1
                active_energy_j += e_tx
                if v_uc >= V_UC_MIN and (0.5 * c_uc * (v_uc**2 - V_UC_MIN**2)) >= e_tx:
                    v_uc = np.sqrt(2.0 * (0.5 * c_uc * (v_uc**2) - e_tx) / c_uc)
                    uc_covered += 1
                else:
                    e_bat = max(0.0, e_bat - e_tx)
                    peak_bat_tx_ma = max(peak_bat_tx_ma, 195.0)
                last_t, last_h, last_p, last_i = temp[t], hum[t], pres[t], irr[t]
                mins_since = 0
        else:
            mins_since += DT_MIN

        hist_soc.append(100.0 * e_bat / e_bat_max)
        hist_vuc.append(v_uc)
        hist_pv.append(p_pv)
        hist_tx.append(did_tx)

    # Centralized raw uploads: 288 * 320B = 92.16 KB/day; FL sync ~4.8 KB/day (up + down)
    raw_kb_day   = 92.16 if strategy == "centralized" else 0.0
    fl_kb_day    = 4.80 if strategy == "fl" else 0.0
    telem_kb_day = (tx_events / DAYS) * 0.22
    total_kb_day = raw_kb_day + fl_kb_day + telem_kb_day

    return {
        "tx": tx_events,
        "supp_pct": 100.0 * (1.0 - tx_events / 2880.0) if strategy != "static" else 0.0,
        "wh_day": (active_energy_j / 3600.0) / DAYS,
        "uc_cov_pct": 100.0 * uc_covered / max(1, tx_events),
        "peak_ibat_ma": peak_bat_tx_ma,
        "safe_soc_pct": 100.0 * in_safe_soc / STEPS,
        "leak_j": e_leak_total,
        "uptime_pct": 100.0 * (1.0 - brownouts / STEPS),
        "lpsp": brownouts / STEPS,
        "raw_kb_day": raw_kb_day,
        "fl_kb_day": fl_kb_day,
        "telem_kb_day": telem_kb_day,
        "total_kb_day": total_kb_day,
        "hist": (hist_soc, hist_vuc, hist_pv, hist_tx),
    }

res_stat = simulate_node("static")
res_cent = simulate_node("centralized")
res_fl   = simulate_node("fl")

print(f"\n=== CONVERGENCE ===")
print(f"Final FL Val Loss: {fl_val_losses[-1]:.5f} | Final Centralized Val Loss: {cent_val_losses[-1]:.5f}")

print("\n=== TABLE 4.2: TX & ENERGY ===")
for n, r in [("Static 15-min", res_stat), ("Centralized ML", res_cent), ("Proposed FL", res_fl)]:
    saved = 100.0 * (1.0 - r["wh_day"] / res_stat["wh_day"])
    print(f"{n:15s} | TX={r['tx']:4d} | Supp={r['supp_pct']:5.1f}% | Wh/day={r['wh_day']:.4f} | Saved={saved:5.1f}%")

print("\n=== TABLE 4.3 & 4.4: HESS & RELIABILITY ===")
for n, r in [("Static 15-min", res_stat), ("Centralized ML", res_cent), ("Proposed FL", res_fl)]:
    print(f"{n:15s} | UC_Cov={r['uc_cov_pct']:5.1f}% | Peak_Ibat={r['peak_ibat_ma']:5.1f}mA | "
          f"SafeSoC={r['safe_soc_pct']:5.1f}% | Leak={r['leak_j']:.1f}J | Uptime={r['uptime_pct']:.2f}% | LPSP={r['lpsp']:.4f}")

print("\n=== TABLE 4.5: COMMUNICATION (KB/DAY) ===")
for n, r in [("Centralized ML", res_cent), ("Proposed FL", res_fl)]:
    print(f"{n:15s} | Raw={r['raw_kb_day']:.2f} | FL={r['fl_kb_day']:.2f} | Telem={r['telem_kb_day']:.2f} | Total={r['total_kb_day']:.2f}")
print(f"Payload Reduction Factor (Centralized / FL): {res_cent['total_kb_day'] / res_fl['total_kb_day']:.2f}x")

# ==================== 5. GENERATE 4 CHAPTER 4 PDF FIGURES ====================
plt.figure(figsize=(7, 4.2))
plt.plot(range(1, ROUNDS + 1), fl_val_losses, marker="o", markersize=3.5, label="Proposed On-Device FL (FedProx)")
plt.plot(range(1, ROUNDS + 1), cent_val_losses, linestyle="--", label="Centralized ML Baseline")
plt.xlabel("Federated Communication Round")
plt.ylabel("Validation Loss")
plt.title("Simulated Global Model Convergence Across 30 Rounds")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("figures/fl_convergence.pdf")
plt.close()

t_hrs = np.arange(576) * (DT_MIN / 60.0)
_, _, _, _, _, temp, _, _, _ = datasets[0]
plt.figure(figsize=(7.5, 4.2))
plt.plot(t_hrs, temp[:576], color="tab:blue", label="Simulated Temp (C)")
tx_fl = np.where(np.array(res_fl["hist"][3][:576]) == 1)[0]
tx_st = np.where(np.array(res_stat["hist"][3][:576]) == 1)[0]
plt.scatter(tx_st * (DT_MIN/60.0), temp[tx_st] + 0.4, marker="|", s=45, color="gray", alpha=0.6, label="Fixed 15-min TX")
plt.scatter(tx_fl * (DT_MIN/60.0), temp[tx_fl], marker="^", s=35, color="tab:red", label="FL Event-Triggered TX")
plt.xlabel("Time (Hours)")
plt.ylabel("Ambient Temperature (C)")
plt.title("48-Hour Weather Variation vs. Suppressed and Triggered Transmissions")
plt.grid(True, alpha=0.3)
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig("figures/tx_suppression.pdf")
plt.close()

s0, s1 = 5 * 288, 9 * 288
t_days = np.arange(s0, s1) / 288.0
fig, axes = plt.subplots(3, 1, figsize=(7.5, 6.2), sharex=True)
axes[0].plot(t_days, res_fl["hist"][2][s0:s1], color="tab:orange")
axes[0].set_ylabel("Solar PV (W)")
axes[0].grid(True, alpha=0.3)
axes[1].plot(t_days, res_fl["hist"][0][s0:s1], color="tab:green", label="Proposed FL")
axes[1].plot(t_days, res_stat["hist"][0][s0:s1], color="gray", linestyle="--", label="Static Baseline")
axes[1].set_ylabel("Battery SoC (%)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[2].plot(t_days, res_fl["hist"][1][s0:s1], color="tab:purple")
axes[2].axhline(V_UC_MIN, color="red", linestyle=":", label="V_uc,min (2.5 V)")
axes[2].set_ylabel("Ultracap (V)")
axes[2].set_xlabel("Time (Days)")
axes[2].legend()
axes[2].grid(True, alpha=0.3)
fig.suptitle("Simulated Hybrid Storage Trajectory During Sunny and Overcast Conditions")
plt.tight_layout()
plt.savefig("figures/hess_trace.pdf")
plt.close()

labels = ["Static 15-min", "Centralized ML", "Proposed FL"]
values = [res_stat["total_kb_day"], res_cent["total_kb_day"], res_fl["total_kb_day"]]
plt.figure(figsize=(6.5, 4.0))
bars = plt.bar(labels, values, color=["gray", "tab:red", "tab:blue"], width=0.5)
for b in bars:
    plt.text(b.get_x() + b.get_width()/2, b.get_height() + 1.5, f"{b.get_height():.1f} KB", ha="center", fontsize=9)
plt.ylabel("Daily Wireless Payload per Node (KB/day)")
plt.title("Daily Communication Overhead Comparison")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("figures/comm_overhead.pdf")
plt.close()
print("Saved all 4 PDF figures to figures/!")