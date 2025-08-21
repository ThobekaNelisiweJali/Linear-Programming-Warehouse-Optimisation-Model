# Phase 5: Streamlit App for Warehouse LP Optimization — Upgraded UI/UX + Preprocessing
# Save as `app.py` and run with:
#   streamlit run app.py
# Recommended install:
#   pip install streamlit pulp pandas numpy plotly xlsxwriter
# Optional (only if you want PDF export via HTML-to-PDF tools):
#   pip install jinja2

import io
import typing as t

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    import pulp
except Exception:
    pulp = None

# -----------------------------
# App Config & Theming
# -----------------------------
st.set_page_config(
    page_title="Zonix Warehouse Optimizer",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Light CSS polish (optional branding hook)
st.markdown(
    """
    <style>
      .kpi-card {background: #0f172a0D; padding: 14px 16px; border-radius: 16px; border:1px solid #e5e7eb}
      .small-note {color:#64748b; font-size:0.9rem}
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Schemas (model expects these after preprocessing)
# -----------------------------
DEFAULT_ITEMS_SCHEMA = {
    "sku_id": "string",
    "volume": "float (m³)",
    "demand": "float",
    "is_hazardous": "0/1",
    "is_priority": "0/1",
    "current_zone": "string"
}

DEFAULT_ZONES_SCHEMA = {
    "zone_id": "string",
    "capacity": "float (m³)",
    "distance_to_exit": "float",
    "is_exit_adjacent": "0/1"
}

# -----------------------------
# Helpers
# -----------------------------

def normalize_zone_labels(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col not in df.columns:
        return df
    def norm(z):
        if pd.isna(z):
            return z
        s = str(z).strip()
        s_low = s.lower()
        for prefix in ["zone ", "zone-", "zone_", "z "]:
            if s_low.startswith(prefix):
                s = s[len(prefix):]
                break
        s = s.replace("Zone ", "").replace("zone ", "").strip()
        return s.upper()
    df[col] = df[col].map(norm)
    return df

def validate_schema(df: pd.DataFrame, required_cols: t.List[str]) -> t.Tuple[bool, t.List[str]]:
    missing = [c for c in required_cols if c not in df.columns]
    return (len(missing) == 0), missing

# -----------------------------
# Preprocessing to map YOUR columns → expected model schema
# This is tailored to your screenshots:
# Items columns: item_id, category, weight_kg, length_cm, width_cm, height_cm, stackable, fragile, hazardous, temperature_range
# Zones columns: zone_id, max_volume_m3, max_weight_kg, temperature_class, near_loading_dock
# -----------------------------

def preprocess_from_zonix(items_raw: pd.DataFrame, zones_raw: pd.DataFrame) -> t.Tuple[pd.DataFrame, pd.DataFrame]:
    items = items_raw.copy()
    zones = zones_raw.copy()

    # Items
    rename_items = {
        "item_id": "sku_id",
        "hazardous": "is_hazardous",
    }
    items.rename(columns=rename_items, inplace=True)

    # Compute volume in m^3 from cm dimensions if not present
    if "volume" not in items.columns:
        if set(["length_cm", "width_cm", "height_cm"]).issubset(items.columns):
            items["volume"] = (pd.to_numeric(items.get("length_cm", 0), errors="coerce") *
                                pd.to_numeric(items.get("width_cm", 0), errors="coerce") *
                                pd.to_numeric(items.get("height_cm", 0), errors="coerce")) / 1_000_000.0
        else:
            items["volume"] = 0.0

    # Demand default if missing
    if "demand" not in items.columns:
        # Use weight or category popularity if available; fallback 1
        base = pd.to_numeric(items.get("weight_kg", 1), errors="coerce").fillna(1)
        items["demand"] = (base / base.mean()).clip(lower=0.2, upper=5).fillna(1)

    # Priority heuristic: fragile → priority unless provided explicitly
    if "is_priority" not in items.columns:
        frag = items.get("fragile", 0)
        try:
            frag = pd.to_numeric(frag, errors="coerce").fillna(0)
        except Exception:
            frag = frag
        items["is_priority"] = np.where((frag==1) | (frag==True) | (frag=="True"), 1, 0)

    # Hazard cast
    items["is_hazardous"] = pd.to_numeric(items.get("is_hazardous", 0), errors="coerce").fillna(0).astype(int)

    # Current zone default (None)
    if "current_zone" not in items.columns:
        items["current_zone"] = None

    # Zones
    rename_zones = {
        "max_volume_m3": "capacity",
        "near_loading_dock": "is_exit_adjacent",
    }
    zones.rename(columns=rename_zones, inplace=True)

    # Distance: closer if exit-adjacent
    if "distance_to_exit" not in zones.columns:
        adj = pd.to_numeric(zones.get("is_exit_adjacent", 0), errors="coerce").fillna(0).astype(int)
        # Rank by adjacency then by zone_id for a simple gradient
        zones["distance_to_exit"] = np.where(adj==1, 1, 5)

    # Casts
    zones["is_exit_adjacent"] = pd.to_numeric(zones.get("is_exit_adjacent", 0), errors="coerce").fillna(0).astype(int)
    zones["capacity"] = pd.to_numeric(zones.get("capacity", 0), errors="coerce").fillna(0.0)

    # Normalize zone labels
    items = normalize_zone_labels(items, "current_zone")
    zones = normalize_zone_labels(zones, "zone_id")

    # Keep only expected model columns to avoid confusion
    items = items[["sku_id", "volume", "demand", "is_hazardous", "is_priority", "current_zone"]]
    zones = zones[["zone_id", "capacity", "distance_to_exit", "is_exit_adjacent"]]
    return items, zones

# -----------------------------
# LP Model
# -----------------------------

def build_lp(items: pd.DataFrame, zones: pd.DataFrame, priority_weight: float, objective: str = "min_cost"):
    if pulp is None:
        raise RuntimeError("PuLP not installed. Please run: pip install pulp")

    sku_ids = items["sku_id"].tolist()
    zone_ids = zones["zone_id"].tolist()

    vol = dict(zip(items["sku_id"], items["volume"].astype(float)))
    dem = dict(zip(items["sku_id"], items["demand"].astype(float)))
    haz = dict(zip(items["sku_id"], items["is_hazardous"].astype(int)))
    pri = dict(zip(items["sku_id"], items["is_priority"].astype(int)))

    cap = dict(zip(zones["zone_id"], zones["capacity"].astype(float)))
    dist = dict(zip(zones["zone_id"], zones["distance_to_exit"].astype(float)))
    exit_adj = dict(zip(zones["zone_id"], zones["is_exit_adjacent"].astype(int)))

    sense = pulp.LpMinimize if objective == "min_cost" else pulp.LpMaximize
    prob = pulp.LpProblem("Warehouse_Slotting", sense)

    x = pulp.LpVariable.dicts("x", (sku_ids, zone_ids), lowBound=0, upBound=1, cat=pulp.LpBinary)

    if objective == "min_cost":
        base_cost = [dist[j]*dem[i]*x[i][j] for i in sku_ids for j in zone_ids]
        priority_pen = [priority_weight*dem[i]*pri[i]*(1-exit_adj[j])*x[i][j] for i in sku_ids for j in zone_ids]
        prob += pulp.lpSum(base_cost + priority_pen), "Total_Movement_and_Priority_Penalty"
    else:
        prob += pulp.lpSum(vol[i]*x[i][j] for i in sku_ids for j in zone_ids), "Total_Volume_Used"

    for i in sku_ids:
        prob += pulp.lpSum(x[i][j] for j in zone_ids) == 1, f"assign_once_{i}"

    for j in zone_ids:
        prob += pulp.lpSum(vol[i]*x[i][j] for i in sku_ids) <= cap[j], f"capacity_{j}"

    # Hazardous separation (≤1 hazardous per zone)
    for j in zone_ids:
        prob += pulp.lpSum(x[i][j] for i in sku_ids if haz[i]==1) <= 1, f"hazardous_sep_{j}"

    return prob, x

def solve_lp(prob) -> str:
    solver = pulp.PULP_CBC_CMD(msg=False)
    status = prob.solve(solver)
    return pulp.LpStatus[status]

def extract_solution(x, items: pd.DataFrame, zones: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for i in items["sku_id"].tolist():
        for j in zones["zone_id"].tolist():
            var = x[i][j]
            if var.varValue is not None and var.varValue > 0.5:
                rows.append({"sku_id": i, "optimized_zone": j})
    return pd.DataFrame(rows)

# -----------------------------
# Metrics & Visuals
# -----------------------------


def percent_space_used(items, zones, assign_col):
    zones = zones.copy()
    items = items.copy()

    if zones["zone_id"].dtype != items[assign_col].dtype:
        print("⚠️ Converting zone_id and", assign_col, "to string for merge consistency")
        zones["zone_id"] = zones["zone_id"].astype(str)
        items[assign_col] = items[assign_col].astype(str)

    used = (
        items.groupby(assign_col)
        .agg(used_capacity=("space_required", "sum"))
        .reset_index()
    )

    merged = zones.merge(used, how="left", left_on="zone_id", right_on=assign_col)
    merged["used_capacity"] = merged["used_capacity"].fillna(0)
    merged["percent_used"] = (merged["used_capacity"] / merged["capacity"]) * 100

    return merged

# -----------------------------
# Sidebar — Inputs & Filters
# -----------------------------
st.sidebar.header("📂 Upload Data")
items_file = st.sidebar.file_uploader("Items CSV", type=["csv"])
zones_file = st.sidebar.file_uploader("Zones CSV", type=["csv"])

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Optimization Settings")
objective = st.sidebar.selectbox("Objective", ["min_cost", "max_util"], index=0)
priority_weight = st.sidebar.slider("Priority penalty (min_cost)", 0.0, 10.0, 3.0, 0.5)

st.sidebar.markdown("---")
st.sidebar.header("🔎 Filters")
sku_search = st.sidebar.text_input("Search SKU ID contains…", "")

st.title("📦 Zonix Warehouse Optimizer — Before vs After Dashboard")
st.caption("Upload your Items & Zones CSVs. The app maps them to the LP schema, optimizes, and visualizes results.")

# -----------------------------
# Data Loading & Preprocessing
# -----------------------------
if items_file is not None and zones_file is not None:
    items_raw = pd.read_csv(items_file)
    zones_raw = pd.read_csv(zones_file)

    items, zones = preprocess_from_zonix(items_raw, zones_raw)

    ok_i, miss_i = validate_schema(items, list(DEFAULT_ITEMS_SCHEMA.keys()))
    ok_z, miss_z = validate_schema(zones, list(DEFAULT_ZONES_SCHEMA.keys()))
    if not ok_i or not ok_z:
        st.error(f"Schema mismatch. Missing — Items: {miss_i} | Zones: {miss_z}")
        st.stop()

    # Optional SKU filter prior to solving (for exploration on big files)
    if sku_search.strip():
        items_view = items[items["sku_id"].astype(str).str.contains(sku_search, case=False, na=False)]
    else:
        items_view = items

    with st.expander("🔍 Peek at preprocessed data (model-ready)", expanded=False):
        st.write("**Items (first 20):**")
        st.dataframe(items_view.head(20), use_container_width=True)
        st.write("**Zones:**")
        st.dataframe(zones, use_container_width=True)

    # -----------------------------
    # Baseline (Before)
    # -----------------------------
    st.subheader("🔎 Baseline — Before Optimization")
    before_metrics = percent_space_used(items.rename(columns={"current_zone":"assigned_zone"}), zones, "assigned_zone")
    c1, c2 = st.columns([1,1])
    with c1:
        st.dataframe(before_metrics, use_container_width=True)
    with c2:
        fig_before = px.bar(before_metrics.sort_values("zone_id"), x="zone_id", y=before_metrics["pct_used"]*100,
                            labels={"x":"Zone","y":"% Used"}, title="Zone Utilization (Before)")
        st.plotly_chart(fig_before, use_container_width=True)

    # -----------------------------
    # Optimize
    # -----------------------------
    st.markdown("---")
    run = st.button("🚀 Run LP Optimization")
    if run:
        with st.spinner("Solving optimization model…"):
            prob, x = build_lp(items, zones, priority_weight=priority_weight, objective=objective)
            status = solve_lp(prob)

        st.success(f"Solver Status: {status}")
        if status not in ("Optimal", "Feasible"):
            st.error("Model did not find an optimal/feasible solution. Check capacities and constraints.")
            st.stop()

        sol = extract_solution(x, items, zones)
        items_opt = items.merge(sol, on="sku_id", how="left")

        # After metrics
        after_metrics = percent_space_used(items_opt.rename(columns={"optimized_zone":"assigned_zone"}), zones, "assigned_zone")

        # KPIs
        total_capacity = zones["capacity"].sum()
        total_before = before_metrics["used_volume"].sum()
        total_after = after_metrics["used_volume"].sum()
        pct_after = (total_after/total_capacity*100) if total_capacity>0 else 0
        reallocated = (items_opt["current_zone"].fillna("-") != items_opt["optimized_zone"].fillna("-")).sum()

        k1,k2,k3,k4 = st.columns(4)
        with k1: st.markdown(f"<div class='kpi-card'><b>Total Capacity</b><br>{total_capacity:,.2f} m³</div>", unsafe_allow_html=True)
        with k2: st.markdown(f"<div class='kpi-card'><b>Used Volume (Before)</b><br>{total_before:,.2f} m³</div>", unsafe_allow_html=True)
        with k3: st.markdown(f"<div class='kpi-card'><b>Used Volume (After)</b><br>{total_after:,.2f} m³</div>", unsafe_allow_html=True)
        with k4: st.markdown(f"<div class='kpi-card'><b>% Space Used (After)</b><br>{pct_after:,.1f}%</div>", unsafe_allow_html=True)

        # Tabs: Before / After / Comparison
        tab1, tab2, tab3, tab4 = st.tabs(["Before", "After", "Comparison", "Flows"])

        with tab1:
            st.plotly_chart(fig_before, use_container_width=True)
            st.dataframe(before_metrics, use_container_width=True)

        with tab2:
            fig_after = px.bar(after_metrics.sort_values("zone_id"), x="zone_id", y=after_metrics["pct_used"]*100,
                               labels={"x":"Zone","y":"% Used"}, title="Zone Utilization (After)")
            st.plotly_chart(fig_after, use_container_width=True)
            st.dataframe(after_metrics, use_container_width=True)

        with tab3:
            comp = before_metrics[["zone_id","pct_used"]].merge(after_metrics[["zone_id","pct_used"]], on="zone_id", suffixes=("_before","_after"))
            comp = comp.sort_values("zone_id")
            fig_comp = go.Figure()
            fig_comp.add_bar(x=comp["zone_id"], y=comp["pct_used_before"]*100, name="Before %")
            fig_comp.add_bar(x=comp["zone_id"], y=comp["pct_used_after"]*100, name="After %")
            fig_comp.update_layout(barmode="group", title="Zone Utilization — Before vs After", xaxis_title="Zone", yaxis_title="% Used")
            st.plotly_chart(fig_comp, use_container_width=True)

            st.subheader("🔀 Reallocation Recommendations")
            recs = items_opt.copy()
            recs["reallocated"] = np.where(recs["current_zone"] != recs["optimized_zone"], 1, 0)
            moved = recs[recs["reallocated"]==1].sort_values("demand", ascending=False)
            st.dataframe(moved[["sku_id","volume","demand","is_priority","is_hazardous","current_zone","optimized_zone"]], use_container_width=True)

        with tab4:
            # -----------------------------
            # Sankey Diagram Helper
            # -----------------------------
            def sankey_before_after(items_opt: pd.DataFrame):
                # Prepare data for Sankey: current_zone → optimized_zone
                df = items_opt.copy()
                df["current_zone"] = df["current_zone"].fillna("Unassigned")
                df["optimized_zone"] = df["optimized_zone"].fillna("Unassigned")
                flows = df.groupby(["current_zone", "optimized_zone"]).size().reset_index(name="count")
                all_zones = pd.unique(flows[["current_zone", "optimized_zone"]].values.ravel())
                label_map = {zone: idx for idx, zone in enumerate(all_zones)}
                flows["source"] = flows["current_zone"].map(label_map)
                flows["target"] = flows["optimized_zone"].map(label_map)
                fig = go.Figure(go.Sankey(
                    node=dict(
                        pad=15,
                        thickness=20,
                        line=dict(color="black", width=0.5),
                        label=list(all_zones),
                    ),
                    link=dict(
                        source=flows["source"],
                        target=flows["target"],
                        value=flows["count"],
                    )
                ))
                fig.update_layout(title_text="SKU Flows: Before → After Optimization", font_size=12)
                return fig

            st.plotly_chart(sankey_before_after(items_opt), use_container_width=True)

        # "AI" Insights (rule-based summary)
        st.subheader("🧠 Insights Summary")
        improved_zones = (after_metrics["pct_used"] > before_metrics.set_index("zone_id")["pct_used"].reindex(after_metrics["zone_id"]).values)
        n_improved = int(improved_zones.sum())
        txt = []
        txt.append(f"Total used volume changed from {total_before:,.2f} m³ to {total_after:,.2f} m³.")
        txt.append(f"{reallocated} SKUs were reallocated to new zones.")
        txt.append(f"Zone utilization improved in {n_improved} zones out of {len(after_metrics)}.")
        pri_moved = items_opt.query("is_priority == 1 and optimized_zone == optimized_zone")
        txt.append(f"Priority SKUs placed in exit-adjacent zones are encouraged via penalty weight = {priority_weight}.")
        st.markdown("\n".join([f"- {t}" for t in txt]))

        # Downloads
        st.subheader("📥 Downloads")
        items_opt_out = items_opt.rename(columns={"optimized_zone":"assigned_zone"})
        st.download_button(
            "Download Optimized Assignment (CSV)",
            items_opt_out.to_csv(index=False).encode("utf-8"),
            file_name="optimized_assignment.csv",
            mime="text/csv",
        )
        after_csv = after_metrics.to_csv(index=False).encode("utf-8")
        st.download_button("Download Zone Utilization (After) CSV", after_csv, file_name="zone_utilization_after.csv", mime="text/csv")

        # Excel export (one file with two sheets)
        try:
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
                items_opt_out.to_excel(writer, sheet_name="Optimized_Assignment", index=False)
                after_metrics.to_excel(writer, sheet_name="Zone_Utilization_After", index=False)
                before_metrics.to_excel(writer, sheet_name="Zone_Utilization_Before", index=False)
            st.download_button(
                label="Download Full Results (Excel)",
                data=buffer.getvalue(),
                file_name="zonix_optimization_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        except Exception:
            st.caption("Install xlsxwriter for Excel export: pip install xlsxwriter")

else:
    st.info("Upload both Items and Zones CSV files to begin.")

st.markdown("---")
st.caption("Upgraded Phase 5: preprocessing for Zonix schemas, LP optimization with priority & hazardous rules, filters, KPIs, tabs, Sankey, and CSV/Excel exports.")