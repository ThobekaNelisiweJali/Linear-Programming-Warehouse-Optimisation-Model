# Zonix Logistics – Warehouse Optimization Tool 📦⚙️

## 📖 Project Overview

Zonix Logistics operates multi-client distribution warehouses handling thousands of SKUs daily. Traditional manual planning often leads to:

Underutilized warehouse zones

Longer item retrieval times

Constraint violations (e.g., stacking, temperature, hazardous storage)

This project develops a Linear Programming (LP)-based optimization tool that automates warehouse storage allocation while respecting operational constraints. It provides data-driven recommendations, visual outputs, and an interactive dashboard for decision-makers.

## 🎯 Objectives

- Dynamically recommend optimal storage zones for each SKU.

- Maximize space utilization and reduce retrieval times.

- Respect warehouse rules: temperature zones, hazard separation, stacking policies.

- Deliver actionable visualizations and reports for planners and operators.

## 🚀 Key Features

- Optimization Engine: Built using Python + PuLP, solves SKU-to-zone assignments.

- Constraint Handling: Space, temperature, hazard separation, stackability, accessibility.

- Interactive Dashboard: Streamlit-based interface with data upload, optimization runs, and visualizations.

- Visual Reporting: Before vs after optimization metrics, heatmaps, and allocation charts.

- Sensitivity Analysis: Stress tests warehouse allocation under fluctuating demand and capacity scenarios.

## 📂 Data Sources

The project integrates 10 CSV datasets representing real warehouse operations:

- zonix_warehouse_items.csv – SKU dimensions, weight, fragility, hazard type.

- zonix_storage_zones.csv – Zone IDs, capacity, weight/volume limits.

- zonix_item_movements.csv – Frequency and distance of SKU movements.

- zonix_incompatibility_rules.csv – Rules for hazardous/fragile/electronic storage.

- zonix_zone_temperatures.csv – Temperature class of each storage zone.

- zonix_inventory_snapshots.csv – Current SKU stock levels.

- zonix_order_picking_paths.csv – Picking routes within the warehouse.

- zonix_shipment_logs.csv – Historical outbound shipments.

- zonix_equipment_logs.csv – Forklift and equipment usage data.

- zonix_employee_shifts.csv – Workforce schedules.

## 🛠️ Tech Stack

- Programming: Python 3.x

### Libraries:

- Optimization → PuLP, OR-Tools, SciPy.optimize

- Data Processing → Pandas, NumPy

- Visualization → Plotly, Matplotlib, Dash

- Interface → Streamlit

- Deployment (optional): Streamlit Cloud / AWS / Azure

## ⚙️ Installation & Setup

Clone the repository:
````
git clone https://github.com/<your-username>/zonix-warehouse-optimizer.git
cd zonix-warehouse-optimizer
````

Create and activate a virtual environment:
````
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
````

Install dependencies:
````
pip install -r requirements.txt
````

Run the app:
````
streamlit run app.py
````
## 🖥️ Usage

Prepare your warehouse data in CSV format (Items.csv, Zones.csv).

Launch the Streamlit app.

Upload the datasets via the dashboard.

Click Run Optimization.

Explore outputs:

Space utilization before vs after

Allocation changes

Heatmaps of constraints

Downloadable Excel reports

📊 Tip: Place screenshots of the dashboard in /docs/screenshots/ and link them here for better visualization.

📈 Project Phases

Phase 1 – Planning & Requirements: Identified warehouse challenges.

Phase 2 – Data Extraction & Preprocessing: Cleaned, merged, and standardized datasets.

Phase 3 – LP Model Design: Formulated decision variables, objectives, and constraints.

Phase 4 – Tool Development: Built Python + PuLP solver integrated into Streamlit.

Phase 5 – Visualization & Dashboarding: Created interactive UI and visual analytics.

Phase 6 – Sensitivity Analysis: Tested robustness under demand and capacity fluctuations.

📊 Results & Benefits

+15% higher space efficiency.

-10% reduction in item retrieval time.

Automated layout planning → hours reduced to minutes.

Better safety and regulatory compliance.

Scalable to multiple warehouses.

🔮 Roadmap / Next Steps

 Real-time WMS integration for live optimization.

 SKU tagging and advanced filtering in dashboard.

 Full grid-based warehouse map visualization.

 Cloud deployment for enterprise access.

 AI-powered demand forecasting integration.

👥 Contributors

Thobie Jali – Project Lead, Optimization Model & Dashboard Development.

Zonix Logistics Team – Provided operational insights and data.

🙌 Acknowledgements

Open-source contributors of PuLP and Streamlit.

Research references from operations research and supply chain optimization literature.
