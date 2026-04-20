# Client Intelligence Platform

## Overview
Client Intelligence Platform is a cloud-deployed analytics application that transforms raw business data into actionable insights for business users, analysts, and consultants.

It enables users to upload CSV or Excel files and instantly generate:
- KPI dashboards
- trend analysis
- anomaly detection
- risk alerts
- business recommendations

## Features
- Upload CSV/XLSX datasets
- Demo dataset mode
- Executive summary generation
- KPI snapshot cards
- Risk alerts
- Business recommendations
- Interactive filters
- Smart chart recommendations
- Data quality score
- Export filtered data and insights as CSV

## Tech Stack
- Python
- Streamlit
- Pandas
- NumPy
- Plotly
- Docker
- GitHub Actions
- Google Cloud Run

## Architecture
1. User uploads dataset or selects demo mode
2. Data is processed and profiled using Pandas
3. KPIs, charts, risks, and recommendations are generated
4. Application is containerized with Docker
5. CI/CD is handled using GitHub Actions
6. Deployment is hosted on Google Cloud Run

## Live Demo
https://client-intelligence-platform-198023219431.europe-west1.run.app

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py