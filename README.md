Perfeito. Abaixo está o **README completo**, já escrito em tom profissional, direto e com linguagem de produto/growth — sem cara de tutorial e sem exageros.

---

# 📊 Growth Funnel & Retention Analysis

**E-commerce Product Analytics Case Study**

## Project Overview

This project analyzes user behavior in a large-scale e-commerce environment to understand how product engagement translates into revenue and long-term value. The goal is to identify where users drop off in the conversion funnel, how purchasing behavior evolves over time, and which segments truly drive business impact.

The analysis simulates the type of work a Growth or Product Analytics team would perform to support data-driven decision-making across marketing, product, and revenue functions.

---

## Business Context

The dataset represents event-level user interactions from a multi-category e-commerce platform, including product views, cart additions, and purchases across two months (October and November 2019).

From a business perspective, this analysis answers critical questions such as:

* Is user engagement improving over time?
* Are increases in activity translating into real revenue?
* Which product segments drive growth versus just traffic?
* Are new customers returning and generating long-term value?

This mirrors real-world scenarios where growth teams must balance acquisition scale with sustainable customer value.

---

## Key Questions This Analysis Answers

* How efficient is the view → cart → purchase funnel?
* Did conversion performance improve month over month?
* Which price segments generate revenue versus just engagement?
* Which product categories are driving growth?
* How does customer retention behave over time?
* What is the observed lifetime value (LTV) of different cohorts?

---

## Dataset & Tooling

**Dataset**
Public Kaggle dataset: *E-commerce Behavior Data from Multi-Category Store*
~110 million raw event records

**Technology Stack**

* Python
* DuckDB (large-scale analytical processing)
* Pandas
* Matplotlib & Seaborn (visualization)
* Streamlit (interactive dashboard – final delivery)

To ensure performance and scalability, raw CSV files were converted into optimized Parquet format and queried using DuckDB.

---

## Data Modeling & Performance Decisions

To handle the size of the dataset efficiently, the following modeling steps were applied:

* Conversion of raw CSV files to Parquet for efficient storage and querying
* Use of DuckDB to perform analytical queries directly on Parquet files
* Creation of enriched event views with:

  * `event_date`
  * `event_month`
  * `event_week`
* Separation of raw and processed data to maintain a clean project structure

These choices reflect real-world data engineering considerations within analytics teams.

---

## Funnel Analysis

The core user journey analyzed is:

**View → Cart → Purchase**

### Key Findings

* **View-to-cart conversion doubled** from October to November
  This indicates stronger product engagement and increased purchase intent.

* **Cart-to-purchase conversion remained relatively stable**, suggesting checkout performance did not improve at the same pace as engagement.

* **Overall purchase conversion (view-to-purchase) increased only slightly**, meaning higher engagement did not fully translate into proportional purchase growth.

This suggests improvements in product discovery or merchandising, but remaining friction at the final conversion stage.

---

## Revenue & Price Segment Insights

Revenue analysis reveals a clear concentration of value:

* **Revenue increased significantly month-over-month**
* **High-priced products represent the vast majority of total revenue**
* Low-priced items generate traffic and engagement but contribute minimally to overall revenue

This indicates that growth in November was largely driven by stronger performance in premium product segments rather than broad-based conversion across all price tiers.

---

## Retention & Cohort Analysis

Customer retention was analyzed using weekly purchase cohorts.

### Key Findings

* **Strong early repeat behavior**: ~15–20% of customers return within the first week after their initial purchase
* **Sharp drop after week 2**, suggesting limited long-term engagement
* **More recent cohorts show lower early retention**, which may indicate:

  * Lower acquisition quality
  * Higher price sensitivity
  * Growth driven by broader, less targeted traffic

These patterns highlight opportunities for lifecycle marketing and post-purchase engagement strategies.

---

## Observed LTV (Lifetime Value)

An observed LTV proxy was calculated based on total revenue per user within each cohort.

Key observation:

* Older cohorts show higher observed LTV due to longer activity windows
* Newer cohorts display lower early value, reinforcing the idea that recent growth may be volume-driven rather than value-driven

This emphasizes the importance of balancing acquisition scale with long-term customer value.

---

## Key Takeaways

* Product engagement improved significantly in November
* Revenue growth was primarily driven by high-priced product segments
* Increased funnel activity did not fully translate into proportional purchase growth
* Early retention is strong, but long-term retention drops quickly
* Recent customer cohorts may have lower long-term value

These findings suggest that while acquisition and engagement improved, further optimization is needed in conversion and retention to maximize sustainable growth.

---

## Live Dashboard (Coming Next)

An interactive Streamlit dashboard will accompany this project, allowing:

* Filtering by month and segment
* Visualization of funnel performance
* Exploration of retention behavior
* Revenue and LTV breakdowns

---

## Project Structure

```
growth-funnel-analytics/
│
├── data/
│   ├── raw/          # raw dataset (ignored)
│   └── processed/    # parquet files
│
├── notebooks/
│   ├── 01_data_foundation.ipynb
│   ├── 02_growth_funnel.ipynb
│   ├── 03_retention_and_cohorts.ipynb
│   ├── 04_revenue_ltv_insights.ipynb
│
├── app/              # Streamlit dashboard (coming)
├── README.md
```

---

## Next Steps

Possible extensions for future iterations:

* Attribution modeling across traffic sources
* Predictive LTV modeling
* A/B test simulation and experimentation design
* Integration with product feature usage data

---

This project demonstrates end-to-end product analytics thinking (from user behavior and funnel performance to revenue and retention) with a focus on turning data into actionable business insights.
