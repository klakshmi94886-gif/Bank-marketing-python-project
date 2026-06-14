# Bank Marketing Campaign — Predictive Pipeline & EDA Report

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![Data Integrity](https://img.shields.io/badge/Data%20Integrity-Validated-success.svg)](#2-data-inspection--structural-integrity)
[![ML Pipeline](https://img.shields.io/badge/Pipeline-Imbalance--Corrected-orange.svg)](#5-recommended-preprocessing-pipeline)

This repository contains the data analytics artifacts, exploratory profiles, and the automated preprocessing pipeline for the `bankmarketing.csv` dataset. The primary goal of this framework is to isolate key customer demographics and build an operational predictive workflow to determine whether a client will subscribe to a long-term financial product (term deposit).

---

## 1. Executive Summary
A diagnostic inspection of the banking dataset highlights a fundamental mathematical hurdle: a sharp skewness toward non-subscription (`y = "no"`). 

Because native metrics like generalized accuracy will fail (yielding deceptively high scores on an unoptimized model), this project implements class-independent evaluation frameworks optimizing for **Precision**, **Recall**, and **Area Under the Precision-Recall Curve (AUPRC)**.

---

## 2. Data Inspection & Structural Integrity
The dataset contains **17 active monitoring columns** across distinct schemas:

### Data Type Schema Mapping
* **Continuous Numeric Variables:** `age`, `balance`, `day`, `duration`, `campaign`, `pdays`, `previous`. Stored natively as 64-bit integers or floats.
* **Categorical Text Features:** `job`, `marital`, `education`, `default`, `housing`, `loan`, `contact`, `month`, `poutcome`. Stored as standard system objects.

### Null-Value & Integrity Traps
A programmatic scan (`df.isnull().sum()`) confirms **zero explicit missing numerical cells**. However, missing data is masked contextually as `"unknown"` strings inside categorical fields like `job`, `education`, and `poutcome`.

---

## 3. Key Data Insights & Structural Findings

### Descriptive Profiles
* **Extreme Financial Variances:** The `balance` metric exhibits massive economic spreads, requiring mandatory standardization due to the deep delta between maximum assets and negative balances.
* **Multi-Collinearity Redundancy:** Pearson product-moment correlation analysis isolated high linear dependencies between sequential context features, specifically `pdays` and `previous`. These linear redundancies require feature modification during operational engineering.

### Categorical Schema Matrix
| Feature Name | Cardinality | Strategic Evaluation Implication |
| :--- | :--- | :--- |
| `job` | Multi-class | Determines structural occupation tier target bands. |
| `marital` | 3 Categories | Evaluates household responsibility correlation. |
| `housing` | Binary (Yes/No) | Identifies existing debt/liability overhead. |

---

## 4. Recommended Preprocessing Pipeline
To successfully transition this analytical EDA blueprint into an operational machine learning model, the preprocessing architecture executes a strict four-step modular sequence:
