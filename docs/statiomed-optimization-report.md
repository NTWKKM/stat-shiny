# 📊 Medical Stat Tool - Master Optimization & Development Plan

**Document Version:** 4.0 (Integrated Logic & UI Edition)
**Date:** January 24, 2026
**Status:** 🚀 Production-Hardening (Validation & Optimization Phase)
**Target:** Enterprise-Grade / Medical Publication Standard

---

## 📑 Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Architecture & Health](#2-system-architecture--health)
3. [Module Optimization Roadmap (Updated)](#3-module-optimization-roadmap-updated)
4. [Deep Dive: Key Technical Implementations](#4-deep-dive-key-technical-implementations)
5. [Quality Assurance & R-Validation](#5-quality-assurance--r-validation)
6. [Deployment & DevOps](#6-deployment--devops)

---

## 1. Executive Summary

Medical Stat Tool (stat-shiny) ได้พัฒนาเข้าสู่ระยะที่มีความเสถียรสูง (High Stability) โครงสร้างปัจจุบันได้แยกส่วนการคำนวณทางสถิติ (Statistical Logic) ออกจากส่วนแสดงผล (UI) อย่างชัดเจน ตามหลักการ MVC Pattern

### 🎯 Strategic Focus (v4.0)

* **Logic Isolation:** แยก Logic ที่ซับซ้อน (เช่น Logistic Regression ใน `utils/logic.py`) ออกจาก UI files เพื่อให้ง่ายต่อการ Test และ Maintain
* **5-Tab Navigation Standard:** ยึดโครงสร้าง UI ใหม่ 5 Tabs (Data, General, Advanced, Clinical, Settings) เป็นมาตรฐานหลัก
* **Data Integrity:** ใช้ `utils/data_cleaning.py` เป็น Gatekeeper หลักในการจัดการ Missing Data และ Type Casting อย่างเข้มงวด
* **HTML-First Export:** ทุก Module ต้องสามารถออกรายงาน Single-File HTML ที่ฝัง Plotly Interactive Graph ได้สมบูรณ์ (`utils/plotly_html_renderer.py`)

---

## 2. System Architecture & Health

### 2.1 Hybrid Architecture (Shiny + Pure Python)

ระบบได้เปลี่ยนจาก Monolithic Shiny App มาเป็น **Modular Architecture** ที่สมบูรณ์:

```mermaid
graph TD
    User[User / Client] --> App[app.py / Navbar Shell (5 Tabs)]
    App --> Tabs[UI Components (tabs/*)]
    Tabs --> Helper[UI Helpers (utils/ui_helpers.py)]
    Tabs --> DataPipe[Data Cleaning Pipeline (utils/data_cleaning.py)]
    Tabs --> StatEngine[Pure Python Logic (utils/logic.py, diag_test.py)]
    StatEngine --> Libs[Statsmodels / Scikit-learn / Lifelines]
    StatEngine --> Renderer[Plotly HTML Renderer (utils/plotly_html_renderer.py)]

```

### 2.2 Critical Components Status

| Component | File Source | Status | Improvement Needed |
| --- | --- | --- | --- |
| **Data Pipeline** | `utils/data_cleaning.py` | 🟢 **Excellent** | Vectorized cleaning, Outlier detection, Quality reports ครบถ้วน |
| **Core Regression** | `utils/logic.py` | 🟢 **Good** | แยก Logic แล้ว รองรับ Firth/Logit, Interaction Terms, VIF |
| **Diagnostic UI** | `tabs/tab_diag.py` | 🟢 **Feature-Rich** | รองรับ ROC, DCA, Chi-Square พร้อม Download Report |
| **UI Structure** | `tabs/*` | 🟡 **Transitioning** | กำลังปรับเข้าสู่โครงสร้าง 5 Tabs ตามแผน `UI_IMPROVEMENT` |
| **Validation** | `tests/*` | 🟡 **In Progress** | มี E2E (`test_app_flow.py`) แล้ว แต่ต้องการ Statistical Validation เทียบกับ R เพิ่มเติม |

---

## 3. Module Optimization Roadmap (Updated)

### 🟢 PHASE 1: Architecture & Core Logic (Completed/Refining)

* **Objective:** แยก Business Logic ออกจาก UI และสร้าง Data Pipeline ที่แข็งแกร่ง
* **Achievements:**
* ✅ **Data Cleaning:** `utils/data_cleaning.py` รองรับการจัดการ Missing Values และ Outliers แบบ Vectorized
* ✅ **Regression Logic:** `utils/logic.py` รองรับการทำ Logistic Regression แบบ MVC, คำนวณ OR/AOR และ Interaction Terms ได้
* ✅ **Diagnostic Tool:** `tabs/tab_diag.py` สามารถรัน ROC/DCA และ Export HTML ได้จริง

### 🟡 PHASE 2: UI Standardization & Clinical Validation (Current Focus)

* **Objective:** ปรับ UI ให้เป็น 5 Tabs (ตามแผน UI Improvement) และตรวจสอบความถูกต้องทางสถิติเทียบกับ R
* **Action Items:**

#### A. UI Refactoring (Big 5 Restructure)

* [ ] **Merge Tabs:** ย้าย `tab_core_regression.py`, `tab_survival.py` เข้าไปอยู่ภายใต้ NavMenu "🔬 Advanced Statistics"
* [ ] **Code Reduction:** ใช้ `utils/ui_helpers.py` เพื่อลดโค้ดที่ซ้ำซ้อนในไฟล์ UI (เช่น Card Wrapper, Section Header)

#### B. Statistical Validation (Hardening)

* [ ] **Regression:** เพิ่ม Unit Test ตรวจสอบค่า OR/CI ของ `utils/logic.py` เทียบกับ output จาก R (glm)
* [ ] **Survival:** ตรวจสอบ Assumption Checks (Schoenfeld residuals) ใน `tab_survival.py`
* [ ] **Table 1:** ตรวจสอบการปัดเศษทศนิยม (Decimal Standardization) ให้ตรงตามมาตรฐานวารสารการแพทย์

### 🔴 PHASE 3: Advanced Features & Reporting (Next Steps)

* **Objective:** เพิ่มขีดความสามารถในการออกรายงานและการวิเคราะห์ขั้นสูง
* **Action Items:**

1. **Batch Report Generation:** สร้างปุ่ม "Generate All Reports" เพื่อรวมผลวิเคราะห์หลาย Module เป็น HTML ไฟล์เดียว
2. **AI Integration:** เตรียม Prompt Template สำหรับส่งผล Stats ไปให้ LLM ช่วยเขียนสรุปผล (Interpretation)
3. **Performance:** Implement Caching (`@functools.lru_cache` หรือ Shiny caching) สำหรับ dataset ขนาดใหญ่ (>50k rows)

---

## 4. Deep Dive: Key Technical Implementations

### 4.1 The Statistical Engine (`utils/logic.py`)

หัวใจสำคัญของการคำนวณที่แยกออกจาก UI รองรับการทำงานแบบ "Pure Python" ทำให้ Test ง่าย

```python
# ตัวอย่างโครงสร้างที่ใช้งานจริง
def run_binary_logit(y, X, method="default", ci_method="wald"):
    """
    Core function ที่ return raw params, conf_int, pvalues
    โดยไม่มี dependency กับ Shiny UI
    """
    # 1. Validation (via validate_logit_data)
    # 2. Method Selection (Firth vs Logit)
    # 3. Model Fitting (statsmodels)
    # 4. Return Dictionary/TypedDict

```

### 4.2 Robust Data Cleaning (`utils/data_cleaning.py`)

ระบบทำความสะอาดข้อมูลที่ออกแบบมาเพื่อ Medical Data โดยเฉพาะ:

* **Smart Numeric Conversion:** จัดการค่าติด Special Characters เช่น `"<5"`, `">100"`, `1,200` ได้อัตโนมัติ
* **Missing Data Strategy:** รองรับทั้ง `complete-case` และการระบุ `missing_codes` (เช่น -99, 999)
* **Audit Trail:** ทุกขั้นตอนการ Clean จะถูก Log และสามารถ generate report สรุปผลกระทบ (Data Loss) ได้

### 4.3 Embedded HTML Reports (`utils/plotly_html_renderer.py`)

เทคนิคการฝัง Plotly JS และ CSS ลงในไฟล์เดียว เพื่อให้รายงานเปิดได้ทุกที่โดยไม่ต้องต่อเน็ต (Offline-ready)

* ใช้ **CDN Injection** สำหรับ Bootstrap/MathJax เมื่อออนไลน์
* ใช้ **Base64 Encoding** สำหรับรูปภาพ static
* รองรับ **Responsive Design** สำหรับการเปิดบน iPad/Tablet

---

## 5. Quality Assurance & R-Validation

### 5.1 E2E Testing Strategy (`tests/e2e/test_app_flow.py`)

ปัจจุบันใช้ **Playwright** ในการทดสอบ User Flow:

* ✅ App Loading & Title Check
* ✅ Tab Navigation (ครบ 5 หมวดหมู่หลัก)
* ✅ File Upload Interaction
* ✅ Error Handling (Console Log Check)

### 5.2 Statistical Unit Tests Needed

ต้องเพิ่ม Test Suite เพื่อเทียบผลลัพธ์กับ R โดยเฉพาะ:

```python
# ตัวอย่างแผนการ Test ในอนาคต
def test_logistic_vs_r_results():
    # Load Benchmark Dataset (e.g., Titanic)
    py_res = run_binary_logit(y, X)
    r_res = load_r_benchmark("logistic_benchmark.csv")
    
    # Assert Coefficient match within tolerance 1e-5
    assert np.allclose(py_res['coef'], r_res['coef'], atol=1e-5)

```

---

## 6. Deployment & DevOps

### 6.1 Containerization

* **Docker:** ใช้งาน `Dockerfile` ที่ optimize แล้ว (Python 3.12-slim)
* **Environment:** แยก `requirements.txt` (Dev) และ `requirements-prod.txt` (Prod) เพื่อลดขนาด Image

### 6.2 Maintenance Protocol

1. **CSS Sync:** ห้ามแก้ `static/styles.css` โดยตรง ให้แก้ใน `tabs/_styling.py` แล้วรัน `utils/update_css.py`
2. **Repo Structure:** รักษาโครงสร้าง Folder ให้สะอาด ห้ามวางไฟล์ Python นอกเหนือจาก `app.py`, `config.py` ไว้ที่ Root โดยไม่จำเป็น

---
