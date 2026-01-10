# Multiple Comparison Corrections (Bonferroni, Holm, BH)

    - ใช้กับการรันหลาย test พร้อมกัน เช่น:
        - หลาย landmark times
        - หลาย subgroup / interaction
    - แนะนำ 2 แบบ:

**1.1 Survival / Landmark / Subgroup tab**
    - ถ้าคุณมี tab พวก:
        - `Survival Analysis`
        - `Landmark Analysis`
        - `Subgroup / Interaction Analysis`
    - ใส่ option เช่น:
        - “Multiple comparison correction: [ None | Bonferroni | Holm | BH ]”
    - แล้วแสดงผล corrected p‑value ใน summary table ของ tab นั้นเลย

**1.2 Global “Advanced Statistics” / “Sensitivity” tab** (ถ้าคุณอยากรวม)
    - สำหรับงานที่ user รันหลายการวิเคราะห์หลาย outcome/group พร้อมกัน
    - ทำเป็น tab ระดับสูง เช่น:
        - `Advanced / Multiple Testing`
    - ให้ user เลือกชุด p-values ที่จะ apply correction

***

## สรุปสั้น ๆ วางใน stat-shiny แบบนี้น่าจะลงตัวที่สุด

- **Multiple Comparison Corrections** →
  - ถ้าผูกกับ survival/landmark → อยู่ใน **Survival / Landmark / Subgroup tab**
  - ถ้าทำ generic correction tool → เพิ่ม **Advanced / Multiple Testing tab** แยก
