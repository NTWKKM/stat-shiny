from playwright.sync_api import Page, expect
from shiny.pytest import create_app_fixture

# ชี้ไปที่ app.py ของคุณ
app = create_app_fixture("../../app.py")

def test_app_loads(page: Page, app):
    """ทดสอบว่าเปิดเว็บขึ้น และ Title ถูกต้อง"""
    page.goto(app.url)
    expect(page).to_have_title("Medical Stat Tool")

def test_navigation_to_survival(page: Page, app):
    """ทดสอบว่าคลิกเปลี่ยน Tab ได้"""
    page.goto(app.url)
    
    # คลิก Tab Survival Analysis
    # (ต้องมั่นใจว่าใน UI ตั้งชื่อตรงกัน หรือใช้ data-value)
    page.get_by_role("tab", name="Survival Analysis").click()
    
    # ตรวจสอบว่ามี element เฉพาะของหน้านั้นโผล่มาไหม
    # Fixed: use .first to resolve strict mode error (matches 4 elements)
    expect(page.get_by_text("Kaplan-Meier").first).to_be_visible()
