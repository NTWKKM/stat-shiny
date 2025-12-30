"""
🎨 Shiny UI Styling Module

Provides CSS utilities, styled components, and helper functions for consistent
UI styling across all Shiny modules using the professional Navy Blue theme.

Usage:
    from tabs._styling import get_shiny_css, style_card, style_button
    
    ui.HTML(get_shiny_css())
"""

from tabs._common import get_color_palette


def get_shiny_css():
    """
    Returns global CSS for Shiny app styling with professional Navy Blue theme.
    
    Usage:
        In your Shiny app UI:
        ui.tags.head(ui.HTML(get_shiny_css()))
    """
    COLORS = get_color_palette()
    
    css = f"""
    <style>
        /* ===========================
           GLOBAL STYLES
           =========================== */
        
        :root {{
            --color-primary: {COLORS['primary']};
            --color-primary-dark: {COLORS['primary_dark']};
            --color-primary-light: {COLORS['primary_light']};
            --color-success: {COLORS['success']};
            --color-danger: {COLORS['danger']};
            --color-warning: {COLORS['warning']};
            --color-info: {COLORS['info']};
            --color-text: {COLORS['text']};
            --color-text-secondary: {COLORS['text_secondary']};
            --color-border: {COLORS['border']};
            --color-background: {COLORS['background']};
            --color-surface: {COLORS['surface']};
        }}
        
        * {{
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', 'Cantarell', sans-serif;
            background-color: {COLORS['background']};
            color: {COLORS['text']};
        }}
        
        /* ===========================
           SHINY CARDS
           =========================== */
        
        .bslib-card {{
            border: 1px solid {COLORS['border']};
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08), 0 2px 6px rgba(0, 0, 0, 0.05);
            transition: box-shadow 0.2s ease;
        }}
        
        .bslib-card:hover {{
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.12), 0 8px 16px rgba(0, 0, 0, 0.08);
        }}
        
        .bslib-card-header {{
            background-color: {COLORS['primary_light']};
            border-bottom: 2px solid {COLORS['primary']};
            font-weight: 600;
            color: {COLORS['primary_dark']};
            padding: 14px 16px;
        }}
        
        .bslib-card-body {{
            padding: 16px;
        }}
        
        /* ===========================
           BUTTONS
           =========================== */
        
        .btn {{
            border-radius: 6px;
            font-weight: 500;
            transition: all 0.2s ease;
            border: none;
        }}
        
        .btn-primary {{
            background-color: {COLORS['primary']};
            border-color: {COLORS['primary']};
            color: white;
        }}
        
        .btn-primary:hover {{
            background-color: {COLORS['primary_dark']};
            border-color: {COLORS['primary_dark']};
            box-shadow: 0 4px 8px rgba(30, 58, 95, 0.3);
        }}
        
        .btn-primary:focus,
        .btn-primary:active {{
            background-color: {COLORS['primary_dark']};
            border-color: {COLORS['primary_dark']};
            outline: 2px solid {COLORS['primary_light']};
            outline-offset: 2px;
        }}
        
        .btn-success {{
            background-color: {COLORS['success']};
            border-color: {COLORS['success']};
            color: white;
        }}
        
        .btn-success:hover {{
            filter: brightness(0.9);
            box-shadow: 0 4px 8px rgba(34, 167, 101, 0.3);
        }}
        
        .btn-danger {{
            background-color: {COLORS['danger']};
            border-color: {COLORS['danger']};
            color: white;
        }}
        
        .btn-danger:hover {{
            filter: brightness(0.9);
            box-shadow: 0 4px 8px rgba(231, 72, 86, 0.3);
        }}
        
        .btn-warning {{
            background-color: {COLORS['warning']};
            border-color: {COLORS['warning']};
            color: #000;
        }}
        
        .btn-warning:hover {{
            filter: brightness(0.9);
            box-shadow: 0 4px 8px rgba(255, 185, 0, 0.3);
        }}
        
        /* ===========================
           FORM INPUTS
           =========================== */
        
        .form-control {{
            border: 1px solid {COLORS['border']};
            border-radius: 6px;
            padding: 10px 12px;
            font-size: 14px;
            transition: border-color 0.2s ease, box-shadow 0.2s ease;
        }}
        
        .form-control:focus {{
            border-color: {COLORS['primary']};
            box-shadow: 0 0 0 3px rgba(30, 58, 95, 0.15);
            outline: none;
        }}
        
        .form-select {{
            border: 1px solid {COLORS['border']};
            border-radius: 6px;
            padding: 10px 12px;
            font-size: 14px;
        }}
        
        .form-select:focus {{
            border-color: {COLORS['primary']};
            box-shadow: 0 0 0 3px rgba(30, 58, 95, 0.15);
            outline: none;
        }}
        
        .form-label {{
            font-weight: 500;
            color: {COLORS['text']};
            margin-bottom: 6px;
            font-size: 13px;
        }}
        
        /* ===========================
           NAVIGATION & TABS
           =========================== */
        
        .navbar {{
            background-color: {COLORS['primary_dark']} !important;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }}
        
        .navbar-brand {{
            color: white !important;
            font-weight: 600;
            font-size: 18px;
        }}
        
        /* NAVBAR NAV-LINKS: White text (not faded) */
        .nav-link {{
            color: white !important;
            font-weight: 500;
            transition: color 0.2s ease;
        }}
        
        .nav-link:hover,
        .nav-link.active {{
            color: white !important;
            background-color: rgba(255, 255, 255, 0.15) !important;
            border-radius: 4px;
        }}
        
        /* ===========================
           MAIN TABS (Top Navigation)
           =========================== */
        
        .nav-tabs {{
            border-bottom: 2px solid {COLORS['border']};
        }}
        
        .nav-tabs .nav-link {{
            color: {COLORS['text_secondary']};
            border: none;
            border-bottom: 3px solid transparent;
            font-weight: 500;
        }}
        
        .nav-tabs .nav-link:hover {{
            border-bottom-color: {COLORS['primary_light']};
            color: {COLORS['primary']};
        }}
        
        .nav-tabs .nav-link.active {{
            color: {COLORS['primary']};
            background-color: transparent;
            border-bottom-color: {COLORS['primary']};
        }}
        
        /* ===========================
           SUBTABS (Inside Tabs)
           TEXT COLOR FIX - Ensure visibility on both light and dark themes
           =========================== */
        
        /* Default state: inactive subtabs */
        .nav-item .nav-link[role="tab"]:not(.active) {{
            /* Light theme: medium gray, Dark theme: lighter gray */
            color: {COLORS['text_secondary']} !important;
        }}
        
        /* Hover state: inactive subtabs */
        .nav-item .nav-link[role="tab"]:not(.active):hover {{
            /* Light theme: navy tint, Dark theme: lighter navy */
            color: {COLORS['primary']} !important;
            border-bottom-color: {COLORS['primary_light']};
        }}
        
        /* Active state: always visible */
        .nav-item .nav-link[role="tab"].active {{
            /* Light theme: dark navy, Dark theme: light navy */
            color: {COLORS['primary']} !important;
            border-bottom-color: {COLORS['primary']};
        }}
        
        /* ===========================
           ALERTS & NOTIFICATIONS
           =========================== */
        
        .alert {{
            border-radius: 6px;
            border: 1px solid transparent;
            padding: 12px 16px;
        }}
        
        .alert-success {{
            background-color: rgba(34, 167, 101, 0.1);
            border-color: {COLORS['success']};
            color: {COLORS['success']};
        }}
        
        .alert-danger {{
            background-color: rgba(231, 72, 86, 0.1);
            border-color: {COLORS['danger']};
            color: {COLORS['danger']};
        }}
        
        .alert-warning {{
            background-color: rgba(255, 185, 0, 0.1);
            border-color: {COLORS['warning']};
            color: #000;
        }}
        
        .alert-info {{
            background-color: rgba(30, 58, 95, 0.1);
            border-color: {COLORS['primary']};
            color: {COLORS['primary_dark']};
        }}
        
        /* ===========================
           TABLES
           =========================== */
        
        .table {{
            font-size: 13px;
            border-collapse: collapse;
        }}
        
        .table thead th {{
            background-color: {COLORS['primary_dark']};
            color: white;
            border-color: {COLORS['primary']};
            font-weight: 600;
            padding: 12px;
        }}
        
        .table tbody td {{
            border-color: {COLORS['border']};
            padding: 10px 12px;
            vertical-align: middle;
        }}
        
        .table tbody tr:nth-child(even) {{
            background-color: {COLORS['primary_light']};
        }}
        
        .table tbody tr:hover {{
            background-color: {COLORS['primary_light']};
        }}
        
        /* ===========================
           UTILITIES
           =========================== */
        
        .text-primary {{
            color: {COLORS['primary']} !important;
        }}
        
        .text-success {{
            color: {COLORS['success']} !important;
        }}
        
        .text-danger {{
            color: {COLORS['danger']} !important;
        }}
        
        .text-warning {{
            color: {COLORS['warning']} !important;
        }}
        
        .text-info {{
            color: {COLORS['info']} !important;
        }}
        
        .bg-primary-light {{
            background-color: {COLORS['primary_light']} !important;
        }}
        
        .bg-surface {{
            background-color: {COLORS['surface']} !important;
        }}
        
        .border-primary {{
            border: 1px solid {COLORS['primary']} !important;
            border-radius: 6px;
        }}
        
        /* ===========================
           BADGES & INDICATORS
           =========================== */
        
        .badge {{
            border-radius: 12px;
            padding: 4px 12px;
            font-size: 12px;
            font-weight: 600;
        }}
        
        .badge-primary {{
            background-color: {COLORS['primary']};
            color: white;
        }}
        
        .badge-success {{
            background-color: {COLORS['success']};
            color: white;
        }}
        
        .badge-danger {{
            background-color: {COLORS['danger']};
            color: white;
        }}
        
        .badge-warning {{
            background-color: {COLORS['warning']};
            color: #000;
        }}
        
        /* ===========================
           CUSTOM SHINY COMPONENTS
           =========================== */
        
        .status-badge {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 6px 12px;
            border-radius: 6px;
            font-weight: 500;
            font-size: 13px;
        }}
        
        .status-badge.matched {{
            background-color: rgba(34, 167, 101, 0.15);
            color: {COLORS['success']};
            border: 1px solid {COLORS['success']};
        }}
        
        .status-badge.unmatched {{
            background-color: rgba(231, 72, 86, 0.15);
            color: {COLORS['danger']};
            border: 1px solid {COLORS['danger']};
        }}
        
        /* ===========================
           RESPONSIVE DESIGN
           =========================== */
        
        @media (max-width: 768px) {{
            .bslib-card-body {{
                padding: 12px;
            }}
            
            .btn {{
                font-size: 13px;
                padding: 8px 12px;
            }}
            
            .form-control,
            .form-select {{
                font-size: 16px; /* Prevents zoom on iOS */
            }}
        }}
    </style>
    """
    
    return css


def style_card_header(title: str, icon: str = "") -> str:
    """
    Generate styled card header HTML.
    
    Args:
        title: Header text
        icon: Optional emoji or icon
        
    Returns:
        HTML string for card header
    """
    COLORS = get_color_palette()
    return f"""
    <div style="
        background-color: {COLORS['primary_light']};
        border-bottom: 2px solid {COLORS['primary']};
        padding: 14px 16px;
        font-weight: 600;
        color: {COLORS['primary_dark']};
        border-radius: 6px 6px 0 0;
    ">
        {icon} {title}
    </div>
    """


def style_status_badge(status: str, text: str) -> str:
    """
    Generate styled status badge.
    
    Args:
        status: 'success', 'danger', 'warning', 'info'
        text: Badge text
        
    Returns:
        HTML string for status badge
    """
    COLORS = get_color_palette()
    color_map = {
        'success': COLORS['success'],
        'danger': COLORS['danger'],
        'warning': COLORS['warning'],
        'info': COLORS['info'],
    }
    color = color_map.get(status, COLORS['info'])
    opacity = '0.15' if status != 'warning' else '0.1'
    text_color = '#000' if status == 'warning' else color
    
    return f"""
    <span style="
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 12px;
        border-radius: 6px;
        background-color: rgba({color[1:].upper()}, {opacity});
        color: {text_color};
        border: 1px solid {color};
        font-weight: 600;
        font-size: 13px;
    ">
        {text}
    </span>
    """


def style_alert(alert_type: str, message: str, title: str = "") -> str:
    """
    Generate styled alert/notification box.
    
    Args:
        alert_type: 'success', 'danger', 'warning', 'info'
        message: Alert message
        title: Optional alert title
        
    Returns:
        HTML string for alert
    """
    COLORS = get_color_palette()
    color_map = {
        'success': COLORS['success'],
        'danger': COLORS['danger'],
        'warning': COLORS['warning'],
        'info': COLORS['primary'],
    }
    color = color_map.get(alert_type, COLORS['info'])
    icon_map = {
        'success': '✅',
        'danger': '❌',
        'warning': '⚠️',
        'info': 'ℹ️',
    }
    icon = icon_map.get(alert_type, '')
    
    title_html = f"<strong>{title}</strong><br>" if title else ""
    
    return f"""
    <div style="
        background-color: rgba({color[1:]}, 0.1);
        border: 1px solid {color};
        border-radius: 6px;
        padding: 12px 16px;
        color: {color};
    ">
        {icon} {title_html}{message}
    </div>
    """


def get_color_code(color_name: str) -> str:
    """
    Get hex color code by name.
    
    Args:
        color_name: Color name (e.g., 'primary', 'success')
        
    Returns:
        Hex color code
    """
    colors = get_color_palette()
    return colors.get(color_name, '#1E3A5F')
