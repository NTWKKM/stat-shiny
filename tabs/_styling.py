"""
🎨 Shiny UI Styling Module - Professional Navy Blue Medical Analytics Theme

Provides comprehensive CSS utilities, styled components, and helper functions for consistent
UI styling across all Shiny modules using a professional Navy Blue theme with enhanced
visual hierarchy, spacing, and modern design patterns.

Features:
- Professional Navy Blue color palette with medical-grade aesthetics
- Enhanced card designs with better shadows and hover effects
- Modern button styles with improved accessibility
- Refined form inputs with better focus states
- Improved spacing and typography hierarchy
- Responsive design optimizations
- Better status indicators and badges
- Enhanced table styling

Usage:
    from tabs._styling import get_shiny_css, style_card_header, style_status_badge, style_alert

    ui.HTML(get_shiny_css())
"""

from tabs._common import get_color_palette


def get_shiny_css():
    """
    Provide the full CSS stylesheet implementing the Navy Blue theme for Shiny apps.
    
    Includes root variables (colors, spacing, radii, shadows, transitions, typography), Bootstrap-compatible component styles (.card, .card-header, .btn, .form-control, .nav-tabs, etc.), utility classes, custom components, and responsive adjustments.
    
    Returns:
        css (str): An HTML string containing a <style>...</style> block with the complete CSS for the theme.
    """
    COLORS = get_color_palette()

    css = f"""
    <style>
        /* ===========================
           CSS VARIABLES & ROOT STYLES
           =========================== */
        
        :root {{
            --color-primary: {COLORS['primary']};
            --color-primary-dark: {COLORS['primary_dark']};
            --color-primary-light: {COLORS['primary_light']};
            --color-smoke-white: {COLORS['smoke_white']};
            --color-success: {COLORS['success']};
            --color-danger: {COLORS['danger']};
            --color-warning: {COLORS['warning']};
            --color-info: {COLORS['info']};
            --color-neutral: {COLORS['neutral']};
            --color-text: {COLORS['text']};
            --color-text-secondary: {COLORS['text_secondary']};
            --color-border: {COLORS['border']};
            --color-background: {COLORS['background']};
            --color-surface: {COLORS['surface']};
            
            /* Spacing System */
            --spacing-xs: 4px;
            --spacing-sm: 8px;
            --spacing-md: 12px;
            --spacing-lg: 16px;
            --spacing-xl: 24px;
            --spacing-2xl: 32px;
            
            /* Border Radius */
            --radius-sm: 4px;
            --radius-md: 6px;
            --radius-lg: 8px;
            --radius-xl: 12px;
            
            /* Shadows */
            --shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.05);
            --shadow-md: 0 2px 6px rgba(0, 0, 0, 0.08), 0 1px 3px rgba(0, 0, 0, 0.04);
            --shadow-lg: 0 4px 12px rgba(0, 0, 0, 0.12), 0 8px 16px rgba(0, 0, 0, 0.06);
            --shadow-xl: 0 8px 24px rgba(0, 0, 0, 0.15);
            
            /* Transitions */
            --transition-fast: 150ms cubic-bezier(0.4, 0, 0.2, 1);
            --transition-normal: 250ms cubic-bezier(0.4, 0, 0.2, 1);
            
            /* Typography */
            --font-family-base: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', 'Cantarell', sans-serif;
            --font-family-mono: 'Courier New', monospace;
        }}
        
        * {{
            box-sizing: border-box;
        }}
        
        .report-footer {{
            text-align: center;
            padding: var(--spacing-xl) 0;
            margin-top: var(--spacing-2xl);
            border-top: 1px solid {COLORS['border']};
            color: {COLORS['text_secondary']};
            font-size: 13px;
            background-color: {COLORS['surface']};
        }}
        
        .report-footer a {{
            color: {COLORS['primary']};
            text-decoration: none;
            font-weight: 600;
        }}
        
        .report-footer a:hover {{
            text-decoration: underline;
        }}
        
        /* ===========================
           LAYOUT & CONTAINER
           =========================== */
        
        /* Main app container - centered and max-width on larger screens */
        .app-container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 16px 24px 32px;
        }}
        
        /* Add top margin to app-container when it follows navbar */
        .navbar ~ .app-container {{
            margin-top: 8px;
        }}
        
        /* Responsive padding on mobile */
        @media (max-width: 768px) {{
            .app-container {{
                padding: 12px 16px 24px;
            }}
        }}
        
        /* ===========================
           GLOBAL STYLES
           =========================== */
        
        html {{
            font-size: 14px;
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
        }}
        
        body {{
            font-family: var(--font-family-base);
            background-color: {COLORS['background']};
            color: {COLORS['text']};
            line-height: 1.6;
            margin: 0;
            padding: 0;
        }}
        
        /* ===========================
           TYPOGRAPHY
           =========================== */
        
        h1 {{
            font-size: 28px;
            font-weight: 700;
            margin: var(--spacing-lg) 0;
            color: {COLORS['primary_dark']};
            letter-spacing: -0.5px;
        }}
        
        h2 {{
            font-size: 24px;
            font-weight: 600;
            margin: var(--spacing-lg) 0 var(--spacing-md) 0;
            color: {COLORS['primary_dark']};
            letter-spacing: -0.3px;
        }}
        
        h3 {{
            font-size: 20px;
            font-weight: 600;
            margin: var(--spacing-md) 0;
            color: {COLORS['primary']};
        }}
        
        h4 {{
            font-size: 17px;
            font-weight: 600;
            margin: var(--spacing-md) 0;
            color: {COLORS['primary']};
        }}
        
        h5 {{
            font-size: 15px;
            font-weight: 600;
            margin: var(--spacing-sm) 0;
            color: {COLORS['text']};
        }}
        
        h6 {{
            font-size: 13px;
            font-weight: 600;
            margin: var(--spacing-sm) 0;
            color: {COLORS['text_secondary']};
        }}
        
        p {{
            margin: 0 0 var(--spacing-md) 0;
            font-size: 14px;
            line-height: 1.6;
        }}
        
        a {{
            color: {COLORS['primary']};
            text-decoration: none;
            transition: color var(--transition-fast);
            font-weight: 500;
        }}
        
        a:hover {{
            color: {COLORS['primary_dark']};
            text-decoration: underline;
        }}
        
        /* ===========================
           SHINY CARDS & CONTAINERS
           (Updated to standard Bootstrap .card classes)
           =========================== */
        
        .card {{
            border: 1px solid {COLORS['border']};
            border-radius: var(--radius-lg);
            box-shadow: var(--shadow-md);
            transition: all var(--transition-normal);
            background-color: {COLORS['surface']};
            overflow: hidden;
            margin-bottom: 12px;  /* Reduced vertical spacing between cards */
        }}
        
        .card:hover {{
            box-shadow: var(--shadow-lg);
            border-color: {COLORS['primary_light']};
        }}
        
        .card-header {{
            background: linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['primary_dark']} 100%);
            border: none;
            border-bottom: 2px solid {COLORS['primary_dark']};
            font-weight: 600;
            color: white;
            padding: var(--spacing-lg) var(--spacing-lg);
            font-size: 15px;
            text-transform: none;
        }}
        
        .card-body {{
            padding: 14px 16px;  /* Slightly tighter padding */
            line-height: 1.6;
        }}
        
        .card-footer {{
            background-color: {COLORS['background']};
            border-top: 1px solid {COLORS['border']};
            padding: var(--spacing-md) var(--spacing-lg);
        }}
        
        /* ===========================
           BUTTONS
           =========================== */
        
        .btn {{
            border: none;
            border-radius: var(--radius-md);
            font-weight: 600;
            font-size: 13px;
            padding: 10px 16px;
            transition: all var(--transition-fast);
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: var(--spacing-sm);
            white-space: nowrap;
            user-select: none;
        }}
        
        .btn:focus {{
            outline: 2px solid {COLORS['primary']};
            outline-offset: 2px;
        }}
        
        .btn:disabled {{
            opacity: 0.5;
            cursor: not-allowed;
        }}
        
        /* Primary Buttons */
        .btn-primary {{
            background: linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['primary_dark']} 100%);
            color: white;
            box-shadow: 0 2px 4px rgba(30, 58, 95, 0.2);
        }}
        
        .btn-primary:hover:not(:disabled) {{
            background: linear-gradient(135deg, {COLORS['primary_dark']} 0%, {COLORS['primary_dark']} 100%);
            box-shadow: 0 4px 12px rgba(30, 58, 95, 0.35);
            transform: translateY(-1px);
        }}
        
        .btn-primary:active:not(:disabled) {{
            transform: translateY(0);
            box-shadow: 0 2px 4px rgba(30, 58, 95, 0.25);
        }}
        
        /* Secondary Buttons */
        .btn-secondary {{
            background-color: {COLORS['primary_light']};
            color: {COLORS['primary_dark']};
            border: 1px solid {COLORS['primary']};
        }}
        
        .btn-secondary:hover:not(:disabled) {{
            background-color: {COLORS['primary']};
            color: white;
            box-shadow: 0 2px 8px rgba(30, 58, 95, 0.2);
        }}
        
        .btn-secondary:active:not(:disabled) {{
            background-color: {COLORS['primary_dark']};
        }}
        
        /* Success Buttons */
        .btn-success {{
            background-color: {COLORS['success']};
            color: white;
            box-shadow: 0 2px 4px rgba(34, 167, 101, 0.2);
        }}
        
        .btn-success:hover:not(:disabled) {{
            filter: brightness(0.9);
            box-shadow: 0 4px 12px rgba(34, 167, 101, 0.35);
            transform: translateY(-1px);
        }}
        
        /* Danger Buttons */
        .btn-danger {{
            background-color: {COLORS['danger']};
            color: white;
            box-shadow: 0 2px 4px rgba(231, 72, 86, 0.2);
        }}
        
        .btn-danger:hover:not(:disabled) {{
            filter: brightness(0.9);
            box-shadow: 0 4px 12px rgba(231, 72, 86, 0.35);
            transform: translateY(-1px);
        }}
        
        /* Warning Buttons */
        .btn-warning {{
            background-color: {COLORS['warning']};
            color: #000;
            box-shadow: 0 2px 4px rgba(255, 185, 0, 0.2);
        }}
        
        .btn-warning:hover:not(:disabled) {{
            filter: brightness(0.95);
            box-shadow: 0 4px 12px rgba(255, 185, 0, 0.3);
            transform: translateY(-1px);
        }}
        
        /* Outline Buttons */
        .btn-outline-primary {{
            border: 2px solid {COLORS['primary']};
            color: {COLORS['primary']};
            background-color: transparent;
        }}
        
        .btn-outline-primary:hover:not(:disabled) {{
            background-color: {COLORS['primary_light']};
        }}
        
        /* ===========================
           FORM INPUTS
           =========================== */
        
        .form-control {{
            border: 1px solid {COLORS['border']};
            border-radius: var(--radius-md);
            padding: var(--spacing-sm) var(--spacing-md);
            font-size: 14px;
            font-family: var(--font-family-base);
            background-color: {COLORS['surface']};
            color: {COLORS['text']};
            transition: all var(--transition-fast);
            line-height: 1.5;
        }}
        
        .form-control:focus {{
            border-color: {COLORS['primary']};
            background-color: {COLORS['surface']};
            box-shadow: 0 0 0 3px rgba(30, 58, 95, 0.1);
            outline: none;
        }}
        
        .form-control::placeholder {{
            color: {COLORS['text_secondary']};
            opacity: 0.7;
        }}
        
        .form-control:disabled {{
            background-color: {COLORS['background']};
            opacity: 0.6;
            cursor: not-allowed;
        }}
        
        /* Select/Dropdown */
        .form-select {{
            border: 1px solid {COLORS['border']};
            border-radius: var(--radius-md);
            padding: var(--spacing-sm) var(--spacing-md);
            font-size: 14px;
            background-color: {COLORS['surface']};
            color: {COLORS['text']};
            transition: all var(--transition-fast);
            cursor: pointer;
            appearance: none;
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='{COLORS['primary_dark'].replace('#', '%23')}' stroke-width='2'%3E%3Cpolyline points='6 9 12 15 18 9'%3E%3C/polyline%3E%3C/svg%3E");
            background-repeat: no-repeat;
            background-position: right var(--spacing-md) center;
            background-size: 16px;
            padding-right: 36px;
        }}
        
        .form-select:focus {{
            border-color: {COLORS['primary']};
            box-shadow: 0 0 0 3px rgba(30, 58, 95, 0.1);
            outline: none;
        }}
        
        /* Form Label */
        .form-label {{
            font-weight: 600;
            font-size: 13px;
            color: {COLORS['text']};
            margin-bottom: var(--spacing-sm);
            display: block;
        }}
        
        .form-label.required::after {{
            content: ' *';
            color: {COLORS['danger']};
        }}
        
        /* Form Group */
        .form-group {{
            margin-bottom: var(--spacing-lg);
        }}
        
        /* Textarea */
        textarea.form-control {{
            resize: vertical;
            min-height: 100px;
            font-family: var(--font-family-base);
        }}
        
        /* ===========================
           NAVIGATION & NAVBAR
           =========================== */
        
        .navbar {{
            background-color: {COLORS['smoke_white']} !important;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
            border-bottom: 1px solid {COLORS['border']};
            padding: var(--spacing-md) var(--spacing-lg);
        }}
        
        .navbar-brand {{
            color: {COLORS['primary_dark']} !important;
            font-weight: 700;
            font-size: 18px;
            letter-spacing: -0.5px;
        }}
        
        /* Navbar Text & Links */
        .navbar .nav-link {{
            color: {COLORS['text_secondary']} !important;
            font-weight: 500;
            font-size: 14px;
            padding: var(--spacing-sm) var(--spacing-md) !important;
            margin: 0 var(--spacing-sm);
            border-radius: var(--radius-sm);
            transition: all var(--transition-fast);
        }}
        
        .navbar .nav-link:hover {{
            color: {COLORS['primary']} !important;
            background-color: rgba(30, 58, 95, 0.08);
        }}
        
        .navbar .nav-link.active {{
            color: {COLORS['primary_dark']} !important;
            background-color: {COLORS['primary_light']};
        }}
        
        /* Sidebar Navigation */
        .sidebar .nav-link,
        .bslib-sidebar-layout .sidebar-content .nav-link {{
            color: rgba(255, 255, 255, 0.85) !important;
            font-weight: 500;
            padding: var(--spacing-md) var(--spacing-lg) !important;
            border-radius: var(--radius-sm);
            transition: all var(--transition-fast);
            margin: var(--spacing-xs) 0;
        }}
        
        .sidebar .nav-link:hover,
        .sidebar .nav-link.active,
        .bslib-sidebar-layout .sidebar-content .nav-link:hover,
        .bslib-sidebar-layout .sidebar-content .nav-link.active {{
            color: #ffffff !important;
            background-color: rgba(255, 255, 255, 0.15) !important;
        }}
        
        /* ===========================
           TABS
           =========================== */
        
        .nav-tabs {{
            border-bottom: 2px solid {COLORS['border']};
            margin-bottom: var(--spacing-lg);
        }}
        
        .nav-tabs .nav-link {{
            color: {COLORS['text_secondary']};
            border: none;
            border-bottom: 3px solid transparent;
            font-weight: 600;
            padding: var(--spacing-md) var(--spacing-lg);
            margin-bottom: -2px;
            transition: all var(--transition-fast);
            font-size: 14px;
        }}
        
        .nav-tabs .nav-link:hover {{
            border-bottom-color: {COLORS['primary_light']};
            color: {COLORS['primary']};
        }}
        
        .nav-tabs .nav-link.active {{
            color: {COLORS['primary_dark']};
            background-color: transparent;
            border-bottom-color: {COLORS['primary']};
        }}
        
        /* Subtabs */
        .nav-item .nav-link[role="tab"]:not(.active) {{
            color: {COLORS['text_secondary']} !important;
            font-weight: 500;
            padding: var(--spacing-sm) var(--spacing-md);
        }}
        
        .nav-item .nav-link[role="tab"]:not(.active):hover {{
            color: {COLORS['primary']} !important;
            background-color: {COLORS['primary_light']};
        }}
        
        .nav-item .nav-link[role="tab"].active {{
            color: {COLORS['primary_dark']} !important;
            background-color: {COLORS['primary_light']};
            font-weight: 600;
        }}
        
        /* ===========================
           ALERTS & NOTIFICATIONS
           =========================== */
        
        .alert {{
            border-radius: var(--radius-lg);
            border: 1px solid;
            padding: var(--spacing-md) var(--spacing-lg);
            margin-bottom: var(--spacing-lg);
            display: flex;
            align-items: flex-start;
            gap: var(--spacing-md);
        }}
        
        .alert-success {{
            background-color: rgba(34, 167, 101, 0.08);
            border-color: {COLORS['success']};
            color: {COLORS['success']};
        }}
        
        .alert-danger {{
            background-color: rgba(231, 72, 86, 0.08);
            border-color: {COLORS['danger']};
            color: {COLORS['danger']};
        }}
        
        .alert-warning {{
            background-color: rgba(255, 185, 0, 0.08);
            border-color: {COLORS['warning']};
            color: #856404;
        }}
        
        .alert-info {{
            background-color: rgba(30, 58, 95, 0.08);
            border-color: {COLORS['primary']};
            color: {COLORS['primary_dark']};
        }}
        
        /* ===========================
           TABLES
           =========================== */
        
        .table {{
            font-size: 13px;
            border-collapse: collapse;
            margin-bottom: var(--spacing-lg);
        }}
        
        .table thead th {{
            background: linear-gradient(135deg, {COLORS['primary_dark']} 0%, {COLORS['primary']} 100%);
            color: white;
            border: none;
            font-weight: 600;
            padding: var(--spacing-md);
            text-align: left;
            letter-spacing: 0.5px;
        }}
        
        .table tbody td {{
            border-bottom: 1px solid {COLORS['border']};
            padding: var(--spacing-md);
            vertical-align: middle;
        }}
        
        .table tbody tr {{
            transition: background-color var(--transition-fast);
        }}
        
        .table tbody tr:nth-child(even) {{
            background-color: {COLORS['background']};
        }}
        
        .table tbody tr:hover {{
            background-color: {COLORS['primary_light']};
        }}

        .sig-p {{
            color: #fff;
            background-color: {COLORS['danger']};
            padding: 2px 6px;
            border-radius: 3px;
            font-weight: 600;
        }}

        .shiny-table th, table.dataframe th {{
            background-color: {COLORS['background']} !important;
            color: {COLORS['text']} !important;
            font-weight: 600;
        }}
        
        /* Table Responsiveness - Enable horizontal scrolling on mobile */
        .table-responsive {{
            overflow-x: auto;
            -webkit-overflow-scrolling: touch;
            border-radius: var(--radius-md);
        }}
        
        /* Error Cell Highlighting - For data quality issues */
        .cell-error {{
            background-color: #fee !important;
            border: 1px solid {COLORS['danger']} !important;
            color: {COLORS['danger']} !important;
            font-weight: 600 !important;
        }}
        
        /* ===========================
           BADGES & STATUS INDICATORS
           =========================== */
        
        .badge {{
            display: inline-flex;
            align-items: center;
            gap: 4px;
            border-radius: 12px;
            padding: 4px 10px;
            font-size: 12px;
            font-weight: 600;
            white-space: nowrap;
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
        
        .badge-info {{
            background-color: {COLORS['info']};
            color: white;
        }}
        
        /* Status Badge (Custom) */
        .status-badge {{
            display: inline-flex;
            align-items: center;
            gap: var(--spacing-sm);
            padding: var(--spacing-sm) var(--spacing-md);
            border-radius: var(--radius-lg);
            font-weight: 600;
            font-size: 12px;
            border: 1px solid;
        }}
        
        .status-badge.matched {{
            background-color: rgba(34, 167, 101, 0.1);
            color: {COLORS['success']};
            border-color: {COLORS['success']};
        }}
        
        .status-badge.unmatched {{
            background-color: rgba(231, 72, 86, 0.1);
            color: {COLORS['danger']};
            border-color: {COLORS['danger']};
        }}
        
        .status-badge.warning {{
            background-color: rgba(255, 185, 0, 0.1);
            color: {COLORS['warning']};
            border-color: {COLORS['warning']};
        }}
        
        /* ===========================
           UTILITY CLASSES
           =========================== */
        
        .text-primary {{
            color: {COLORS['primary']} !important;
        }}
        
        .text-primary-dark {{
            color: {COLORS['primary_dark']} !important;
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
        
        .text-secondary {{
            color: {COLORS['text_secondary']} !important;
        }}
        
        .text-muted {{
            color: {COLORS['text_secondary']} !important;
            opacity: 0.8;
        }}
        
        .bg-primary {{
            background-color: {COLORS['primary']} !important;
            color: white;
        }}
        
        .bg-primary-light {{
            background-color: {COLORS['primary_light']} !important;
        }}
        
        .bg-surface {{
            background-color: {COLORS['surface']} !important;
        }}
        
        .bg-background {{
            background-color: {COLORS['background']} !important;
        }}
        
        .border-primary {{
            border: 1px solid {COLORS['primary']} !important;
            border-radius: var(--radius-md);
        }}
        
        .border-top {{
            border-top: 2px solid {COLORS['border']} !important;
            padding-top: var(--spacing-lg);
            margin-top: var(--spacing-lg);
        }}
        
        .divider {{
            height: 1px;
            background-color: {COLORS['border']};
            margin: var(--spacing-lg) 0;
        }}
        
        /* ===========================
           VIF SPECIFIC STYLES
           =========================== */
        
        .vif-container {{
            margin-top: var(--spacing-lg);
            padding: var(--spacing-md);
            background-color: var(--color-surface);
            border: 1px solid var(--color-border);
            border-radius: var(--radius-md);
            max-width: 500px;
        }}
        
        .vif-title {{
            font-size: 14px;
            font-weight: 600;
            color: var(--color-primary-dark);
            margin-bottom: var(--spacing-sm);
            display: flex;
            align-items: center;
            gap: 6px;
        }}
        
        .vif-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }}
        
        .vif-table th {{
            text-align: left;
            padding: var(--spacing-sm);
            background-color: var(--color-smoke-white);
            color: var(--color-text-secondary);
            font-weight: 600;
            border-bottom: 2px solid var(--color-border);
        }}
        
        .vif-table td {{
            padding: var(--spacing-sm);
            border-bottom: 1px solid var(--color-border);
        }}
        
        .vif-value {{
            font-family: var(--font-family-mono);
            font-weight: 600;
        }}
        
        .vif-warning {{
            color: var(--color-danger);
            font-weight: 700;
        }}
        
        .vif-caution {{
            color: var(--color-warning);
        }}
        
        .vif-footer {{
            margin-top: var(--spacing-sm);
            font-size: 11px;
            color: var(--color-text-secondary);
            font-style: italic;
        }}
        .mt-1 {{ margin-top: var(--spacing-xs); }}
        .mt-2 {{ margin-top: var(--spacing-sm); }}
        .mt-3 {{ margin-top: var(--spacing-md); }}
        .mt-4 {{ margin-top: var(--spacing-lg); }}
        .mt-5 {{ margin-top: var(--spacing-xl); }}
        
        .mb-1 {{ margin-bottom: var(--spacing-xs); }}
        .mb-2 {{ margin-bottom: var(--spacing-sm); }}
        .mb-3 {{ margin-bottom: var(--spacing-md); }}
        .mb-4 {{ margin-bottom: var(--spacing-lg); }}
        .mb-5 {{ margin-bottom: var(--spacing-xl); }}
        
        .p-2 {{ padding: var(--spacing-sm); }}
        .p-3 {{ padding: var(--spacing-md); }}
        .p-4 {{ padding: var(--spacing-lg); }}
        .p-5 {{ padding: var(--spacing-xl); }}
        
        /* ===========================
           CUSTOM COMPONENTS
           =========================== */
        
        .stat-box {{
            background-color: {COLORS['surface']};
            border: 1px solid {COLORS['border']};
            border-radius: var(--radius-lg);
            padding: var(--spacing-lg);
            text-align: center;
            box-shadow: var(--shadow-sm);
            transition: all var(--transition-normal);
        }}
        
        .stat-box:hover {{
            box-shadow: var(--shadow-md);
            border-color: {COLORS['primary_light']};
        }}
        
        .stat-box-label {{
            font-size: 12px;
            font-weight: 600;
            color: {COLORS['text_secondary']};
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: var(--spacing-sm);
        }}
        
        .stat-box-value {{
            font-size: 28px;
            font-weight: 700;
            color: {COLORS['primary_dark']};
            line-height: 1;
        }}
        
        .stat-box-subtext {{
            font-size: 12px;
            color: {COLORS['text_secondary']};
            margin-top: var(--spacing-sm);
        }}
        
        /* Info Box / Panel */
        .info-panel {{
            background-color: {COLORS['primary_light']};
            border-left: 4px solid {COLORS['primary']};
            border-radius: 0 var(--radius-md) var(--radius-md) 0;
            padding: var(--spacing-lg);
            margin-bottom: var(--spacing-lg);
        }}
        
        .info-panel.success {{
            background-color: rgba(34, 167, 101, 0.1);
            border-left-color: {COLORS['success']};
        }}
        
        .info-panel.danger {{
            background-color: rgba(231, 72, 86, 0.1);
            border-left-color: {COLORS['danger']};
        }}
        
        .info-panel.warning {{
            background-color: rgba(255, 185, 0, 0.1);
            border-left-color: {COLORS['warning']};
        }}
        
        /* Data Grid Container */
        .data-grid {{
            overflow-x: auto;
            border-radius: var(--radius-lg);
            box-shadow: var(--shadow-sm);
        }}
        
        /* ===========================
           MISSING DATA SECTION
           =========================== */
        
        .missing-data-section {{
            background-color: #f0f7ff;
            border-left: 4px solid {COLORS['primary']};
            padding: var(--spacing-lg);
            margin: var(--spacing-xl) 0;
            border-radius: 0 var(--radius-md) var(--radius-md) 0;
        }}
        
        .missing-data-section h4 {{
            color: {COLORS['primary_dark']};
            margin-top: 0;
            margin-bottom: var(--spacing-md);
        }}
        
        .missing-data-section h5 {{
            color: {COLORS['text']};
            margin-top: var(--spacing-lg);
            margin-bottom: var(--spacing-sm);
        }}
        
        .missing-table {{
            width: 100%;
            border-collapse: collapse;
            margin: var(--spacing-md) 0;
            font-size: 13px;
        }}
        
        .missing-table th,
        .missing-table td {{
            padding: var(--spacing-sm) var(--spacing-md);
            text-align: left;
            border-bottom: 1px solid {COLORS['border']};
        }}
        
        .missing-table th {{
            background-color: #e8f1ff;
            font-weight: 600;
            color: {COLORS['primary_dark']};
        }}
        
        .missing-table tr.high-missing {{
            background-color: #fff3cd;
        }}
        
        .missing-table tr.high-missing td {{
            color: #856404;
            font-weight: 500;
        }}
        
        .warning-box {{
            background-color: #fff3cd;
            border: 1px solid {COLORS['warning']};
            padding: var(--spacing-md);
            border-radius: var(--radius-md);
            margin-top: var(--spacing-md);
        }}
        
        .warning-box strong {{
            color: #856404;
        }}
        
        .warning-box ul {{
            margin: var(--spacing-sm) 0 0 var(--spacing-lg);
            padding: 0;
        }}
        
        .warning-box li {{
            color: #856404;
            margin-bottom: var(--spacing-xs);
        }}
        
        /* Missing config column styling */
        .variable-config-column {{
            padding-right: var(--spacing-md);
            border-right: 1px solid {COLORS['border']};
        }}
        
        .missing-config-column {{
            background-color: {COLORS['background']};
            padding: var(--spacing-md);
            border-radius: var(--radius-md);
        }}
        
        .missing-config-column h5,
        .missing-config-column h6 {{
            color: {COLORS['primary']};
            margin-bottom: var(--spacing-md);
        }}
        
        /* ===========================
           RESPONSIVENESS
           =========================== */
        
        @media (max-width: 768px) {{
            :root {{
                --spacing-lg: 12px;
                --spacing-xl: 16px;
            }}
            
            .card-body {{
                padding: 12px 14px;
            }}
            
            .btn {{
                font-size: 12px;
                padding: 8px 12px;
            }}
            
            .form-control,
            .form-select {{
                font-size: 16px; /* Prevents zoom on iOS */
                padding: 12px;
            }}
            
            h1 {{ font-size: 24px; }}
            h2 {{ font-size: 20px; }}
            h3 {{ font-size: 18px; }}
            
            .stat-box-value {{
                font-size: 24px;
            }}
            
            .table {{
                font-size: 12px;
            }}
            
            .table thead th,
            .table tbody td {{
                padding: var(--spacing-sm);
            }}
            
            .table-responsive {{
                margin-bottom: var(--spacing-md);
            }}
        }}
        
        @media (max-width: 480px) {{
            .navbar {{
                padding: var(--spacing-sm) var(--spacing-md);
            }}
            
            .navbar-brand {{
                font-size: 16px;
            }}
            
            .btn {{
                width: 100%;
                margin-bottom: var(--spacing-sm);
            }}
        }}

        /* ===========================
           TVC SPECIFIC STYLES
           =========================== */
        
        .tvc-preset-container {{
            margin-bottom: 10px;
        }}
        
        .tvc-preset-label {{
            margin-right: 10px; 
            font-weight: 600; 
            color: #555;
        }}
        
        .tvc-preset-btn {{
            margin-right: 5px !important;
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
        background: linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['primary_dark']} 100%);
        border-bottom: 2px solid {COLORS['primary_dark']};
        padding: 14px 16px;
        font-weight: 600;
        color: white;
        border-radius: 6px 6px 0 0;
        font-size: 15px;
    ">
        {icon} {title}
    </div>
    """


def _hex_to_rgb(hex_color: str) -> str:
    """Convert hex color to RGB values for rgba()."""
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        return "0, 0, 0"  # Fallback for invalid input
    try:
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    except ValueError:
        return "0, 0, 0"  # Fallback for non-hex characters
    return f"{r}, {g}, {b}"


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
        "success": COLORS["success"],
        "danger": COLORS["danger"],
        "warning": COLORS["warning"],
        "info": COLORS["info"],
    }
    color = color_map.get(status, COLORS["info"])
    text_color = "#000" if status == "warning" else color

    return f"""
    <span style="
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 12px;
        border-radius: 6px;
        background-color: rgba({_hex_to_rgb(color)}, 0.1);
        color: {text_color};
        border: 1px solid {color};
        font-weight: 600;
        font-size: 12px;
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
        "success": COLORS["success"],
        "danger": COLORS["danger"],
        "warning": COLORS["warning"],
        "info": COLORS["primary"],
    }
    color = color_map.get(alert_type, COLORS["info"])
    icon_map = {
        "success": "✅",
        "danger": "❌",
        "warning": "⚠️",
        "info": "ℹ️",
    }
    icon = icon_map.get(alert_type, "")

    title_html = f"<strong>{title}</strong><br>" if title else ""

    return f"""
    <div style="
        background-color: rgba({_hex_to_rgb(color)}, 0.1);
        border: 1px solid {color};
        border-radius: 6px;
        padding: 12px 16px;
        color: {color};
        margin-bottom: 16px;
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
    return colors.get(color_name, "#1E3A5F")