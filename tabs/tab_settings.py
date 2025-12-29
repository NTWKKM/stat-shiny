"""Settings Tab for Shiny Application

Provides interactive configuration management for statistical analysis parameters,
UI settings, logging configuration, and performance tuning.

Usage:
    from tabs import tab_settings
    
    # In app.py
    ui.nav_panel("⚙️ Settings", tab_settings.settings_ui("settings"))
    
    # In server function
    tab_settings.settings_server("settings", CONFIG)
"""

from shiny import ui, reactive, render
from config import CONFIG
from logger import get_logger

logger = get_logger(__name__)


def settings_ui(id: str) -> ui.TagChild:
    """
    Create the UI for settings tab with configuration controls.
    
    Parameters:
        id (str): Shiny module ID for namespacing
    
    Returns:
        ui.TagChild: Settings UI layout with tabs and controls
    """
    return ui.navset_tab(
        # ==========================================
        # 1. ANALYSIS SETTINGS TAB
        # ==========================================
        ui.nav_panel(
            "📊 Analysis",
            ui.layout_sidebar(
                ui.sidebar(
                    ui.h5("Analysis Parameters"),
                    
                    # Logistic Regression Section
                    ui.h6("🔹 Logistic Regression"),
                    ui.input_select(
                        f"{id}-logit_method",
                        "Method",
                        choices={"auto": "Auto", "firth": "Firth", "bfgs": "BFGS", "default": "Default"},
                        selected=CONFIG.get('analysis.logit_method'),
                        width="100%"
                    ),
                    ui.input_slider(
                        f"{id}-logit_screening_p",
                        "Screening P-value",
                        min=0.0, max=1.0, value=CONFIG.get('analysis.logit_screening_p'),
                        step=0.01, width="100%"
                    ),
                    ui.input_numeric(
                        f"{id}-logit_max_iter",
                        "Max Iterations",
                        value=CONFIG.get('analysis.logit_max_iter'),
                        min=10, max=5000, width="100%"
                    ),
                    ui.input_numeric(
                        f"{id}-logit_min_cases",
                        "Min Cases for Multivariate",
                        value=CONFIG.get('analysis.logit_min_cases'),
                        min=1, max=100, width="100%"
                    ),
                    
                    ui.br(),
                    
                    # Survival Analysis Section
                    ui.h6("🔹 Survival Analysis"),
                    ui.input_select(
                        f"{id}-survival_method",
                        "Survival Method",
                        choices={"kaplan-meier": "Kaplan-Meier", "weibull": "Weibull"},
                        selected=CONFIG.get('analysis.survival_method'),
                        width="100%"
                    ),
                    ui.input_select(
                        f"{id}-cox_method",
                        "Cox Method (Tie Handling)",
                        choices={"efron": "Efron", "breslow": "Breslow"},
                        selected=CONFIG.get('analysis.cox_method'),
                        width="100%"
                    ),
                    
                    ui.br(),
                    
                    # Variable Detection Section
                    ui.h6("🔹 Variable Detection"),
                    ui.input_numeric(
                        f"{id}-var_detect_threshold",
                        "Unique Value Threshold",
                        value=CONFIG.get('analysis.var_detect_threshold'),
                        min=1, max=50, width="100%"
                    ),
                    ui.input_slider(
                        f"{id}-var_detect_decimal_pct",
                        "Decimal % Threshold",
                        min=0.0, max=1.0, value=CONFIG.get('analysis.var_detect_decimal_pct'),
                        step=0.05, width="100%"
                    ),
                    
                    ui.br(),
                    
                    # P-value Formatting Section
                    ui.h6("🔹 P-value Bounds (NEJM)"),
                    ui.input_slider(
                        f"{id}-pvalue_bounds_lower",
                        "Lower Bound",
                        min=0.0, max=0.1, value=CONFIG.get('analysis.pvalue_bounds_lower'),
                        step=0.001, width="100%"
                    ),
                    ui.input_slider(
                        f"{id}-pvalue_bounds_upper",
                        "Upper Bound",
                        min=0.9, max=1.0, value=CONFIG.get('analysis.pvalue_bounds_upper'),
                        step=0.001, width="100%"
                    ),
                    ui.input_text(
                        f"{id}-pvalue_format_small",
                        "Small P Format",
                        value=CONFIG.get('analysis.pvalue_format_small'),
                        width="100%"
                    ),
                    ui.input_text(
                        f"{id}-pvalue_format_large",
                        "Large P Format",
                        value=CONFIG.get('analysis.pvalue_format_large'),
                        width="100%"
                    ),
                    
                    ui.br(),
                    
                    # Missing Data Section
                    ui.h6("🔹 Missing Data"),
                    ui.input_select(
                        f"{id}-missing_strategy",
                        "Missing Data Strategy",
                        choices={"complete-case": "Complete-case", "drop": "Drop"},
                        selected=CONFIG.get('analysis.missing_strategy'),
                        width="100%"
                    ),
                    ui.input_numeric(
                        f"{id}-missing_threshold_pct",
                        "Missing Flag Threshold (%)",
                        value=CONFIG.get('analysis.missing_threshold_pct'),
                        min=0, max=100, width="100%"
                    ),
                    
                    ui.input_action_button(f"{id}-btn_save_analysis", "💾 Save Analysis Settings",
                                          class_="btn-primary", width="100%"),
                    
                    width=300,
                    bg="#f8f9fa"
                ),
                
                ui.card(
                    ui.card_header("📋 Analysis Settings Guide"),
                    ui.markdown("""
                    ### Logistic Regression
                    - **Auto**: Best for beginners
                    - **Firth**: Stable for small samples or rare events
                    - **Screening P**: 0.05-0.20 typical range
                    
                    ### Survival Analysis
                    - **Kaplan-Meier**: Non-parametric (recommended)
                    - **Efron**: Better for tied event times
                    
                    ### P-value Format (NEJM Standard)
                    - Lower: 0.001 (display as \"<0.001\")
                    - Upper: 0.999 (display as \">0.999\")
                    """),
                    full_screen=True
                )
            )
        ),
        
        # ==========================================
        # 2. UI & DISPLAY TAB
        # ==========================================
        ui.nav_panel(
            "🎨 UI & Display",
            ui.layout_sidebar(
                ui.sidebar(
                    ui.h5("Display Settings"),
                    
                    # Page Setup
                    ui.h6("🔹 Page Setup"),
                    ui.input_text(
                        f"{id}-page_title",
                        "Page Title",
                        value=CONFIG.get('ui.page_title'),
                        width="100%"
                    ),
                    ui.input_select(
                        f"{id}-theme",
                        "Theme",
                        choices={"light": "Light", "dark": "Dark", "auto": "Auto"},
                        selected=CONFIG.get('ui.theme'),
                        width="100%"
                    ),
                    ui.input_select(
                        f"{id}-layout",
                        "Layout",
                        choices={"wide": "Wide", "centered": "Centered"},
                        selected=CONFIG.get('ui.layout'),
                        width="100%"
                    ),
                    
                    ui.br(),
                    
                    # Tables
                    ui.h6("🔹 Tables"),
                    ui.input_numeric(
                        f"{id}-table_max_rows",
                        "Max Table Rows",
                        value=CONFIG.get('ui.table_max_rows'),
                        min=10, max=10000, width="100%"
                    ),
                    ui.input_checkbox(
                        f"{id}-table_pagination",
                        "Enable Pagination",
                        value=CONFIG.get('ui.table_pagination'),
                    ),
                    ui.input_numeric(
                        f"{id}-table_decimal_places",
                        "Decimal Places",
                        value=CONFIG.get('ui.table_decimal_places'),
                        min=0, max=10, width="100%"
                    ),
                    
                    ui.br(),
                    
                    # Plots
                    ui.h6("🔹 Plots"),
                    ui.input_numeric(
                        f"{id}-plot_width",
                        "Plot Width (inches)",
                        value=CONFIG.get('ui.plot_width'),
                        min=5, max=50, width="100%"
                    ),
                    ui.input_numeric(
                        f"{id}-plot_height",
                        "Plot Height (inches)",
                        value=CONFIG.get('ui.plot_height'),
                        min=3, max=30, width="100%"
                    ),
                    ui.input_numeric(
                        f"{id}-plot_dpi",
                        "Plot DPI",
                        value=CONFIG.get('ui.plot_dpi'),
                        min=50, max=600, width="100%"
                    ),
                    ui.input_text(
                        f"{id}-plot_style",
                        "Plot Style",
                        value=CONFIG.get('ui.plot_style'),
                        width="100%"
                    ),
                    
                    ui.input_action_button(f"{id}-btn_save_ui", "💾 Save UI Settings",
                                          class_="btn-primary", width="100%"),
                    
                    width=300,
                    bg="#f8f9fa"
                ),
                
                ui.card(
                    ui.card_header("🎯 UI Settings Guide"),
                    ui.markdown("""
                    ### Recommended Settings
                    - **Plot Width**: 10-14 inches
                    - **Plot Height**: 5-8 inches
                    - **Plot DPI**: 100-300 (higher = sharper)
                    - **Theme**: Auto follows system preference
                    - **Decimal Places**: 3 for most stats
                    """),
                    full_screen=True
                )
            )
        ),
        
        # ==========================================
        # 3. LOGGING TAB
        # ==========================================
        ui.nav_panel(
            "📝 Logging",
            ui.layout_sidebar(
                ui.sidebar(
                    ui.h5("Logging Configuration"),
                    
                    # Global Settings
                    ui.h6("🔹 Global"),
                    ui.input_checkbox(
                        f"{id}-logging_enabled",
                        "Enable Logging",
                        value=CONFIG.get('logging.enabled'),
                    ),
                    ui.input_select(
                        f"{id}-logging_level",
                        "Log Level",
                        choices={"DEBUG": "DEBUG", "INFO": "INFO", "WARNING": "WARNING",
                                "ERROR": "ERROR", "CRITICAL": "CRITICAL"},
                        selected=CONFIG.get('logging.level'),
                        width="100%"
                    ),
                    
                    ui.br(),
                    
                    # File Logging
                    ui.h6("🔹 File Logging"),
                    ui.input_checkbox(
                        f"{id}-file_enabled",
                        "Enable File Logging",
                        value=CONFIG.get('logging.file_enabled'),
                    ),
                    ui.input_text(
                        f"{id}-log_dir",
                        "Log Directory",
                        value=CONFIG.get('logging.log_dir'),
                        width="100%"
                    ),
                    ui.input_text(
                        f"{id}-log_file",
                        "Log Filename",
                        value=CONFIG.get('logging.log_file'),
                        width="100%"
                    ),
                    
                    ui.br(),
                    
                    # Console Logging
                    ui.h6("🔹 Console"),
                    ui.input_checkbox(
                        f"{id}-console_enabled",
                        "Enable Console Logging",
                        value=CONFIG.get('logging.console_enabled'),
                    ),
                    ui.input_select(
                        f"{id}-console_level",
                        "Console Level",
                        choices={"DEBUG": "DEBUG", "INFO": "INFO", "WARNING": "WARNING",
                                "ERROR": "ERROR", "CRITICAL": "CRITICAL"},
                        selected=CONFIG.get('logging.console_level'),
                        width="100%"
                    ),
                    
                    ui.br(),
                    
                    # Event Logging
                    ui.h6("🔹 Log Events"),
                    ui.input_checkbox(
                        f"{id}-log_file_ops",
                        "File Operations",
                        value=CONFIG.get('logging.log_file_operations'),
                    ),
                    ui.input_checkbox(
                        f"{id}-log_data_ops",
                        "Data Operations",
                        value=CONFIG.get('logging.log_data_operations'),
                    ),
                    ui.input_checkbox(
                        f"{id}-log_analysis_ops",
                        "Analysis Operations",
                        value=CONFIG.get('logging.log_analysis_operations'),
                    ),
                    ui.input_checkbox(
                        f"{id}-log_performance",
                        "Performance Timing",
                        value=CONFIG.get('logging.log_performance'),
                    ),
                    
                    ui.input_action_button(f"{id}-btn_save_logging", "💾 Save Logging Settings",
                                          class_="btn-primary", width="100%"),
                    
                    width=300,
                    bg="#f8f9fa"
                ),
                
                ui.card(
                    ui.card_header("📊 Logging Status"),
                    ui.output_text(f"{id}-txt_logging_status"),
                    full_screen=True
                )
            )
        ),
        
        # ==========================================
        # 4. PERFORMANCE TAB
        # ==========================================
        ui.nav_panel(
            "⚡ Performance",
            ui.layout_sidebar(
                ui.sidebar(
                    ui.h5("Performance Tuning"),
                    
                    ui.input_checkbox(
                        f"{id}-caching_enabled",
                        "Enable Caching",
                        value=CONFIG.get('performance.enable_caching'),
                    ),
                    ui.input_numeric(
                        f"{id}-cache_ttl",
                        "Cache TTL (seconds)",
                        value=CONFIG.get('performance.cache_ttl'),
                        min=60, max=86400, width="100%"
                    ),
                    
                    ui.br(),
                    
                    ui.input_checkbox(
                        f"{id}-compression_enabled",
                        "Enable Compression",
                        value=CONFIG.get('performance.enable_compression'),
                    ),
                    ui.input_numeric(
                        f"{id}-num_threads",
                        "Number of Threads",
                        value=CONFIG.get('performance.num_threads'),
                        min=1, max=32, width="100%"
                    ),
                    
                    ui.input_action_button(f"{id}-btn_save_perf", "💾 Save Performance Settings",
                                          class_="btn-primary", width="100%"),
                    
                    width=300,
                    bg="#f8f9fa"
                ),
                
                ui.card(
                    ui.card_header("ℹ️ Performance Guide"),
                    ui.markdown("""
                    ### Caching
                    - **TTL**: How long to keep cached results (seconds)
                    - **Typical**: 3600 (1 hour)
                    
                    ### Threading
                    - **Threads**: CPU cores available for parallel processing
                    - **Typical**: Set to # of CPU cores
                    
                    ### Compression
                    - Reduces memory usage for large datasets
                    - May slightly increase CPU usage
                    """),
                    full_screen=True
                )
            )
        ),
        
        # ==========================================
        # 5. ADVANCED TAB
        # ==========================================
        ui.nav_panel(
            "🛠️ Advanced",
            ui.layout_sidebar(
                ui.sidebar(
                    ui.h5("Advanced Settings"),
                    
                    # Validation
                    ui.h6("🔹 Validation"),
                    ui.input_checkbox(
                        f"{id}-strict_mode",
                        "Strict Mode",
                        value=CONFIG.get('validation.strict_mode'),
                    ),
                    ui.input_checkbox(
                        f"{id}-validate_inputs",
                        "Validate Inputs",
                        value=CONFIG.get('validation.validate_inputs'),
                    ),
                    ui.input_checkbox(
                        f"{id}-validate_outputs",
                        "Validate Outputs",
                        value=CONFIG.get('validation.validate_outputs'),
                    ),
                    ui.input_checkbox(
                        f"{id}-auto_fix_errors",
                        "Auto-fix Errors",
                        value=CONFIG.get('validation.auto_fix_errors'),
                    ),
                    
                    ui.br(),
                    
                    # Debug
                    ui.h6("🔹 Debug"),
                    ui.input_checkbox(
                        f"{id}-debug_enabled",
                        "Enable Debug Mode",
                        value=CONFIG.get('debug.enabled'),
                    ),
                    ui.input_checkbox(
                        f"{id}-debug_verbose",
                        "Verbose Output",
                        value=CONFIG.get('debug.verbose'),
                    ),
                    ui.input_checkbox(
                        f"{id}-profile_performance",
                        "Profile Performance",
                        value=CONFIG.get('debug.profile_performance'),
                    ),
                    ui.input_checkbox(
                        f"{id}-show_timings",
                        "Show Timings",
                        value=CONFIG.get('debug.show_timings'),
                    ),
                    
                    ui.input_action_button(f"{id}-btn_save_advanced", "💾 Save Advanced Settings",
                                          class_="btn-primary", width="100%"),
                    
                    width=300,
                    bg="#f8f9fa"
                ),
                
                ui.card(
                    ui.card_header("⚠️ Advanced Options"),
                    ui.markdown("""
                    ### Validation
                    - **Strict**: Stop on validation errors
                    - **Validate**: Check data types and formats
                    
                    ### Debug
                    - **Debug Mode**: Show detailed debugging info
                    - **Verbose**: Print intermediate steps
                    - **Profile**: Measure CPU/memory usage
                    """),
                    full_screen=True
                )
            )
        ),
        
        id=f"{id}-tabs"
    )


def settings_server(id: str, config) -> None:
    """
    Server logic for settings tab with reactive configuration updates.
    
    Parameters:
        id (str): Shiny module ID matching UI
        config: CONFIG object for reading/updating settings
    
    Returns:
        None
    """
    from shiny import input as shiny_input
    
    @render.text
    def txt_logging_status() -> str:
        """Display current logging configuration status."""
        enabled = config.get('logging.enabled')
        level = config.get('logging.level')
        status = "✅ ENABLED" if enabled else "❌ DISABLED"
        return f"Logging Status: {status}\nLevel: {level}"
    
    # ==========================================
    # ANALYSIS SETTINGS SAVE
    # ==========================================
    @reactive.Effect
    @reactive.event(shiny_input[f"{id}-btn_save_analysis"])
    def _save_analysis_settings():
        """Save analysis settings when button clicked."""
        try:
            config.update('analysis.logit_method', shiny_input[f"{id}-logit_method"]())
            config.update('analysis.logit_screening_p', float(shiny_input[f"{id}-logit_screening_p"]()))
            config.update('analysis.logit_max_iter', int(shiny_input[f"{id}-logit_max_iter"]()))
            config.update('analysis.logit_min_cases', int(shiny_input[f"{id}-logit_min_cases"]()))
            config.update('analysis.survival_method', shiny_input[f"{id}-survival_method"]())
            config.update('analysis.cox_method', shiny_input[f"{id}-cox_method"]())
            config.update('analysis.var_detect_threshold', int(shiny_input[f"{id}-var_detect_threshold"]()))
            config.update('analysis.var_detect_decimal_pct', float(shiny_input[f"{id}-var_detect_decimal_pct"]()))
            config.update('analysis.pvalue_bounds_lower', float(shiny_input[f"{id}-pvalue_bounds_lower"]()))
            config.update('analysis.pvalue_bounds_upper', float(shiny_input[f"{id}-pvalue_bounds_upper"]()))
            config.update('analysis.pvalue_format_small', shiny_input[f"{id}-pvalue_format_small"]())
            config.update('analysis.pvalue_format_large', shiny_input[f"{id}-pvalue_format_large"]())
            config.update('analysis.missing_strategy', shiny_input[f"{id}-missing_strategy"]())
            config.update('analysis.missing_threshold_pct', int(shiny_input[f"{id}-missing_threshold_pct"]()))
            
            logger.info("✅ Analysis settings saved")
            ui.notification_show("✅ Analysis settings saved", type="message")
        except Exception as e:
            logger.error(f"Error saving analysis settings: {e}")
            ui.notification_show(f"❌ Error: {e}", type="error")
    
    # ==========================================
    # UI SETTINGS SAVE
    # ==========================================
    @reactive.Effect
    @reactive.event(shiny_input[f"{id}-btn_save_ui"])
    def _save_ui_settings():
        """Save UI settings when button clicked."""
        try:
            config.update('ui.page_title', shiny_input[f"{id}-page_title"]())
            config.update('ui.theme', shiny_input[f"{id}-theme"]())
            config.update('ui.layout', shiny_input[f"{id}-layout"]())
            config.update('ui.table_max_rows', int(shiny_input[f"{id}-table_max_rows"]()))
            config.update('ui.table_pagination', bool(shiny_input[f"{id}-table_pagination"]()))
            config.update('ui.table_decimal_places', int(shiny_input[f"{id}-table_decimal_places"]()))
            config.update('ui.plot_width', int(shiny_input[f"{id}-plot_width"]()))
            config.update('ui.plot_height', int(shiny_input[f"{id}-plot_height"]()))
            config.update('ui.plot_dpi', int(shiny_input[f"{id}-plot_dpi"]()))
            config.update('ui.plot_style', shiny_input[f"{id}-plot_style"]())
            
            logger.info("✅ UI settings saved")
            ui.notification_show("✅ UI settings saved", type="message")
        except Exception as e:
            logger.error(f"Error saving UI settings: {e}")
            ui.notification_show(f"❌ Error: {e}", type="error")
    
    # ==========================================
    # LOGGING SETTINGS SAVE
    # ==========================================
    @reactive.Effect
    @reactive.event(shiny_input[f"{id}-btn_save_logging"])
    def _save_logging_settings():
        """Save logging settings when button clicked."""
        try:
            config.update('logging.enabled', bool(shiny_input[f"{id}-logging_enabled"]()))
            config.update('logging.level', shiny_input[f"{id}-logging_level"]())
            config.update('logging.file_enabled', bool(shiny_input[f"{id}-file_enabled"]()))
            config.update('logging.log_dir', shiny_input[f"{id}-log_dir"]())
            config.update('logging.log_file', shiny_input[f"{id}-log_file"]())
            config.update('logging.console_enabled', bool(shiny_input[f"{id}-console_enabled"]()))
            config.update('logging.console_level', shiny_input[f"{id}-console_level"]())
            config.update('logging.log_file_operations', bool(shiny_input[f"{id}-log_file_ops"]()))
            config.update('logging.log_data_operations', bool(shiny_input[f"{id}-log_data_ops"]()))
            config.update('logging.log_analysis_operations', bool(shiny_input[f"{id}-log_analysis_ops"]()))
            config.update('logging.log_performance', bool(shiny_input[f"{id}-log_performance"]()))
            
            logger.info("✅ Logging settings saved")
            ui.notification_show("✅ Logging settings saved", type="message")
        except Exception as e:
            logger.error(f"Error saving logging settings: {e}")
            ui.notification_show(f"❌ Error: {e}", type="error")
    
    # ==========================================
    # PERFORMANCE SETTINGS SAVE
    # ==========================================
    @reactive.Effect
    @reactive.event(shiny_input[f"{id}-btn_save_perf"])
    def _save_perf_settings():
        """Save performance settings when button clicked."""
        try:
            config.update('performance.enable_caching', bool(shiny_input[f"{id}-caching_enabled"]()))
            config.update('performance.cache_ttl', int(shiny_input[f"{id}-cache_ttl"]()))
            config.update('performance.enable_compression', bool(shiny_input[f"{id}-compression_enabled"]()))
            config.update('performance.num_threads', int(shiny_input[f"{id}-num_threads"]()))
            
            logger.info("✅ Performance settings saved")
            ui.notification_show("✅ Performance settings saved", type="message")
        except Exception as e:
            logger.error(f"Error saving performance settings: {e}")
            ui.notification_show(f"❌ Error: {e}", type="error")
    
    # ==========================================
    # ADVANCED SETTINGS SAVE
    # ==========================================
    @reactive.Effect
    @reactive.event(shiny_input[f"{id}-btn_save_advanced"])
    def _save_advanced_settings():
        """Save advanced settings when button clicked."""
        try:
            config.update('validation.strict_mode', bool(shiny_input[f"{id}-strict_mode"]()))
            config.update('validation.validate_inputs', bool(shiny_input[f"{id}-validate_inputs"]()))
            config.update('validation.validate_outputs', bool(shiny_input[f"{id}-validate_outputs"]()))
            config.update('validation.auto_fix_errors', bool(shiny_input[f"{id}-auto_fix_errors"]()))
            config.update('debug.enabled', bool(shiny_input[f"{id}-debug_enabled"]()))
            config.update('debug.verbose', bool(shiny_input[f"{id}-debug_verbose"]()))
            config.update('debug.profile_performance', bool(shiny_input[f"{id}-profile_performance"]()))
            config.update('debug.show_timings', bool(shiny_input[f"{id}-show_timings"]()))
            
            logger.info("✅ Advanced settings saved")
            ui.notification_show("✅ Advanced settings saved", type="message")
        except Exception as e:
            logger.error(f"Error saving advanced settings: {e}")
            ui.notification_show(f"❌ Error: {e}", type="error")
