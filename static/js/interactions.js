/**
 * ⚡ Client-Side Interaction Handlers
 * -----------------------------------
 * Manages user interactions, responsive layouts, ⌘K command palette,
 * clipboard copy, and accessible chart-table twin synchronization.
 */

$(document).ready(function () {
    // ==========================================
    // 1. CLIPBOARD HANDLER (Word / Rich Text)
    // ==========================================
    Shiny.addCustomMessageHandler("copy_rich_html", function (message) {
        if (navigator.clipboard && window.ClipboardItem) {
            var blobHtml = new Blob([message.html], { type: "text/html" });
            var blobText = new Blob([message.text || ""], { type: "text/plain" });
            var data = [new ClipboardItem({ "text/html": blobHtml, "text/plain": blobText })];
            navigator.clipboard.write(data).then(function () {
                if (typeof Shiny.setInputValue === "function") {
                    Shiny.setInputValue("copy_success_trigger", Date.now(), { priority: "event" });
                }
            }).catch(function (err) {
                console.error("Failed to copy table: ", err);
            });
        }
    });

    // ==========================================
    // 2. MOBILE MENU & RESPONSIVE RESIZE
    // ==========================================
    var mobileMenuInitialized = false;
    function initMobileMenu() {
        if (mobileMenuInitialized) return;

        function getSidebar() {
            var sidebar = $('#sidebar');
            if (!sidebar.length) {
                sidebar = $('.app-container .nav-pills').first().closest('[class*="col-"]');
            }
            return sidebar;
        }

        if ($(window).width() < 768) {
            getSidebar().hide();
        }

        $(document).on('click', '#mobile_menu_btn', function () {
            getSidebar().toggle();
        });

        // Debounced window & plotly resize handler
        var resizeTimer;
        $(window).resize(function () {
            clearTimeout(resizeTimer);
            resizeTimer = setTimeout(function () {
                var sidebar = getSidebar();
                if ($(window).width() >= 768) {
                    sidebar.show();
                } else {
                    sidebar.hide();
                }

                // Trigger Plotly relayout on resize
                if (window.Plotly) {
                    $('.js-plotly-plot').each(function () {
                        try {
                            Plotly.Plots.resize(this);
                        } catch (e) {}
                    });
                }
            }, 200);
        });

        mobileMenuInitialized = true;
    }

    // ==========================================
    // 3. NAVBAR BRAND HOME CLICK
    // ==========================================
    $(document).on('click', '#navbar_brand_home', function (e) {
        e.preventDefault();
        var homeLink = document.querySelector('.navbar-nav .nav-link[data-value="home"]');
        if (homeLink) {
            homeLink.click();
        }
    });

    // ==========================================
    // 4. COMMAND PALETTE (⌘K / Ctrl+K)
    // ==========================================
    var MODULE_REGISTRY = [
        { id: "home", title: "🏠 Home", category: "General", desc: "Overview, quick links, and platform features", keywords: "home dashboard start landing" },
        { id: "data", title: "📁 Data Management", category: "Intake & Prep", desc: "Import CSV/Excel, type casting, missing imputation (KNN/MICE)", keywords: "data import upload csv excel imputation clean mice outliers" },
        { id: "bm", title: "📋 Table 1 & Matching", category: "Baseline & Matching", desc: "Patient baseline characteristics, SMD, and Propensity Score Matching (PSM)", keywords: "table 1 baseline matching psm propensity score smd balance" },
        { id: "diagnostic", title: "📊 Diagnostic Tests", category: "General Statistics", desc: "Sensitivity, specificity, ROC/AUC curve, DeLong test, Likelihood ratios", keywords: "roc auc sensitivity specificity diagnostic accuracy ppv npv youden delong" },
        { id: "corr", title: "📈 Correlation Analysis", category: "General Statistics", desc: "Pearson, Spearman correlation matrix, scatterplots, heatmaps", keywords: "correlation pearson spearman heatmap scatter relationship" },
        { id: "agreement", title: "🤝 Agreement & Reliability", category: "General Statistics", desc: "Bland-Altman plot, Cohen's Kappa, Fleiss Kappa, Intraclass Correlation (ICC)", keywords: "agreement reliability bland altman kappa icc intraclass concordance" },
        { id: "adv_stats", title: "📉 Advanced Statistics", category: "General Statistics", desc: "Non-parametric tests (Mann-Whitney, Kruskal-Wallis, Friedman, ANOVA)", keywords: "anova kruskal wallis mann whitney wilcoxon friedman non parametric" },
        { id: "regression", title: "🔬 Multivariable Regression", category: "Advanced Modeling", desc: "Linear OLS, Logistic Binary, Firth Penalized Logistic, Odds Ratios, Forest Plot", keywords: "regression logistic linear firth odds ratio forest plot calibration dca" },
        { id: "survival", title: "⏱️ Survival Analysis", category: "Advanced Modeling", desc: "Kaplan-Meier curves, Log-Rank test, Cox Proportional Hazards, Schoenfeld residuals", keywords: "survival kaplan meier cox hazard ratio log rank time to event schoenfeld" },
        { id: "adv_regression", title: "🧬 Advanced Regression", category: "Advanced Modeling", desc: "Poisson, Negative Binomial, Linear Mixed Models (LMM), GEE, Bayesian MCMC", keywords: "poisson negative binomial lmm mixed models gee gam bayesian mcmc repeated" },
        { id: "meta_analysis", title: "📚 Meta-Analysis", category: "Advanced Modeling", desc: "Forest plot, Funnel plot, Fixed/Random effects, Egger test, Heterogeneity (I²)", keywords: "meta analysis forest funnel egger fixed random effect heterogeneity tau" },
        { id: "sample_size", title: "🏥 Sample Size Calculator", category: "Clinical Tools", desc: "Power estimation for two means, proportions, survival, and non-inferiority", keywords: "sample size power calculation clinical trial hypothesis test non inferiority" },
        { id: "causal", title: "🎯 Causal Inference", category: "Clinical Tools", desc: "Instrumental variables (2SLS), Marginal structural models, IPTW, Doubly Robust", keywords: "causal inference iv 2sls iptw doubly robust treatment effect confounding" },
        { id: "settings", title: "⚙️ System Settings", category: "System", desc: "Decimal precision, p-value thresholds, confidence levels, export formats", keywords: "settings config precision p value threshold preferences export" }
    ];

    var cmdActiveIndex = 0;
    var filteredModules = MODULE_REGISTRY.slice();
    var previouslyFocusedElement = null;

    function createCommandPaletteDOM() {
        if ($('#cmd_palette_backdrop').length) return;

        var html = [
            '<div id="cmd_palette_backdrop" class="cmd-palette-backdrop" role="dialog" aria-modal="true" aria-label="Command Palette">',
            '  <div class="cmd-palette-modal">',
            '    <div class="cmd-palette-header">',
            '      <span style="font-size: 16px;">🔍</span>',
            '      <input type="text" id="cmd_palette_input" class="cmd-palette-input" placeholder="Search modules, statistical tests, or estimators (e.g. firth, cox, psm)..." autocomplete="off" />',
            '      <span class="cmd-palette-badge">ESC</span>',
            '    </div>',
            '    <div id="cmd_palette_body" class="cmd-palette-body"></div>',
            '    <div class="cmd-palette-footer">',
            '      <span><kbd style="padding:2px 5px;background:#e2e8f0;border-radius:3px;">↑</kbd> <kbd style="padding:2px 5px;background:#e2e8f0;border-radius:3px;">↓</kbd> to navigate</span>',
            '      <span><kbd style="padding:2px 5px;background:#e2e8f0;border-radius:3px;">↵</kbd> to select</span>',
            '      <span><kbd style="padding:2px 5px;background:#e2e8f0;border-radius:3px;">ESC</kbd> to close</span>',
            '    </div>',
            '  </div>',
            '</div>'
        ].join('');

        $('body').append(html);
    }

    function renderCommandPaletteItems() {
        var $body = $('#cmd_palette_body');
        $body.empty();

        if (!filteredModules.length) {
            $body.html('<div style="padding: 24px; text-align: center; color: #64748b; font-size: 14px;">No matching statistical modules found</div>');
            return;
        }

        var currentCategory = "";
        filteredModules.forEach(function (item, index) {
            if (item.category !== currentCategory) {
                currentCategory = item.category;
                $body.append('<div class="cmd-palette-group-title">' + currentCategory + '</div>');
            }

            var isActive = (index === cmdActiveIndex) ? ' active' : '';
            var itemHtml = [
                '<div class="cmd-palette-item' + isActive + '" data-tab-id="' + item.id + '" data-index="' + index + '">',
                '  <div class="cmd-palette-item-left">',
                '    <span class="cmd-palette-item-title">' + item.title + '</span>',
                '    <span class="cmd-palette-item-desc">' + item.desc + '</span>',
                '  </div>',
                '  <span class="cmd-palette-badge" style="font-size: 10px;">Jump ↵</span>',
                '</div>'
            ].join('');

            $body.append(itemHtml);
        });

        // Scroll active item into view
        var $active = $body.find('.cmd-palette-item.active');
        if ($active.length) {
            var bodyTop = $body.scrollTop();
            var bodyHeight = $body.height();
            var activeTop = $active.position().top + bodyTop;
            var activeHeight = $active.outerHeight();

            if (activeTop < bodyTop) {
                $body.scrollTop(activeTop - 10);
            } else if (activeTop + activeHeight > bodyTop + bodyHeight) {
                $body.scrollTop(activeTop + activeHeight - bodyHeight + 10);
            }
        }
    }

    function openCommandPalette() {
        createCommandPaletteDOM();
        previouslyFocusedElement = document.activeElement;
        filteredModules = MODULE_REGISTRY.slice();
        cmdActiveIndex = 0;
        $('#cmd_palette_input').val('');
        renderCommandPaletteItems();
        $('#cmd_palette_backdrop').addClass('open');
        setTimeout(function () {
            $('#cmd_palette_input').focus();
        }, 50);
    }

    function closeCommandPalette() {
        var $backdrop = $('#cmd_palette_backdrop');
        $backdrop.removeClass('open');
        if (previouslyFocusedElement && typeof previouslyFocusedElement.focus === 'function') {
            previouslyFocusedElement.focus();
            previouslyFocusedElement = null;
        }
    }

    function activateTab(tabId) {
        closeCommandPalette();
        var link = document.querySelector('.navbar-nav .nav-link[data-value="' + tabId + '"], .navbar-nav a[data-value="' + tabId + '"]');
        if (link) {
            link.click();
        } else {
            // Check dropdown menus
            var dropdownLink = document.querySelector('.dropdown-menu .nav-link[data-value="' + tabId + '"], .dropdown-menu a[data-value="' + tabId + '"]');
            if (dropdownLink) {
                dropdownLink.click();
            }
        }
    }

    // Global keyboard listener for ⌘K / Ctrl+K and modal focus trapping
    $(document).on('keydown', function (e) {
        var isOpen = $('#cmd_palette_backdrop').hasClass('open');
        if ((e.metaKey || e.ctrlKey) && (e.key === 'k' || e.key === 'K')) {
            e.preventDefault();
            if (isOpen) {
                closeCommandPalette();
            } else {
                openCommandPalette();
            }
        } else if (isOpen) {
            if (e.key === 'Escape') {
                e.preventDefault();
                closeCommandPalette();
            } else if (e.key === 'Tab') {
                // Focus trap inside the modal
                var focusables = $('#cmd_palette_backdrop').find('input, button, [tabindex]:not([tabindex="-1"])').filter(':visible');
                if (focusables.length > 0) {
                    var firstEl = focusables[0];
                    var lastEl = focusables[focusables.length - 1];
                    if (e.shiftKey) {
                        if (document.activeElement === firstEl || document.activeElement === document.body) {
                            e.preventDefault();
                            lastEl.focus();
                        }
                    } else {
                        if (document.activeElement === lastEl) {
                            e.preventDefault();
                            firstEl.focus();
                        }
                    }
                } else {
                    e.preventDefault();
                }
            }
        }
    });

    // Input filtering in command palette
    $(document).on('input', '#cmd_palette_input', function () {
        var query = $(this).val().toLowerCase().trim();
        if (!query) {
            filteredModules = MODULE_REGISTRY.slice();
        } else {
            var words = query.split(/\s+/);
            filteredModules = MODULE_REGISTRY.filter(function (mod) {
                var target = (mod.title + " " + mod.desc + " " + mod.keywords + " " + mod.category).toLowerCase();
                return words.every(function (w) { return target.indexOf(w) !== -1; });
            });
        }
        cmdActiveIndex = 0;
        renderCommandPaletteItems();
    });

    // Key navigation inside command palette
    $(document).on('keydown', '#cmd_palette_input', function (e) {
        if (e.key === 'ArrowDown') {
            e.preventDefault();
            if (filteredModules.length > 0) {
                cmdActiveIndex = (cmdActiveIndex + 1) % filteredModules.length;
                renderCommandPaletteItems();
            }
        } else if (e.key === 'ArrowUp') {
            e.preventDefault();
            if (filteredModules.length > 0) {
                cmdActiveIndex = (cmdActiveIndex - 1 + filteredModules.length) % filteredModules.length;
                renderCommandPaletteItems();
            }
        } else if (e.key === 'Enter') {
            e.preventDefault();
            if (filteredModules.length > 0 && filteredModules[cmdActiveIndex]) {
                activateTab(filteredModules[cmdActiveIndex].id);
            }
        }
    });

    // Click on palette item or backdrop
    $(document).on('click', '.cmd-palette-item', function () {
        var tabId = $(this).data('tab-id');
        if (tabId) {
            activateTab(tabId);
        }
    });

    $(document).on('click', '#cmd_palette_backdrop', function (e) {
        if (e.target === this) {
            closeCommandPalette();
        }
    });

    // Accessible Plotly Chart Twin toggle with ARIA synchronization
    $(document).on('click', '.chart-twin-toggle-btn', function () {
        var $btn = $(this);
        var targetId = $btn.data('target-id');
        var view = $btn.data('view'); // 'chart' or 'table'
        var $wrapper = $('#' + targetId);
        if ($wrapper.length) {
            var $toolbar = $btn.closest('.chart-twin-toolbar');
            $toolbar.find('.chart-twin-toggle-btn').removeClass('active').attr('aria-pressed', 'false');
            $btn.addClass('active').attr('aria-pressed', 'true');

            if (view === 'table') {
                $wrapper.find('.chart-view-panel').hide();
                $wrapper.find('.table-view-panel').show();
            } else {
                $wrapper.find('.table-view-panel').hide();
                $wrapper.find('.chart-view-panel').show();
                if (window.Plotly) {
                    $wrapper.find('.js-plotly-plot').each(function () {
                        Plotly.Plots.resize(this);
                    });
                }
            }
        }
    });

    // Initialize DOM
    $(document).on('shiny:connected', function () {
        initMobileMenu();
        createCommandPaletteDOM();
    });
});
