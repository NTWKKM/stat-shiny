$(document).ready(function () {
    // Handler for setting element styles dynamically
    Shiny.addCustomMessageHandler("set_element_style", function (message) {
        var el = document.getElementById(message.id);
        if (el) {
            for (var styleName in message.style) {
                if (message.style.hasOwnProperty(styleName)) {
                    el.style[styleName] = message.style[styleName];
                }
            }
        } else {
            console.warn("set_element_style: Element with id '" + message.id + "' not found.");
        }
    });

    // Handler for setting inner text dynamically
    Shiny.addCustomMessageHandler("set_inner_text", function (message) {
        var el = document.getElementById(message.id);
        if (el) {
            el.innerText = message.text;
        } else {
            console.warn("set_inner_text: Element with id '" + message.id + "' not found.");
        }
    });
    // Mobile Menu Logic
    function initMobileMenu() {
        var sidebar = $('.app-container .nav-pills').first().closest('[class*="col-"]');
        if (!sidebar.length) sidebar = $('#sidebar'); // Fallback to id if available

        // Initial state
        if ($(window).width() < 768) {
            sidebar.hide();
        }

        // Toggle button handler
        $(document).on('click', '#mobile_menu_btn', function () {
            sidebar.toggle();
        });

        // Debounced resize handler
        var resizeTimer;
        $(window).resize(function () {
            clearTimeout(resizeTimer);
            resizeTimer = setTimeout(function () {
                if ($(window).width() >= 768) {
                    sidebar.show();
                }
            }, 250);
        });
    }

    // Navbar Brand Home Click Listener
    $(document).on('click', '#navbar_brand_home', function (e) {
        e.preventDefault();
        var homeLink = document.querySelector('.navbar-nav .nav-link[data-value="home"]');
        if (homeLink) homeLink.click();
    });

    // Initialize when Shiny connects (ensures DOM is ready)
    $(document).on('shiny:connected', function () {
        initMobileMenu();
    });
});
