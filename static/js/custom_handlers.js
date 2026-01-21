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
        var sidebar = $('.nav-pills').closest('[class*="col-"]');

        // Initial state
        if ($(window).width() < 768) {
            sidebar.hide();
        }

        // Toggle button handler
        $(document).on('click', '#mobile_menu_btn', function () {
            sidebar.toggle();
        });

        // Safe resize handler
        $(window).resize(function () {
            if ($(window).width() >= 768) {
                sidebar.show();
            } else {
                // Start hidden on mobile resize? Or keep state? Keep state is better.
            }
        });
    }

    // Initialize when Shiny connects (ensures DOM is ready)
    $(document).on('shiny:connected', function () {
        initMobileMenu();
    });
});
