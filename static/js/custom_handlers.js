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
});
