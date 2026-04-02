// document.addEventListener("DOMContentLoaded", function () {

//     const nav = document.getElementById("top_nav");
//     if (!nav) return;

//     let lastScroll = 0;

//     window.addEventListener("scroll", () => {
//         const currentScroll = window.pageYOffset;

//         if (currentScroll <= 0) {
//             nav.style.transform = "translateY(0)";
//             return;
//         }

//         if (currentScroll > lastScroll) {
//             nav.style.transform = "translateY(-100%)";
//         } else {
//             nav.style.transform = "translateY(0)";
//         }

//         lastScroll = currentScroll;
//     });

// });

function updateThemeColor(mode) {
    let meta = document.querySelector('meta[name="theme-color"]');
    if (!meta) {
        meta = document.createElement('meta');
        meta.name = "theme-color";
        document.head.appendChild(meta);
    }

    meta.content = "#2c2c30";
}

document.addEventListener("DOMContentLoaded", function () {
    updateThemeColor(document.documentElement.dataset.mode);
});