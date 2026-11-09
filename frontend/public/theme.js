// Applies the saved or system colour theme before the app renders, so the page never flashes.
(function () {
  var saved = null;
  try {
    saved = localStorage.getItem("citarilo-theme");
  } catch (e) {}
  var theme = saved === "light" || saved === "dark" ? saved : matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  document.documentElement.dataset.theme = theme;
})();
