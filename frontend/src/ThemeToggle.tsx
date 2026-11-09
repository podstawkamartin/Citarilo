import { useState } from "react";

type Theme = "light" | "dark";

/** Sun/moon switch; the choice is remembered in this browser (see public/theme.js). */
export function ThemeToggle() {
  const [theme, setTheme] = useState<Theme>(() =>
    document.documentElement.dataset.theme === "dark" ? "dark" : "light",
  );

  function toggle() {
    const next: Theme = theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try {
      localStorage.setItem("citarilo-theme", next);
    } catch {
      // Storage can be unavailable (private mode); the switch still works for this page.
    }
    setTheme(next);
  }

  return (
    <button
      type="button"
      className="theme"
      onClick={toggle}
      aria-label={theme === "dark" ? "Přepnout na světlý režim" : "Přepnout na tmavý režim"}
      title="Světlý / tmavý režim"
    >
      <svg className="sun" viewBox="0 0 24 24" aria-hidden="true">
        <circle cx="12" cy="12" r="4.2" />
        <path d="M12 2.5v2.2M12 19.3v2.2M2.5 12h2.2M19.3 12h2.2M5.3 5.3l1.6 1.6M17.1 17.1l1.6 1.6M5.3 18.7l1.6-1.6M17.1 6.9l1.6-1.6" />
      </svg>
      <svg className="moon" viewBox="0 0 24 24" aria-hidden="true">
        <path d="M20 14.5A8 8 0 1 1 9.5 4a6.5 6.5 0 0 0 10.5 10.5z" />
      </svg>
    </button>
  );
}
