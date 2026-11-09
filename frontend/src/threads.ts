import { useCallback, useLayoutEffect, useState } from "react";

export type Link = { from: string; to: string; group: number };
export type Thread = { d: string; group: number; key: string };

/**
 * Curved threads between port elements inside `root`, measured from their real positions.
 * Ports are found by `data-port`; a hidden port (zero size, e.g. on mobile) draws nothing.
 */
export function useThreads(root: React.RefObject<HTMLElement | null>, links: Link[]): Thread[] {
  const [threads, setThreads] = useState<Thread[]>([]);

  const measure = useCallback(() => {
    const el = root.current;
    if (!el) return;
    const box = el.getBoundingClientRect();
    const port = (name: string) => {
      const p = el.querySelector(`[data-port="${CSS.escape(name)}"]`);
      if (!p) return null;
      const r = p.getBoundingClientRect();
      if (r.width === 0) return null;
      return { x: r.left + r.width / 2 - box.left, y: r.top + r.height / 2 - box.top };
    };
    const next: Thread[] = [];
    for (const link of links) {
      const a = port(link.from);
      const b = port(link.to);
      if (!a || !b) continue;
      const x1 = a.x + 5;
      const x2 = b.x - 5;
      const k = (x2 - x1) / 2;
      next.push({
        d: `M ${x1} ${a.y} C ${x1 + k} ${a.y}, ${x2 - k} ${b.y}, ${x2} ${b.y}`,
        group: link.group,
        key: `${link.from}>${link.to}`,
      });
    }
    setThreads(next);
  }, [root, links]);

  useLayoutEffect(() => {
    measure();
    const el = root.current;
    if (!el || typeof ResizeObserver === "undefined") return;
    // Any change in size (fonts loading, a document opening, a window resize) moves the ports.
    const observer = new ResizeObserver(measure);
    observer.observe(el);
    for (const child of el.children) observer.observe(child);
    document.fonts?.ready.then(measure);
    return () => observer.disconnect();
  }, [root, measure]);

  return threads;
}
