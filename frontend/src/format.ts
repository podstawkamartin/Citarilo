/** Czech noun form for a count: 1 dokument, 2–4 dokumenty, 5+ dokumentů. */
export function plural(n: number, one: string, few: string, many: string): string {
  return n === 1 ? one : n >= 2 && n <= 4 ? few : many;
}
