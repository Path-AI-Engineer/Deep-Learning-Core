export type NumericRow = Record<string, number>;

export function parseCsv(text: string): NumericRow[] {
  const lines = text.trim().split(/\r?\n/).filter(Boolean);
  if (lines.length < 2) throw new Error("CSV must contain a header and at least one data row.");
  const headers = lines[0].split(",").map((value) => value.trim());
  if (headers.some((header) => !header)) throw new Error("CSV headers cannot be empty.");
  return lines.slice(1).map((line, rowIndex) => {
    const cells = line.split(",").map((value) => value.trim());
    const values = cells.map(Number);
    if (
      values.length !== headers.length ||
      cells.some((value) => value === "") ||
      values.some((value) => !Number.isFinite(value))
    ) {
      throw new Error(`Row ${rowIndex + 2} does not match the numeric schema.`);
    }
    return Object.fromEntries(headers.map((header, index) => [header, values[index]]));
  });
}
