import { describe, expect, it } from "vitest";
import { parseCsv } from "./csv";

describe("parseCsv", () => {
  it("parses numeric rows with ordered headers", () => {
    expect(parseCsv("a,b\n1,2\n3,4")).toEqual([
      { a: 1, b: 2 },
      { a: 3, b: 4 },
    ]);
  });

  it("rejects missing and non-numeric values", () => {
    expect(() => parseCsv("a,b\n1,")).toThrow("does not match");
    expect(() => parseCsv("a,b\n1,nope")).toThrow("does not match");
  });
});
