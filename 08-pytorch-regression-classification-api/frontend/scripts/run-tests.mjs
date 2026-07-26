import assert from "node:assert/strict";
import { renderToStaticMarkup } from "react-dom/server";

import { StatePanel } from "../.runtime-tests/components/StatePanel.js";
import { parseCsv } from "../.runtime-tests/features/batch/csv.js";

const rows = parseCsv("a,b\n1,2\n3,4");
assert.deepEqual(rows, [
  { a: 1, b: 2 },
  { a: 3, b: 4 },
]);
assert.throws(() => parseCsv("a,b\n1,"), /does not match/);
assert.throws(() => parseCsv("a,b\n1,nope"), /does not match/);

const errorMarkup = renderToStaticMarkup(
  StatePanel({
    kind: "error",
    title: "Model unavailable",
    description: "Try again.",
  }),
);
assert.match(errorMarkup, /role="alert"/);
assert.match(errorMarkup, /Model unavailable/);

console.log("Frontend runtime tests passed: CSV validation and recoverable error state.");
