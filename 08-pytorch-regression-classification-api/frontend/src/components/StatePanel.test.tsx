import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { StatePanel } from "./StatePanel";

describe("StatePanel", () => {
  it("exposes recoverable errors as alerts", () => {
    render(<StatePanel kind="error" title="Model unavailable" description="Try again." />);
    expect(screen.getByRole("alert")).toHaveTextContent("Model unavailable");
  });
});
