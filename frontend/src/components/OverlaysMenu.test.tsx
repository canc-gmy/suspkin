import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { OverlaysMenu } from "./OverlaysMenu";

function mount(hasStudy: boolean) {
  const onCentres = vi.fn();
  const onRegions = vi.fn();
  render(
    <div>
      <span>outside</span>
      <OverlaysMenu centres={new Set()} onCentres={onCentres} regions={new Set()} onRegions={onRegions} hasStudy={hasStudy} />
    </div>,
  );
  return { onCentres, onRegions };
}

describe("OverlaysMenu", () => {
  it("opens, toggles a centre, closes on Esc and on an outside click", () => {
    const { onCentres } = mount(true);
    const btn = screen.getByRole("button", { name: "Overlays" });
    expect(btn.getAttribute("aria-expanded")).toBe("false");
    fireEvent.click(btn);
    expect(screen.getAllByRole("checkbox")).toHaveLength(7);
    fireEvent.click(screen.getByLabelText("Pitch centre"));
    expect([...(onCentres.mock.calls[0]![0] as Set<string>)]).toEqual(["pitch"]);
    fireEvent.keyDown(document, { key: "Escape" });
    expect(screen.queryByRole("checkbox")).toBeNull();
    fireEvent.click(btn);
    fireEvent.mouseDown(screen.getByText("outside"));
    expect(screen.queryByRole("checkbox")).toBeNull();
  });

  it("toggles a region role when a study is attached", () => {
    const { onRegions } = mount(true);
    fireEvent.click(screen.getByRole("button", { name: "Overlays" }));
    fireEvent.click(screen.getByLabelText("Keep-outs (outside)"));
    expect([...(onRegions.mock.calls[0]![0] as Set<string>)]).toEqual(["outside"]);
  });

  it("disables the region checkboxes without a study", () => {
    mount(false);
    fireEvent.click(screen.getByRole("button", { name: "Overlays" }));
    expect((screen.getByLabelText("Design domains") as HTMLInputElement).disabled).toBe(true);
    expect((screen.getByLabelText("Pitch centre") as HTMLInputElement).disabled).toBe(false);
  });
});
