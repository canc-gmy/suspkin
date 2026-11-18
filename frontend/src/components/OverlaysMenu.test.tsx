import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { NO_STUDY_HINT } from "../lib/regionSource";
import { OverlaysMenu } from "./OverlaysMenu";

const STUDIES = [{ id: "c", name: "C  study" }, { id: "d", name: "D  run" }];

function mount(hasStudy: boolean) {
  const onOpenFiles = vi.fn();
  const onSource = vi.fn();
  const onCentres = vi.fn();
  const onRegions = vi.fn();
  render(
    <div>
      <span>outside</span>
      <OverlaysMenu centres={new Set()} onCentres={onCentres} regions={new Set()} onRegions={onRegions} sources={hasStudy ? STUDIES : []} source={hasStudy ? "c" : null} onSource={onSource} onOpenFiles={onOpenFiles} />
    </div>,
  );
  return { onCentres, onRegions, onOpenFiles, onSource };
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

  it("lists the open studies and reports a pick", () => {
    const { onSource } = mount(true);
    fireEvent.click(screen.getByRole("button", { name: "Overlays" }));
    expect(screen.queryByText(NO_STUDY_HINT)).toBeNull();
    fireEvent.change(screen.getByRole("combobox", { name: "Study" }), { target: { value: "d" } });
    expect(onSource).toHaveBeenCalledWith("d");
    expect(screen.getAllByRole("option").map((o) => o.textContent)).toEqual(["C  study", "D  run"]);
  });

  it("disables the region checkboxes without a study and explains why", () => {
    const { onOpenFiles } = mount(false);
    fireEvent.click(screen.getByRole("button", { name: "Overlays" }));
    const box = screen.getByLabelText("Design domains") as HTMLInputElement;
    expect(box.disabled).toBe(true);
    const hint = document.getElementById(box.getAttribute("aria-describedby") as string);
    expect(hint?.textContent).toContain(NO_STUDY_HINT);
    fireEvent.click(screen.getByRole("button", { name: "Open files" }));
    expect(onOpenFiles).toHaveBeenCalledOnce();
    expect((screen.getByLabelText("Pitch centre") as HTMLInputElement).disabled).toBe(false);
  });
});
