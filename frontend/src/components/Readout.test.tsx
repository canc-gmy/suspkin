import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { DataSource, DatasetInfo } from "../api";
import { SourceContext } from "../api/hooks";
import type { ReadoutTable } from "../lib/readout";
import { Readout } from "./Readout";
import { ResolutionControl, sharedSteps } from "./ResolutionControl";

const table: ReadoutTable = {
  columns: [{ panelId: "camber", title: "Camber [deg]" }],
  rows: [
    {
      revId: "a",
      letter: "A",
      color: "#111",
      cells: [{ parts: [{ prefix: null, text: "−1.312" }], spoken: "−1.312" }],
    },
  ],
};
const base = {
  xLabel: "Wheel travel",
  xUnit: "mm",
  formatX: (x: number) => x.toFixed(2),
  table,
  showPoints: false,
  onShowPoints: () => undefined,
};

afterEach(() => vi.useRealTimers());

describe("Readout", () => {
  it("shows the idle text without a cursor and keeps the revision rows", () => {
    render(<Readout {...base} x={null} pinned={false} onPinX={() => undefined} />);
    expect(screen.getByText("Move over a plot to read values")).toBeTruthy();
    expect(screen.getByText("Camber [deg]")).toBeTruthy();
    expect(screen.getByText("A")).toBeTruthy();
  });

  it("pins at a typed x on Enter and rejects junk", () => {
    const onPin = vi.fn();
    render(<Readout {...base} x={null} pinned={false} onPinX={onPin} />);
    const input = screen.getByLabelText("Wheel travel");
    fireEvent.change(input, { target: { value: "abc" } });
    fireEvent.keyDown(input, { key: "Enter" });
    expect(onPin).not.toHaveBeenCalled();
    expect(screen.getByText("Enter a number in mm")).toBeTruthy();
    fireEvent.change(input, { target: { value: "12,5" } });
    fireEvent.keyDown(input, { key: "Enter" });
    expect(onPin).toHaveBeenCalledWith(12.5);
  });

  it("says it is pinned and announces the values politely, after a delay", () => {
    vi.useFakeTimers();
    const { container } = render(<Readout {...base} x={3} pinned onPinX={() => undefined} />);
    expect(screen.getByText("Pinned, Esc releases")).toBeTruthy();
    const live = container.querySelector("[aria-live='polite']");
    expect(live?.textContent).toBe("");
    act(() => {
      vi.advanceTimersByTime(600);
    });
    expect(live?.textContent).toContain("Wheel travel 3.00 mm");
    expect(live?.textContent).toContain("A: Camber [deg] −1.312");
  });
});

const ds = (id: string, steps: number, fixed: boolean): DatasetInfo => ({
  id,
  label: id,
  kind: fixed ? "bundle" : "vehicle",
  source: `${id}.yaml`,
  color: 0,
  studies: [],
  params: {},
  steps,
  resolution_fixed: fixed,
  has_study: false,
});

function src(readOnly: boolean): DataSource {
  const fail = () => Promise.reject(new Error("not used"));
  return {
    kind: readOnly ? "snapshot" : "http",
    readOnly,
    patchDataset: vi.fn((id: string) => Promise.resolve(ds(id, 61, false))),
    getHealth: fail,
    getFigures: fail,
    listDatasets: () => Promise.resolve([]),
  } as unknown as DataSource;
}

function renderControl(source: DataSource, datasets: DatasetInfo[]) {
  render(
    <SourceContext.Provider value={source}>
      <QueryClientProvider client={new QueryClient()}>
        <ResolutionControl datasets={datasets} />
      </QueryClientProvider>
    </SourceContext.Provider>,
  );
}

describe("ResolutionControl", () => {
  it("re-solves only the vehicle-backed revisions", async () => {
    const s = src(false);
    renderControl(s, [ds("a", 31, false), ds("b", 31, true), ds("c", 31, false)]);
    fireEvent.click(screen.getByRole("radio", { name: "61" }));
    await waitFor(() => expect(s.patchDataset).toHaveBeenCalledTimes(2));
    expect(s.patchDataset).toHaveBeenCalledWith("a", { steps: 61 });
    expect(s.patchDataset).toHaveBeenCalledWith("c", { steps: 61 });
  });

  it("is disabled with a reason in a read-only snapshot", () => {
    renderControl(src(true), [ds("a", 31, false)]);
    expect((screen.getByRole("radio", { name: "61" }) as HTMLButtonElement).disabled).toBe(true);
    expect(screen.getByTitle("Read-only snapshot")).toBeTruthy();
  });

  it("reports the shared resolution, or none when mixed or stored", () => {
    expect(sharedSteps([ds("a", 61, false), ds("b", 61, false)])).toBe("61");
    expect(sharedSteps([ds("a", 61, false), ds("b", 31, false)])).toBe("");
    expect(sharedSteps([ds("a", 31, true)])).toBe("");
  });
});
