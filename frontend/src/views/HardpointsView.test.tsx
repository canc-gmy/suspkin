import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { DataSource, DatasetInfo, Hardpoint } from "../api";
import { SourceContext } from "../api/hooks";
import { useUi } from "../state/store";
import { HardpointsView } from "./HardpointsView";

const info = (id: string, label: string, kind: DatasetInfo["kind"] = "vehicle"): DatasetInfo => ({
  id,
  label,
  kind,
  source: `${label}.yaml`,
  color: 0,
  studies: [],
  params: {},
  steps: 31,
  resolution_fixed: false,
  has_study: false,
});

const POINTS: Hardpoint[] = [
  { axle: "front", point: "lca_outer", side: "left", x: 0, y: 0.58, z: 0.13, overridden: false },
  { axle: "front", point: "lca_outer", side: "right", x: 0, y: -0.58, z: 0.13, overridden: false },
];

function mockSource(over: Partial<DataSource> = {}): DataSource {
  const fail = () => Promise.reject(new Error("not used"));
  return {
    kind: "http",
    readOnly: false,
    getHardpoints: () => Promise.resolve(POINTS),
    getRegions: () => Promise.reject(new Error("not used")),
    deriveDataset: vi.fn(() => Promise.resolve(info("e1", "car edited", "edited"))),
    closeDataset: vi.fn(() => Promise.resolve()),
    saveDataset: fail,
    ...over,
  } as unknown as DataSource;
}

function renderView(src: DataSource, datasets: DatasetInfo[]) {
  useUi.getState().syncDatasets(datasets.map((d) => d.id));
  return render(
    <SourceContext.Provider value={src}>
      <QueryClientProvider client={new QueryClient()}>
        <HardpointsView datasets={datasets} />
      </QueryClientProvider>
    </SourceContext.Provider>,
  );
}

beforeEach(() => {
  useUi.setState({ order: [], letters: {}, hidden: {}, reference: null, fresh: {}, synced: false });
});
afterEach(cleanup);

describe("HardpointsView editing", () => {
  it("sends mirrored edits in metres and leaves edit mode after Solve", async () => {
    const src = mockSource();
    renderView(src, [info("a1", "car")]);
    fireEvent.click(await screen.findByRole("button", { name: "Edit as new revision" }));
    const z = screen.getByLabelText("lca_outer left z in mm") as HTMLInputElement;
    fireEvent.change(z, { target: { value: "142.5" } });
    // the right twin follows, as plain numbers
    expect(screen.queryByLabelText("lca_outer right z in mm")).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Solve" }));
    await waitFor(() =>
      expect(src.deriveDataset).toHaveBeenCalledWith(
        "a1",
        { edits: [{ axle: "front", point: "lca_outer", side: "both", z: 0.1425 }] },
        expect.anything(),
      ),
    );
    await waitFor(() => expect(screen.getByRole("button", { name: "Edit as new revision" })).toBeTruthy());
  });

  it("shows the server's message under the table when the solve is rejected", async () => {
    const { ApiError } = await import("../api");
    const src = mockSource({
      deriveDataset: vi.fn(() => Promise.reject(new ApiError("invalid geometry after editing front.lca_outer: x", 400))),
    });
    renderView(src, [info("a1", "car")]);
    fireEvent.click(await screen.findByRole("button", { name: "Edit as new revision" }));
    fireEvent.change(screen.getByLabelText("lca_outer left z in mm"), { target: { value: "500" } });
    fireEvent.click(screen.getByRole("button", { name: "Solve" }));
    const alert = await screen.findByRole("alert");
    expect(alert.textContent).toContain("front.lca_outer");
    expect(alert.textContent).toContain("solve again");
  });

  it("is disabled in a snapshot", async () => {
    renderView(mockSource({ readOnly: true, kind: "snapshot" }), [info("a1", "car")]);
    const btn = (await screen.findByRole("button", { name: "Edit as new revision" })) as HTMLButtonElement;
    expect(btn.disabled).toBe(true);
    expect(btn.parentElement?.getAttribute("title")).toBe("Read-only snapshot");
  });

  it("offers Save as YAML for edited revisions and asks before replacing", async () => {
    const { ApiError } = await import("../api");
    const save = vi
      .fn()
      .mockRejectedValueOnce(new ApiError("already exists: car_edited.yaml", 409))
      .mockResolvedValueOnce({ path: "car_edited.yaml", size: 10 });
    const confirm = vi.spyOn(window, "confirm").mockReturnValue(true);
    renderView(mockSource({ saveDataset: save }), [info("a1", "car"), info("e1", "car edited", "edited")]);
    fireEvent.click(await screen.findByRole("button", { name: "Save as YAML" }));
    expect((screen.getByLabelText("File name") as HTMLInputElement).value).toBe("car_edited.yaml");
    fireEvent.click(screen.getByRole("button", { name: "Save" }));
    await waitFor(() => expect(save).toHaveBeenCalledTimes(2));
    expect(confirm).toHaveBeenCalledWith("Replace car_edited.yaml?");
    expect(save).toHaveBeenLastCalledWith("e1", { path: "car_edited.yaml", overwrite: true });
    expect((await screen.findAllByText("Saved to car_edited.yaml")).length).toBeGreaterThan(0);
    confirm.mockRestore();
  });
});

describe("HardpointsView auto-solve", () => {
  it("solves once, 600 ms after the last edit, and closes the previous auto revision", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    try {
      let n = 0;
      const derive = vi.fn(() => Promise.resolve(info(`e${++n}`, "car edited", "edited")));
      const src = mockSource({ deriveDataset: derive });
      renderView(src, [info("a1", "car")]);
      fireEvent.click(await screen.findByRole("button", { name: "Edit as new revision" }));
      fireEvent.click(screen.getByLabelText("Auto-solve"));
      const z = screen.getByLabelText("lca_outer left z in mm");
      fireEvent.change(z, { target: { value: "140" } });
      await vi.advanceTimersByTimeAsync(400);
      fireEvent.change(z, { target: { value: "141" } });
      await vi.advanceTimersByTimeAsync(400);
      expect(derive).not.toHaveBeenCalled(); // the wait restarted at the second edit
      await vi.advanceTimersByTimeAsync(300);
      expect(derive).toHaveBeenCalledTimes(1);
      fireEvent.change(z, { target: { value: "142" } });
      await vi.advanceTimersByTimeAsync(700);
      expect(derive).toHaveBeenCalledTimes(2);
      await waitFor(() => expect(src.closeDataset).toHaveBeenCalledWith("e1"));
    } finally {
      vi.useRealTimers();
    }
  });
});
