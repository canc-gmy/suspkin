import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { DataSource, DatasetInfo } from "../api";
import { SourceContext } from "../api/hooks";
import { useUi } from "../state/store";
import { TitleBlock } from "./TitleBlock";

const PALETTE = ["#0072B2", "#D55E00", "#009E73"];

const ds = (id: string, label: string, color: number): DatasetInfo => ({
  id,
  label,
  kind: "vehicle",
  source: `${label}.yaml`,
  color,
  studies: ["front_corner"],
  params: {},
  steps: 31,
  resolution_fixed: false,
  has_study: false,
});

function mockSource(): DataSource {
  const fail = () => Promise.reject(new Error("not used"));
  return {
    kind: "http",
    readOnly: false,
    exportSnapshot: fail,
    getHealth: fail,
    getFigures: fail,
    listDatasets: fail,
    openPath: fail,
    patchDataset: vi.fn((id: string, patch) => Promise.resolve({ ...ds(id, "x", 0), ...patch })),
    closeDataset: vi.fn(() => Promise.resolve()),
    deriveDataset: fail,
    saveDataset: fail,
    getTable: fail,
    getSummary: fail,
    getScene: fail,
    getHardpoints: fail,
    getRegions: fail,
    getHistory: fail,
    browse: fail,
  } as DataSource;
}

function setup(src = mockSource()) {
  const datasets = [ds("a1", "fs_reference", 0), ds("b2", "raised_lca", 1)];
  useUi.getState().syncDatasets(datasets.map((d) => d.id));
  render(
    <SourceContext.Provider value={src}>
      <QueryClientProvider client={new QueryClient()}>
        <TitleBlock datasets={datasets} palette={PALETTE} theme="light" />
      </QueryClientProvider>
    </SourceContext.Provider>,
  );
  return src;
}

beforeEach(() => {
  useUi.setState({ order: [], letters: {}, hidden: {}, reference: null, fresh: {}, synced: false });
});

describe("TitleBlock", () => {
  it("lists revisions with letters, designs and a reference radio", () => {
    setup();
    expect(screen.getByRole("columnheader", { name: "Rev" })).toBeTruthy();
    expect(screen.getByText("fs_reference")).toBeTruthy();
    expect(screen.getByText("raised_lca")).toBeTruthy();
    expect(screen.getByText("A")).toBeTruthy();
    expect(screen.getByText("B")).toBeTruthy();
    expect((screen.getByLabelText("Use revision A as the reference") as HTMLInputElement).checked).toBe(true);
    expect(screen.getByText("Solid left/front, dashed right/rear")).toBeTruthy();
  });

  it("toggles visibility, picks the reference and closes a revision", async () => {
    const src = setup();
    fireEvent.click(screen.getByRole("button", { name: "Hide revision B" }));
    expect(useUi.getState().hidden["b2"]).toBe(true);
    fireEvent.click(screen.getByLabelText("Use revision B as the reference"));
    expect(useUi.getState().reference).toBe("b2");
    fireEvent.click(screen.getByRole("button", { name: "Close revision A" }));
    await waitFor(() => expect(src.closeDataset).toHaveBeenCalledWith("a1"));
  });

  it("renames inline with a PATCH", async () => {
    const src = setup();
    fireEvent.click(screen.getByRole("button", { name: "Rename revision B" }));
    const input = screen.getByLabelText("Name of revision B");
    fireEvent.change(input, { target: { value: "lowered" } });
    fireEvent.keyDown(input, { key: "Enter" });
    await waitFor(() => expect(src.patchDataset).toHaveBeenCalledWith("b2", { label: "lowered" }));
  });

  it("shows a rename error inline", async () => {
    const src = mockSource();
    src.patchDataset = vi.fn(() => Promise.reject(new Error("label already in use: x")));
    setup(src);
    fireEvent.click(screen.getByRole("button", { name: "Rename revision B" }));
    const input = screen.getByLabelText("Name of revision B");
    fireEvent.change(input, { target: { value: "x" } });
    fireEvent.keyDown(input, { key: "Enter" });
    expect((await screen.findByRole("alert")).textContent).toContain("label already in use");
  });
});
