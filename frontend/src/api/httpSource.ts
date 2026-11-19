import { ApiError, type DataSource } from "./source";
import type { ScenePose } from "./types";

function detailText(body: unknown, fallback: string): string {
  if (body && typeof body === "object" && "detail" in body) {
    const d = (body as { detail: unknown }).detail;
    if (typeof d === "string") return d;
    if (Array.isArray(d)) {
      return d
        .map((e) => (e && typeof e === "object" && "msg" in e ? String((e as { msg: unknown }).msg) : ""))
        .filter(Boolean)
        .join("; ");
    }
  }
  return fallback;
}

type Method = Exclude<keyof DataSource, "kind">;
type Ret<K extends Method> = DataSource[K] extends (...args: never[]) => infer R ? R : never;

export class HttpSource implements DataSource {
  readonly kind = "http" as const;
  readonly readOnly = false;

  private readonly base: string;

  constructor(base = "/api/v1") {
    this.base = base;
  }

  private async request<T>(path: string, init?: RequestInit): Promise<T> {
    let res: Response;
    try {
      res = await fetch(this.base + path, init);
    } catch (e) {
      if (init?.signal?.aborted) throw e; // a cancelled request is not a server failure
      throw new ApiError("The suspkin server is not reachable. Check that it is still running.", 0);
    }
    if (!res.ok) {
      let body: unknown = null;
      try {
        body = await res.json();
      } catch {
        // not JSON: fall back to the status line
      }
      throw new ApiError(detailText(body, `${res.status} ${res.statusText}`), res.status);
    }
    if (res.status === 204) return undefined as T;
    return (await res.json()) as T;
  }

  private json(method: string, body: unknown): RequestInit {
    return { method, headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
  }

  getHealth(): Ret<"getHealth"> {
    return this.request("/health");
  }
  getFigures(): Ret<"getFigures"> {
    return this.request("/figures");
  }
  listDatasets(): Ret<"listDatasets"> {
    return this.request("/datasets");
  }
  openPath(path: string): Ret<"openPath"> {
    return this.request("/datasets", this.json("POST", { path }));
  }
  patchDataset(id: string, patch: Parameters<DataSource["patchDataset"]>[1]): Ret<"patchDataset"> {
    return this.request(`/datasets/${encodeURIComponent(id)}`, this.json("PATCH", patch));
  }
  async closeDataset(id: string): Promise<void> {
    await this.request<void>(`/datasets/${encodeURIComponent(id)}`, { method: "DELETE" });
  }
  deriveDataset(id: string, body: Parameters<DataSource["deriveDataset"]>[1], signal?: AbortSignal): Ret<"deriveDataset"> {
    return this.request(`/datasets/${encodeURIComponent(id)}/derive`, { ...this.json("POST", body), signal });
  }
  saveDataset(id: string, body: Parameters<DataSource["saveDataset"]>[1]): Ret<"saveDataset"> {
    return this.request(`/datasets/${encodeURIComponent(id)}/save`, this.json("POST", body));
  }
  getTable(id: string, study: string): Ret<"getTable"> {
    return this.request(`/datasets/${encodeURIComponent(id)}/tables/${encodeURIComponent(study)}`);
  }
  getSummary(id: string): Ret<"getSummary"> {
    return this.request(`/datasets/${encodeURIComponent(id)}/summary`);
  }
  getScene(id: string, pose: ScenePose = {}): Ret<"getScene"> {
    const q = new URLSearchParams();
    for (const [k, v] of Object.entries(pose)) if (v !== undefined) q.set(k, String(v));
    const qs = q.toString();
    return this.request(`/datasets/${encodeURIComponent(id)}/scene${qs ? `?${qs}` : ""}`);
  }
  getHardpoints(id: string): Ret<"getHardpoints"> {
    return this.request(`/datasets/${encodeURIComponent(id)}/hardpoints`);
  }
  getRegions(id: string): Ret<"getRegions"> {
    return this.request(`/datasets/${encodeURIComponent(id)}/regions`);
  }
  getHistory(runId: string): Ret<"getHistory"> {
    return this.request(`/runs/${encodeURIComponent(runId)}/history`);
  }
  exportSnapshot(out?: string, overwrite = false): Ret<"exportSnapshot"> {
    return this.request("/export", this.json("POST", { ...(out ? { out } : {}), overwrite }));
  }
  browse(dir: string): Ret<"browse"> {
    return this.request(`/browse?dir=${encodeURIComponent(dir)}`);
  }
}
