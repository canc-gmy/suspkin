// Regenerate src/api/openapi.json (from the Python app) and src/api/schema.d.ts.
import { spawnSync } from "node:child_process";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const json = resolve(root, "src/api/openapi.json");
const dts = resolve(root, "src/api/schema.d.ts");
const run = (cmd, args) => spawnSync(cmd, args, { cwd: root, stdio: "inherit", shell: true });

const py = ["suspkin.app.openapi", json];
let r = run("uv", ["run", "python", "-m", ...py]);
if (r.status !== 0) r = run("python", ["-m", "uv", "run", "python", "-m", ...py]);
if (r.status !== 0) process.exit(r.status ?? 1);
r = run("npx", ["openapi-typescript", json, "-o", dts]);
process.exit(r.status ?? 1);
