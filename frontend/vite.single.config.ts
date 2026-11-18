import { defineConfig, mergeConfig } from "vite";
import { viteSingleFile } from "vite-plugin-singlefile";
import base from "./vite.config";

// One self-contained HTML file (scripts, styles, fonts inlined) for the snapshot export.
export default mergeConfig(
  base,
  defineConfig({
    plugins: [viteSingleFile()],
    build: { outDir: "../src/suspkin/app/static_single", emptyOutDir: true },
  }),
);
