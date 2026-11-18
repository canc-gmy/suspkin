import "@fontsource/barlow/400.css";
import "@fontsource/barlow/500.css";
import "@fontsource/barlow/600.css";
import "@fontsource/barlow-semi-condensed/400.css";
import "@fontsource/barlow-semi-condensed/500.css";
import "@fontsource/barlow-semi-condensed/600.css";
import "./theme/tokens.css";
import "./styles.css";

import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { App } from "./App";
import { pickSource } from "./api";
import { SourceContext } from "./api/hooks";
import { bindHash, bindTheme } from "./state/store";

bindTheme();
bindHash();
const client = new QueryClient({ defaultOptions: { queries: { retry: 1, refetchOnWindowFocus: false } } });
const source = pickSource();

createRoot(document.getElementById("root") as HTMLElement).render(
  <StrictMode>
    <SourceContext.Provider value={source}>
      <QueryClientProvider client={client}>
        <App />
      </QueryClientProvider>
    </SourceContext.Provider>
  </StrictMode>,
);
