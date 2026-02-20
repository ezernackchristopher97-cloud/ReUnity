import { trpc } from "@/lib/trpc";
import { UNAUTHED_ERR_MSG } from '@shared/const';
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { httpBatchLink, TRPCClientError } from "@trpc/client";
import { createRoot } from "react-dom/client";
import superjson from "superjson";
import App from "./App";
import { AuthProvider } from "./contexts/AuthContext";
import { getLoginUrl } from "./const";
import "./index.css";

// Prevent back navigation to external URLs (like Manus preview URL)
// This ensures users stay on reunityai.com
if (typeof window !== 'undefined') {
  // Replace the initial history state to prevent going back to external URLs
  window.history.replaceState({ reunity: true }, '', window.location.href);
  
  // Listen for popstate (back/forward navigation)
  window.addEventListener('popstate', (event) => {
    // If the state doesn't have our marker, we're navigating to an external page
    if (!event.state?.reunity) {
      // Push the current URL back and stay on the app
      window.history.pushState({ reunity: true }, '', window.location.href);
    }
  });
  
  // Also handle the case where someone tries to navigate away
  // by intercepting link clicks that go to external domains
  document.addEventListener('click', (e) => {
    const target = e.target as HTMLElement;
    const anchor = target.closest('a');
    if (anchor && anchor.href) {
      const url = new URL(anchor.href, window.location.origin);
      // Allow internal links and specific external links (like entropy-physics-ai.com)
      const allowedExternalDomains = ['entropy-physics-ai.com', 'tel:', 'mailto:'];
      const isInternal = url.origin === window.location.origin;
      const isAllowedExternal = allowedExternalDomains.some(domain => 
        anchor.href.includes(domain)
      );
      
      if (!isInternal && !isAllowedExternal && !anchor.target) {
        // For other external links, open in new tab instead of navigating away
        e.preventDefault();
        window.open(anchor.href, '_blank', 'noopener,noreferrer');
      }
    }
  });
}

const queryClient = new QueryClient();

const redirectToLoginIfUnauthorized = (error: unknown) => {
  if (!(error instanceof TRPCClientError)) return;
  if (typeof window === "undefined") return;

  const isUnauthorized = error.message === UNAUTHED_ERR_MSG;

  if (!isUnauthorized) return;

  window.location.href = getLoginUrl();
};

queryClient.getQueryCache().subscribe(event => {
  if (event.type === "updated" && event.action.type === "error") {
    const error = event.query.state.error;
    redirectToLoginIfUnauthorized(error);
    console.error("[API Query Error]", error);
  }
});

queryClient.getMutationCache().subscribe(event => {
  if (event.type === "updated" && event.action.type === "error") {
    const error = event.mutation.state.error;
    redirectToLoginIfUnauthorized(error);
    console.error("[API Mutation Error]", error);
  }
});

const trpcClient = trpc.createClient({
  links: [
    httpBatchLink({
      url: "/api/trpc",
      transformer: superjson,
      fetch(input, init) {
        return globalThis.fetch(input, {
          ...(init ?? {}),
          credentials: "include",
        });
      },
    }),
  ],
});

createRoot(document.getElementById("root")!).render(
  <trpc.Provider client={trpcClient} queryClient={queryClient}>
    <QueryClientProvider client={queryClient}>
      <AuthProvider>
        <App />
      </AuthProvider>
    </QueryClientProvider>
  </trpc.Provider>
);
