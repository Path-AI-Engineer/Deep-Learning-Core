import {useEffect, useState} from "react";
import {api} from "./api/client";
import {Shell} from "./components/Shell";
import {AttentionMathPage} from "./pages/AttentionMathPage";
import {ExperimentsPage} from "./pages/ExperimentsPage";
import {MasksPositionsPage} from "./pages/MasksPositionsPage";
import {OverviewPage} from "./pages/OverviewPage";
import {PaperPage} from "./pages/PaperPage";
import {TracePage} from "./pages/TracePage";
import {TransductionPage} from "./pages/TransductionPage";
import type {Health} from "./types/contracts";

function route() {
  return window.location.pathname.replace(/\/+$/, "") || "/";
}

export default function App() {
  const [active, setActive] = useState(route());
  const [health, setHealth] = useState<Health | null>(null);
  const [mobileOpen, setMobileOpen] = useState(false);
  useEffect(() => {
    const onPopState = () => setActive(route());
    window.addEventListener("popstate", onPopState);
    api.health().then(setHealth).catch(() => setHealth(null));
    return () => window.removeEventListener("popstate", onPopState);
  }, []);
  function navigate(path: string) {
    window.history.pushState({}, "", path);
    setActive(path);
    window.scrollTo({top: 0, behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth"});
  }
  const page = (() => {
    switch (active) {
      case "/attention-math": return <AttentionMathPage />;
      case "/masks-positions": return <MasksPositionsPage />;
      case "/architecture-trace": return <TracePage />;
      case "/transduction": return <TransductionPage />;
      case "/attention-explorer": return <TracePage explorer />;
      case "/experiments": return <ExperimentsPage />;
      case "/paper": return <PaperPage />;
      default: return <OverviewPage navigate={navigate} />;
    }
  })();
  return <Shell active={active} navigate={navigate} health={health} mobileOpen={mobileOpen} setMobileOpen={setMobileOpen}>{page}</Shell>;
}
