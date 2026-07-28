import { useEffect, useState, type ReactNode } from "react";
import { Shell, type RouteId } from "./components/Shell";
import { AboutPage } from "./pages/AboutPage";
import { ClassifyPage } from "./pages/ClassifyPage";
import { ConvolutionPage } from "./pages/ConvolutionPage";
import { EvaluationPage } from "./pages/EvaluationPage";
import { FeatureMapsPage } from "./pages/FeatureMapsPage";
import { OverviewPage } from "./pages/OverviewPage";
import { api, type Health } from "./lib/api";

const pathToRoute: Record<string, RouteId> = {
  "/": "overview",
  "/classify": "classify",
  "/convolution": "convolution",
  "/feature-maps": "feature-maps",
  "/evaluation": "evaluation",
  "/about": "about"
};

export function App() {
  const [path, setPath] = useState(window.location.pathname);
  const [health, setHealth] = useState<Health | null>(null);
  const [mobileOpen, setMobileOpen] = useState(false);
  useEffect(() => {
    const onPop = () => setPath(window.location.pathname);
    window.addEventListener("popstate", onPop);
    api.health().then(setHealth).catch(() => setHealth(null));
    return () => window.removeEventListener("popstate", onPop);
  }, []);
  function navigate(next: string) {
    window.history.pushState({}, "", next);
    setPath(next);
    window.scrollTo({ top: 0, behavior: "smooth" });
  }
  const active = pathToRoute[path] ?? "overview";
  const pages: Record<RouteId, ReactNode> = {
    overview: <OverviewPage health={health} navigate={navigate}/>,
    classify: <ClassifyPage/>,
    convolution: <ConvolutionPage/>,
    "feature-maps": <FeatureMapsPage/>,
    evaluation: <EvaluationPage/>,
    about: <AboutPage/>
  };
  return <Shell active={active} onNavigate={navigate} ready={health?.status === "ready"} mobileOpen={mobileOpen} setMobileOpen={setMobileOpen}>{pages[active]}</Shell>;
}
