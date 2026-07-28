import { useEffect, useState } from "react";
import { Shell } from "./components/Shell";
import { AboutPage } from "./pages/AboutPage";
import { ClassifyPage } from "./pages/ClassifyPage";
import { ComparePage } from "./pages/ComparePage";
import { EvaluationPage } from "./pages/EvaluationPage";
import { OverviewPage } from "./pages/OverviewPage";
import { SequenceLabPage } from "./pages/SequenceLabPage";

const routes: Record<string, (navigate: (path: string) => void) => React.ReactNode> = {
  "/": (navigate) => <OverviewPage navigate={navigate} />,
  "/classify": (navigate) => <ClassifyPage navigate={navigate} />,
  "/sequence-lab": () => <SequenceLabPage />,
  "/compare": () => <ComparePage />,
  "/evaluation": () => <EvaluationPage />,
  "/about": () => <AboutPage />
};

export default function App() {
  const [path, setPath] = useState(window.location.pathname);
  useEffect(() => {
    const update = () => setPath(window.location.pathname);
    window.addEventListener("popstate", update);
    return () => window.removeEventListener("popstate", update);
  }, []);
  const navigate = (target: string) => {
    if (target === path) return;
    window.history.pushState({}, "", target);
    setPath(target);
    window.scrollTo({ top: 0, behavior: "smooth" });
  };
  return <Shell path={routes[path] ? path : "/"} navigate={navigate}>{(routes[path] ?? routes["/"])(navigate)}</Shell>;
}
