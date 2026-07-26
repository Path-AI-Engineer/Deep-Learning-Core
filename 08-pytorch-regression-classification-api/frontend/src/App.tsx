import { AppShell } from "./components/AppShell";
import { AboutPage } from "./pages/AboutPage";
import { BatchPage } from "./pages/BatchPage";
import { ClassificationPage } from "./pages/ClassificationPage";
import { ExperimentsPage } from "./pages/ExperimentsPage";
import { OverviewPage } from "./pages/OverviewPage";
import { RegressionPage } from "./pages/RegressionPage";
import { useRouter } from "./routes/Router";

export default function App() {
  const { path } = useRouter();
  const page = (() => {
    switch (path) {
      case "/regression":
        return <RegressionPage />;
      case "/classification":
        return <ClassificationPage />;
      case "/batch":
        return <BatchPage />;
      case "/experiments":
        return <ExperimentsPage />;
      case "/about":
        return <AboutPage />;
      default:
        return <OverviewPage />;
    }
  })();
  return (
    <AppShell>
      {page}
    </AppShell>
  );
}
