import {Shell} from "./components/Shell";
import {AboutPage} from "./pages/AboutPage";
import {ComparePage} from "./pages/ComparePage";
import {DenoisePage} from "./pages/DenoisePage";
import {EvaluationPage} from "./pages/EvaluationPage";
import {InterpolatePage} from "./pages/InterpolatePage";
import {LatentPage} from "./pages/LatentPage";
import {OverviewPage} from "./pages/OverviewPage";
import {ReconstructPage} from "./pages/ReconstructPage";

const routes:Record<string,()=>React.JSX.Element>={
  "/":OverviewPage,
  "/reconstruct":ReconstructPage,
  "/denoise":DenoisePage,
  "/latent":LatentPage,
  "/interpolate":InterpolatePage,
  "/compare":ComparePage,
  "/evaluation":EvaluationPage,
  "/about":AboutPage,
};
export default function App(){
  const path=window.location.pathname.replace(/\/$/,"")||"/";
  const Page=routes[path]??OverviewPage;
  return <Shell path={path}><Page/></Shell>;
}
