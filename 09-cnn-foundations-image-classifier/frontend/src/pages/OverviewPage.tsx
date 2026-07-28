import type { Health } from "../lib/api";
import { Icon } from "../components/Icons";
import { Metric, PageHeader, Panel } from "../components/Primitives";
import { SpatialNetwork } from "../components/SpatialNetwork";

export function OverviewPage({ health, navigate }: { health: Health | null; navigate: (path: string) => void }) {
  return (
    <>
      <PageHeader
        eyebrow="IMAGE CLASSIFICATION · INSPECTABLE BY DESIGN"
        title="See what the network sees."
        description="A controlled FashionMNIST laboratory that connects convolution mechanics, learned spatial features and honest evaluation evidence."
        aside={<SpatialNetwork/>}
      />
      <div className="metrics-grid">
        <Metric label="Input contract" value="1 × 28 × 28" detail="Grayscale NCHW tensor"/>
        <Metric label="Output space" value="10 classes" detail="Stable FashionMNIST mapping"/>
        <Metric label="Inference" value={health?.model_available ? "Available" : "Pending bundle"} detail={health?.model_version ?? "No trained asset loaded"}/>
        <Metric label="Training mode" value="Disabled" detail="Serving is inference-only"/>
      </div>
      <div className="overview-grid">
        <Panel eyebrow="LEARNING PATH" title="From pixels to evidence" className="journey-panel">
          <div className="journey">
            {[
              ["01", "Convolve", "Inspect local cross-correlation and verify it against PyTorch."],
              ["02", "Classify", "Run the approved CNN against a controlled test sample or upload."],
              ["03", "Inspect", "Capture selected feature maps without claiming causal explanation."],
              ["04", "Evaluate", "Compare CNN and MLP under the same split and test policy."]
            ].map(([number, title, copy]) => <div className="journey-step" key={number}><span>{number}</span><div><strong>{title}</strong><p>{copy}</p></div></div>)}
          </div>
        </Panel>
        <Panel eyebrow="MODEL TOPOLOGY" title="A small CNN you can reason about">
          <div className="network-map" aria-label="CNN architecture diagram">
            {[
              ["Input", "1 × 28 × 28"],
              ["Conv block 1", "16 × 14 × 14"],
              ["Conv block 2", "32 × 7 × 7"],
              ["Classifier", "10 logits"]
            ].map(([name, shape], index) => <div className="network-node" key={name}><span>{index + 1}</span><div><strong>{name}</strong><small>{shape}</small></div></div>)}
          </div>
          <button className="text-action" onClick={() => navigate("/convolution")}>Open convolution lab <Icon name="arrow" size={18}/></button>
        </Panel>
      </div>
    </>
  );
}
