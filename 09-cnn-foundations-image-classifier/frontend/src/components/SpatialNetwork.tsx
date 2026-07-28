import { useState, type CSSProperties } from "react";

const layers = [
  { id: "input", label: "Input", shape: "1 × 28 × 28", depth: 0, size: 118 },
  { id: "conv1", label: "Conv block 1", shape: "16 × 14 × 14", depth: 56, size: 96 },
  { id: "conv2", label: "Conv block 2", shape: "32 × 7 × 7", depth: 112, size: 72 },
  { id: "logits", label: "Class logits", shape: "10", depth: 168, size: 42 }
] as const;

export function SpatialNetwork() {
  const [active, setActive] = useState<(typeof layers)[number]["id"]>("conv2");
  const selected = layers.find((layer) => layer.id === active)!;
  return (
    <div className="spatial-card">
      <div className="spatial-copy">
        <span>INTERACTIVE TENSOR VIEW</span>
        <strong>{selected.label}</strong>
        <small>{selected.shape}</small>
      </div>
      <div className="spatial-scene" role="group" aria-label="Interactive CNN layer stack">
        <div className="spatial-axis" aria-hidden="true"/>
        {layers.map((layer, index) => (
          <button
            aria-label={`${layer.label}, tensor shape ${layer.shape}`}
            aria-pressed={active === layer.id}
            className={active === layer.id ? "tensor-plane tensor-plane--active" : "tensor-plane"}
            key={layer.id}
            onClick={() => setActive(layer.id)}
            style={{
              "--plane-depth": `${layer.depth}px`,
              "--plane-size": `${layer.size}px`,
              "--plane-index": index
            } as CSSProperties}
          >
            <i/><i/><i/><i/>
          </button>
        ))}
      </div>
      <p>Choose a plane to inspect how spatial resolution contracts while channel depth grows.</p>
    </div>
  );
}
