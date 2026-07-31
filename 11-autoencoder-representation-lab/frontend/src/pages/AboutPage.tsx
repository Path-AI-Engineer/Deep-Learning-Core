import {Eyebrow, Panel} from "../components/Primitives";

export function AboutPage(){
  return <div className="page"><div className="page-title"><div><Eyebrow>Architecture, evidence and limits</Eyebrow><h1>A laboratory for questions—not certainty.</h1><p>The system separates self-supervised reconstruction from supervised representation evaluation.</p></div></div><div className="about-grid">
    <Panel><span className="card-index">TRAIN</span><h2>Labels stay outside</h2><p>Autoencoders see only image tensors and reconstruction targets. Labels appear later in equal frozen linear-probe protocols and evaluation overlays.</p></Panel>
    <Panel><span className="card-index">LOAD</span><h2>Immutable CPU bundles</h2><p>Architecture configuration is reconstructed before loading state dictionaries with hash validation and inference-only execution.</p></Panel>
    <Panel><span className="card-index">DATA</span><h2>FashionMNIST target</h2><p>The official pipeline preserves its test split and creates deterministic stratified training and validation indices. The running release uses a clearly labeled offline fixture.</p></Panel>
    <Panel><span className="card-index">PRIVACY</span><h2>Ephemeral uploads</h2><p>Allowed image formats are decoded in memory, converted to grayscale, resized to 28×28 and discarded after the response.</p></Panel>
  </div><Panel className="limits-list"><Eyebrow>Interpretive limits</Eyebrow><h2>This project does not claim</h2><ul><li>that latent distance proves semantic equivalence;</li><li>that low MSE guarantees a useful representation;</li><li>that linear-probe performance demonstrates causality;</li><li>that arbitrary latent coordinates produce valid images;</li><li>that interpolation is probabilistic generation;</li><li>that reconstruction error is production anomaly detection.</li></ul></Panel></div>;
}
