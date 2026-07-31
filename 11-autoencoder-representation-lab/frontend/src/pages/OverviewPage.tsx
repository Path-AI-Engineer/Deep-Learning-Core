import {api} from "../api/client";
import {Eyebrow, Panel, Pill, State} from "../components/Primitives";
import {useResource} from "../hooks/useResource";
import type {ModelRow} from "../types/contracts";

export function OverviewPage() {
  const health=useResource(()=>api.get<{status:string;data_mode:string;active_model:string;bundles_available:string[]}>("/health"));
  const models=useResource(()=>api.get<{items:ModelRow[]}>("/models"));
  return <div className="page">
    <section className="hero">
      <div className="hero-copy"><Eyebrow>Representation learning · controlled evidence</Eyebrow><h1>What survives the <em>bottleneck?</em></h1><p>Compress a 28×28 image, inspect what the decoder recovers, and test whether the learned code preserves linearly accessible structure.</p><div className="hero-actions"><a className="button primary" href="/reconstruct">Open reconstruction lab</a><a className="button ghost" href="/latent">Explore latent space</a></div></div>
      <div className="network-visual" aria-label="Encoder bottleneck decoder diagram"><div className="layer input"><span/><span/><span/><span/></div><div className="links"/><div className="bottleneck"><strong>z</strong><small>16D</small></div><div className="links reverse"/><div className="layer output"><span/><span/><span/><span/></div><div className="visual-labels"><span>784 pixels</span><span>latent code</span><span>784 pixels</span></div></div>
    </section>
    <State loading={health.loading||models.loading} error={health.error||models.error}/>
    {health.data && models.data && <><div className="status-ribbon"><Pill tone="good">API {health.data.status}</Pill><span>{health.data.data_mode.replace("_"," ")}</span><span>Active: {health.data.active_model}</span><span>{models.data.items.length} representations</span></div>
    <section className="question-grid">
      <Panel><span className="card-index">01</span><h2>Reconstruction</h2><p>MSE, MAE, PSNR and SSIM measure different aspects of pixel recovery.</p></Panel>
      <Panel><span className="card-index">02</span><h2>Representation</h2><p>A frozen encoder and equal linear-probe protocol test accessible class information.</p></Panel>
      <Panel><span className="card-index">03</span><h2>Robustness</h2><p>Matched corruptions compare standard and denoising autoencoders against clean targets.</p></Panel>
    </section></>}
    <Panel className="evidence-boundary"><div><Eyebrow>Evidence boundary</Eyebrow><h2>A strong baseline is not a failure.</h2></div><p>PCA currently reconstructs this deterministic fixture better than the neural models. The application keeps that conflict visible. Official FashionMNIST training is a separate, reproducible workflow.</p></Panel>
  </div>;
}
