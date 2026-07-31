import {useMemo, useState} from "react";
import {Heatmap} from "../components/Heatmap";
import {PageHeader, Panel, PanelHeader, Status} from "../components/Primitives";

export function MasksPositionsPage() {
  const [length, setLength] = useState(6);
  const [position, setPosition] = useState(3);
  const matrix = useMemo(() => Array.from({length}, (_, row) => Array.from({length}, (_, column) => column <= row ? 1 : 0)), [length]);
  const encoding = useMemo(() => Array.from({length}, (_, row) => Array.from({length: 8}, (_, dimension) => {
    const base = row / Math.pow(10000, (2 * Math.floor(dimension / 2)) / 8);
    return dimension % 2 === 0 ? Math.sin(base) : Math.cos(base);
  })), [length]);
  return <>
    <PageHeader eyebrow="Leakage control · position signal" title="Make visibility explicit before decoding." copy="Causal masking and positional encoding solve different problems: one restricts information flow; the other makes order observable." meta={<Status tone="warning">True = visible in this UI</Status>} />
    <div className="control-strip"><label>Sequence length <input type="range" min="3" max="10" value={length} onChange={event => {const next = Number(event.target.value); setLength(next); setPosition(Math.min(position, next - 1));}} /><strong>{length}</strong></label><label>Decoder position <input type="range" min="0" max={length - 1} value={position} onChange={event => setPosition(Number(event.target.value))} /><strong>{position}</strong></label></div>
    <div className="two-column wide-left">
      <Panel><PanelHeader label="Causal contract" title={`Position ${position} can observe 0…${position}`} /><Heatmap matrix={matrix} rows={matrix.map((_, index) => `q${index}`)} columns={matrix.map((_, index) => `k${index}`)} label="Causal visibility matrix, one means visible" /><p className="panel-note">Any mass above the diagonal would be future leakage. Padding masks are combined separately over key positions.</p></Panel>
      <Panel><PanelHeader label="Selected query" title="Visible and blocked positions" /><div className="visibility-list">{matrix[position].map((visible, index) => <div className={visible ? "visible" : "blocked"} key={index}><span>k{index}</span><strong>{visible ? "Visible" : "Blocked"}</strong></div>)}</div></Panel>
    </div>
    <Panel><PanelHeader label="Sinusoidal position" title="Frequency changes by feature pair" action={<Status tone="neutral">No gradients</Status>} /><Heatmap matrix={encoding.map(row => row.map(value => (value + 1) / 2))} rows={encoding.map((_, index) => `p${index}`)} columns={encoding[0].map((_, index) => `d${index}`)} label="Normalized sinusoidal positional encoding values" /><p className="panel-note">The heatmap maps [-1, 1] to [0, 1] for display. The model receives the original signed values.</p></Panel>
  </>;
}

