const PALETTE = ["#5eead4", "#60a5fa", "#a78bfa", "#f59e0b", "#fb7185", "#8bffb0"];

function linePath(values: number[], width: number, height: number, min?: number, max?: number) {
  const lower = min ?? Math.min(...values);
  const upper = max ?? Math.max(...values);
  const range = Math.max(upper - lower, 0.000001);
  return values
    .map((value, index) => {
      const x = (index / Math.max(values.length - 1, 1)) * width;
      const y = height - ((value - lower) / range) * height;
      return `${index === 0 ? "M" : "L"}${x.toFixed(2)},${y.toFixed(2)}`;
    })
    .join(" ");
}

export function SignalChart({
  signals,
  channels,
  selected
}: {
  signals: number[][];
  channels: string[];
  selected: number[];
}) {
  const flattened = selected.flatMap((index) => signals[index] ?? []);
  const min = Math.min(...flattened);
  const max = Math.max(...flattened);
  return (
    <div className="chart-frame">
      <svg viewBox="0 0 960 300" role="img" aria-label="Selected inertial signals across 128 timesteps">
        <g className="grid-lines">
          {[0, 1, 2, 3, 4].map((line) => (
            <line key={line} x1="0" y1={line * 75} x2="960" y2={line * 75} />
          ))}
        </g>
        {selected.map((channelIndex, order) => (
          <path
            key={channelIndex}
            d={linePath(signals[channelIndex], 960, 300, min, max)}
            stroke={PALETTE[order % PALETTE.length]}
            className="signal-line"
          />
        ))}
      </svg>
      <div className="chart-legend">
        {selected.map((channelIndex, order) => (
          <span key={channelIndex}>
            <i style={{ background: PALETTE[order % PALETTE.length] }} />
            {channels[channelIndex]}
          </span>
        ))}
      </div>
    </div>
  );
}

export function MiniLineChart({
  series,
  labels
}: {
  series: Array<{ name: string; values: number[] }>;
  labels?: string[];
}) {
  const all = series.flatMap((item) => item.values);
  const min = Math.min(...all);
  const max = Math.max(...all);
  return (
    <div className="mini-chart">
      <svg viewBox="0 0 720 220" role="img" aria-label="Comparison line chart">
        <g className="grid-lines">
          {[0, 1, 2, 3, 4].map((line) => (
            <line key={line} x1="0" y1={line * 55} x2="720" y2={line * 55} />
          ))}
        </g>
        {series.map((item, index) => (
          <path
            key={item.name}
            d={linePath(item.values, 720, 220, min, max)}
            stroke={PALETTE[index % PALETTE.length]}
            className="signal-line"
          />
        ))}
      </svg>
      <div className="chart-legend">
        {series.map((item, index) => (
          <span key={item.name}>
            <i style={{ background: PALETTE[index % PALETTE.length] }} />
            {item.name}
          </span>
        ))}
        {labels && <span className="axis-caption">{labels[0]} → {labels.at(-1)}</span>}
      </div>
    </div>
  );
}

export function ConfusionMatrix({ values, labels }: { values: number[][]; labels: string[] }) {
  const max = Math.max(...values.flat(), 1);
  return (
    <div className="matrix-wrap">
      <div className="matrix-label-y">Actual class</div>
      <div className="matrix-grid" style={{ gridTemplateColumns: `90px repeat(${labels.length}, 1fr)` }}>
        <span />
        {labels.map((label) => <span className="matrix-axis" key={label}>{label.slice(0, 5)}</span>)}
        {values.map((row, rowIndex) => (
          <div className="matrix-row" key={labels[rowIndex]} style={{ display: "contents" }}>
            <span className="matrix-axis matrix-axis-row">{labels[rowIndex]}</span>
            {row.map((value, columnIndex) => (
              <span
                className="matrix-cell"
                key={`${rowIndex}-${columnIndex}`}
                style={{ "--intensity": value / max } as React.CSSProperties}
                title={`${labels[rowIndex]} predicted as ${labels[columnIndex]}: ${value}`}
              >
                {value}
              </span>
            ))}
          </div>
        ))}
      </div>
      <div className="matrix-label-x">Predicted class</div>
    </div>
  );
}
