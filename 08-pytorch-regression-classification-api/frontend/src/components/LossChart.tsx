export function LossChart({
  train,
  validation,
}: {
  train: number[];
  validation: number[];
}) {
  const width = 720;
  const height = 220;
  const all = [...train, ...validation];
  const maximum = Math.max(...all, 1e-6);
  const minimum = Math.min(...all, 0);
  const points = (values: number[]) =>
    values
      .map((value, index) => {
        const x = (index / Math.max(values.length - 1, 1)) * width;
        const y = height - ((value - minimum) / Math.max(maximum - minimum, 1e-6)) * height;
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(" ");

  return (
    <div className="chart-shell">
      <div className="chart-legend">
        <span className="train-line">Training loss</span>
        <span className="validation-line">Validation loss</span>
      </div>
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Training and validation loss">
        <line x1="0" y1={height} x2={width} y2={height} className="axis" />
        <polyline points={points(train)} className="loss train" />
        <polyline points={points(validation)} className="loss validation" />
      </svg>
    </div>
  );
}
