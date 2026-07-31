export function Heatmap({matrix, rows, columns, label}: {matrix: number[][]; rows: string[]; columns: string[]; label: string}) {
  const maximum = Math.max(...matrix.flat(), 0.0001);
  return (
    <div className="heatmap-wrap">
      <div className="heatmap" role="img" aria-label={label} style={{gridTemplateColumns: `minmax(72px, auto) repeat(${columns.length}, minmax(36px, 1fr))`}}>
        <span />
        {columns.map((column, index) => <span className="axis top" key={`${column}-${index}`}>{column.replace("SYMBOL_", "S")}</span>)}
        {matrix.map((row, rowIndex) => [
          <span className="axis side" key={`axis-${rowIndex}`}>{(rows[rowIndex] ?? `q${rowIndex}`).replace("SYMBOL_", "S")}</span>,
          ...row.map((value, columnIndex) => <span className="heat-cell" title={`${value.toFixed(6)}`} style={{"--heat": `${value / maximum}`} as React.CSSProperties} key={`${rowIndex}-${columnIndex}`}>{value.toFixed(2)}</span>),
        ])}
      </div>
      <details className="accessible-table">
        <summary>Open numerical attention table</summary>
        <table><thead><tr><th>Query</th>{columns.map((column, index) => <th key={`${column}-${index}`}>{column}</th>)}</tr></thead>
        <tbody>{matrix.map((row, rowIndex) => <tr key={rowIndex}><th>{rows[rowIndex] ?? `q${rowIndex}`}</th>{row.map((value, columnIndex) => <td key={columnIndex}>{value.toFixed(6)}</td>)}</tr>)}</tbody></table>
      </details>
    </div>
  );
}

