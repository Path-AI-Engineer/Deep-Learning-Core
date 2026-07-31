export function Icon({name}: {name: string}) {
  const paths: Record<string, React.ReactNode> = {
    overview: <><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/></>,
    sigma: <><path d="M18 4H6l6 8-6 8h12"/><path d="M9 12h6"/></>,
    mask: <><path d="M4 5h16v14H4z"/><path d="M4 10h16M10 5v14"/><path d="M12 12h5v5h-5z"/></>,
    trace: <><circle cx="5" cy="12" r="2"/><circle cx="12" cy="5" r="2"/><circle cx="19" cy="12" r="2"/><circle cx="12" cy="19" r="2"/><path d="m6.5 10.5 4-4m3 0 4 4m0 3-4 4m-3 0-4-4"/></>,
    sequence: <><path d="M4 7h12M4 12h16M4 17h9"/><path d="m17 15 3 2-3 2"/></>,
    attention: <><path d="M4 19 19 4"/><circle cx="6" cy="6" r="2"/><circle cx="18" cy="18" r="2"/><path d="M9 5h8v8"/></>,
    experiment: <><path d="M5 20V10m7 10V4m7 16v-7"/><path d="M3 20h18"/></>,
    paper: <><path d="M6 3h9l4 4v14H6z"/><path d="M15 3v5h4M9 12h6M9 16h6"/></>,
    menu: <path d="M4 7h16M4 12h16M4 17h16"/>,
    close: <path d="m6 6 12 12M18 6 6 18"/>,
    arrow: <path d="M5 12h14m-5-5 5 5-5 5"/>,
    layers: <><path d="m12 3 9 5-9 5-9-5z"/><path d="m3 12 9 5 9-5M3 16l9 5 9-5"/></>,
  };
  return (
    <svg aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
      {paths[name] ?? paths.layers}
    </svg>
  );
}

