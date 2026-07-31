type IconProps = {name:string; size?:number};
const paths:Record<string,string> = {
  overview:"M4 4h6v6H4zM14 4h6v6h-6zM4 14h6v6H4zM14 14h6v6h-6z",
  reconstruct:"M4 7h16M7 4v6M17 4v6M5 13h14v7H5z",
  denoise:"M12 3v18M3 12h18M5.6 5.6l12.8 12.8M18.4 5.6 5.6 18.4",
  latent:"M5 17c3-8 11-8 14 0M7 7h.01M12 5h.01M17 8h.01M10 12h.01M16 14h.01",
  interpolate:"M5 12h14M15 8l4 4-4 4M5 8v8",
  compare:"M6 19V9M12 19V5M18 19v-7",
  evaluation:"M4 19h16M6 15l4-4 3 2 5-7",
  about:"M12 17v-5M12 8h.01M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0",
};
export function Icon({name,size=20}:IconProps) {
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d={paths[name] ?? paths.about}/></svg>;
}
