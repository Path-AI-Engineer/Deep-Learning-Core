import type { SVGProps } from "react";

type IconProps = SVGProps<SVGSVGElement>;

const base = {
  viewBox: "0 0 24 24",
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 1.8,
  strokeLinecap: "round" as const,
  strokeLinejoin: "round" as const,
  "aria-hidden": true
};

export function PulseIcon(props: IconProps) {
  return (
    <svg {...base} {...props}>
      <path d="M3 12h3l2-6 4 12 2.5-8 2 4H21" />
    </svg>
  );
}

export function GridIcon(props: IconProps) {
  return (
    <svg {...base} {...props}>
      <rect x="3" y="3" width="7" height="7" rx="2" />
      <rect x="14" y="3" width="7" height="7" rx="2" />
      <rect x="3" y="14" width="7" height="7" rx="2" />
      <rect x="14" y="14" width="7" height="7" rx="2" />
    </svg>
  );
}

export function ActivityIcon(props: IconProps) {
  return (
    <svg {...base} {...props}>
      <path d="M4 17c3-8 5 1 8-8s4 4 8-3" />
      <circle cx="4" cy="17" r="1.2" />
      <circle cx="12" cy="9" r="1.2" />
      <circle cx="20" cy="6" r="1.2" />
    </svg>
  );
}

export function MemoryIcon(props: IconProps) {
  return (
    <svg {...base} {...props}>
      <rect x="5" y="5" width="14" height="14" rx="4" />
      <path d="M9 9h6v6H9zM9 2v3m6-3v3M9 19v3m6-3v3M2 9h3m-3 6h3m14-6h3m-3 6h3" />
    </svg>
  );
}

export function CompareIcon(props: IconProps) {
  return (
    <svg {...base} {...props}>
      <path d="M7 4v14M17 6v14M4 7l3-3 3 3M14 17l3 3 3-3" />
    </svg>
  );
}

export function MatrixIcon(props: IconProps) {
  return (
    <svg {...base} {...props}>
      <path d="M5 3H3v18h2M19 3h2v18h-2" />
      <circle cx="9" cy="8" r="1" />
      <circle cx="15" cy="8" r="1" />
      <circle cx="9" cy="16" r="1" />
      <circle cx="15" cy="16" r="1" />
    </svg>
  );
}

export function InfoIcon(props: IconProps) {
  return (
    <svg {...base} {...props}>
      <circle cx="12" cy="12" r="9" />
      <path d="M12 11v5m0-8h.01" />
    </svg>
  );
}

export function ArrowIcon(props: IconProps) {
  return (
    <svg {...base} {...props}>
      <path d="M5 12h14m-5-5 5 5-5 5" />
    </svg>
  );
}
