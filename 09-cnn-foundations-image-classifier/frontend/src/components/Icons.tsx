import type { ReactNode } from "react";

type IconProps = { name: string; size?: number };

const paths: Record<string, ReactNode> = {
  overview: <><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/></>,
  classify: <><path d="M4 7V4h3M17 4h3v3M20 17v3h-3M7 20H4v-3"/><circle cx="12" cy="12" r="4"/><path d="m10.5 12 1 1 2-2"/></>,
  convolution: <><rect x="3" y="3" width="18" height="18" rx="3"/><path d="M9 3v18M15 3v18M3 9h18M3 15h18"/></>,
  features: <><path d="m12 2 8 4-8 4-8-4 8-4Z"/><path d="m4 10 8 4 8-4M4 14l8 4 8-4M4 18l8 4 8-4"/></>,
  evaluation: <><path d="M4 19V9M10 19V5M16 19v-7M22 19H2"/></>,
  about: <><circle cx="12" cy="12" r="9"/><path d="M12 11v6M12 7h.01"/></>,
  spark: <><path d="m3 12 4-1 2-6 4 14 2-8 6-1"/></>,
  arrow: <path d="M5 12h14m-5-5 5 5-5 5"/>,
  upload: <><path d="M12 16V4m-4 4 4-4 4 4"/><path d="M4 15v4h16v-4"/></>,
  check: <path d="m5 12 4 4L19 6"/>,
  menu: <><path d="M4 7h16M4 12h16M4 17h16"/></>,
  close: <><path d="m6 6 12 12M18 6 6 18"/></>
};

export function Icon({ name, size = 20 }: IconProps) {
  return (
    <svg
      aria-hidden="true"
      className="icon"
      fill="none"
      height={size}
      viewBox="0 0 24 24"
      width={size}
      stroke="currentColor"
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeWidth="1.7"
    >
      {paths[name]}
    </svg>
  );
}
