import type { NextConfig } from "next";

const nextConfig: NextConfig = {
   // Disable ESLint
   eslint: {
    ignoreDuringBuilds: true,
  },
  // Disable TypeScript type checking
  typescript: {
    ignoreBuildErrors: true,
  },
  /* config options here */
};

export default nextConfig;
