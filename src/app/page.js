'use client';

import dynamic from 'next/dynamic';

// Dynamically import the HurricaneTracker component
const DynamicHurricaneTracker = dynamic(
  () => import('./HurricaneTracker'),
  { 
    ssr: false,
    loading: () => (
      <div className="flex items-center justify-center h-screen bg-[#0B1021] text-white">
        <div className="text-xl">Loading Atlas Command Center...</div>
      </div>
    )
  }
);

export default function Page() {
  return <DynamicHurricaneTracker />;
}