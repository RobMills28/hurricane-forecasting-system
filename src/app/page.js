'use client';

import dynamic from 'next/dynamic';

// Dynamically import the HurricaneTracker component
const DynamicHurricaneTracker = dynamic(
  () => import('./HurricaneTracker'),
  { ssr: false }
);

export default function Page() {
  return <DynamicHurricaneTracker />;
}