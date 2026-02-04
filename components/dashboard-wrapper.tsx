'use client'

import dynamic from 'next/dynamic'

const AppDashboard = dynamic(
  () => import('@/components/app-dashboard').then((mod) => ({ default: mod.AppDashboard })),
  { ssr: false }
)

export function DashboardWrapper() {
  return <AppDashboard />
}
