'use client'

import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import type { CompanyOverview } from '@/lib/types'
import { Building2, TrendingUp, DollarSign, BarChart3 } from 'lucide-react'

interface CompanyInfoProps {
  overview: CompanyOverview | null
  isLoading?: boolean
}

export function CompanyInfo({ overview, isLoading }: CompanyInfoProps) {
  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <div className="h-6 w-48 animate-pulse rounded bg-muted" />
          <div className="h-4 w-32 animate-pulse rounded bg-muted" />
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="h-16 w-full animate-pulse rounded bg-muted" />
            <div className="grid grid-cols-2 gap-4">
              {[1, 2, 3, 4].map((i) => (
                <div key={i} className="h-20 animate-pulse rounded bg-muted" />
              ))}
            </div>
          </div>
        </CardContent>
      </Card>
    )
  }

  if (!overview) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Building2 className="h-5 w-5 text-primary" />
            Company Overview
          </CardTitle>
          <CardDescription>Select a stock to view company details</CardDescription>
        </CardHeader>
        <CardContent>
          <p className="text-muted-foreground text-sm">
            Click on any stock card above to see detailed company information.
          </p>
        </CardContent>
      </Card>
    )
  }

  const formatMarketCap = (value: number) => {
    if (value >= 1e12) return `$${(value / 1e12).toFixed(2)}T`
    if (value >= 1e9) return `$${(value / 1e9).toFixed(2)}B`
    if (value >= 1e6) return `$${(value / 1e6).toFixed(2)}M`
    return `$${value.toLocaleString()}`
  }

  const metrics = [
    {
      label: 'Market Cap',
      value: formatMarketCap(overview.marketCap),
      icon: DollarSign,
    },
    {
      label: 'P/E Ratio',
      value: overview.peRatio > 0 ? overview.peRatio.toFixed(2) : 'N/A',
      icon: BarChart3,
    },
    {
      label: 'EPS',
      value: overview.eps > 0 ? `$${overview.eps.toFixed(2)}` : 'N/A',
      icon: TrendingUp,
    },
    {
      label: 'Beta',
      value: overview.beta > 0 ? overview.beta.toFixed(2) : 'N/A',
      icon: BarChart3,
    },
  ]

  return null
}
