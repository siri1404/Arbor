'use client'

import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { ChartContainer, ChartTooltip, type ChartConfig } from '@/components/ui/chart'
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, ResponsiveContainer } from 'recharts'
import type { StockTimeSeries } from '@/lib/types'
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs'

interface StockChartProps {
  data: StockTimeSeries | null
  isLoading?: boolean
  symbol?: string
  interval?: string
  onIntervalChange?: (interval: string) => void
}

const chartConfig = {
  close: {
    label: 'Price',
    color: 'var(--chart-1)',
  },
} satisfies ChartConfig

export function StockChart({ data, isLoading, symbol, interval = 'daily', onIntervalChange }: StockChartProps) {
  if (isLoading) {
    return (
      <Card className="h-full">
        <CardHeader>
          <CardTitle>Price Chart</CardTitle>
          <CardDescription>Loading chart data...</CardDescription>
        </CardHeader>
        <CardContent className="h-[300px] flex items-center justify-center">
          <div className="animate-pulse flex flex-col items-center gap-2">
            <div className="h-4 w-32 rounded bg-muted" />
            <div className="h-[200px] w-full rounded bg-muted" />
          </div>
        </CardContent>
      </Card>
    )
  }

  if (!data || data.data.length === 0) {
    return (
      <Card className="h-full">
        <CardHeader>
          <CardTitle>Price Chart</CardTitle>
          <CardDescription>No data available</CardDescription>
        </CardHeader>
        <CardContent className="h-[300px] flex items-center justify-center">
          <p className="text-muted-foreground">Select a stock to view chart</p>
        </CardContent>
      </Card>
    )
  }

  const chartData = data.data.map((point) => ({
    date: point.date,
    close: point.close,
    displayDate: new Date(point.date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
  }))

  const minPrice = Math.min(...chartData.map((d) => d.close)) * 0.995
  const maxPrice = Math.max(...chartData.map((d) => d.close)) * 1.005

  const firstPrice = chartData[0]?.close || 0
  const lastPrice = chartData[chartData.length - 1]?.close || 0
  const priceChange = lastPrice - firstPrice
  const percentChange = firstPrice > 0 ? ((priceChange / firstPrice) * 100).toFixed(2) : '0.00'
  const isPositive = priceChange >= 0

  return (
    <Card className="h-full">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <div>
            <CardTitle>{symbol || data.symbol} Price Chart</CardTitle>
            <CardDescription className={isPositive ? 'text-success' : 'text-destructive'}>
              {isPositive ? '+' : ''}
              {percentChange}% over period
            </CardDescription>
          </div>
          {onIntervalChange && (
            <Tabs value={interval} onValueChange={onIntervalChange}>
              <TabsList>
                <TabsTrigger value="intraday">1D</TabsTrigger>
                <TabsTrigger value="daily">1M</TabsTrigger>
                <TabsTrigger value="weekly">1Y</TabsTrigger>
              </TabsList>
            </Tabs>
          )}
        </div>
      </CardHeader>
      <CardContent>
        <ChartContainer config={chartConfig} className="h-[300px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={chartData} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
              <defs>
                <linearGradient id="colorPrice" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor={isPositive ? 'var(--chart-1)' : 'var(--chart-2)'} stopOpacity={0.3} />
                  <stop offset="95%" stopColor={isPositive ? 'var(--chart-1)' : 'var(--chart-2)'} stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" className="stroke-border/30" />
              <XAxis
                dataKey="displayDate"
                tickLine={false}
                axisLine={false}
                className="text-xs text-muted-foreground"
                tick={{ fill: 'var(--muted-foreground)' }}
                interval="preserveStartEnd"
              />
              <YAxis
                domain={[minPrice, maxPrice]}
                tickLine={false}
                axisLine={false}
                className="text-xs text-muted-foreground"
                tick={{ fill: 'var(--muted-foreground)' }}
                tickFormatter={(value) => `$${value.toFixed(0)}`}
                width={60}
              />
              <ChartTooltip
                content={({ active, payload, label }) => {
                  if (!active || !payload?.length) return null
                  return (
                    <div className="rounded-lg border border-border/50 bg-background px-3 py-2 shadow-xl">
                      <p className="text-xs text-muted-foreground mb-1">Date: {label}</p>
                      <p className="text-sm font-medium text-foreground">
                        ${Number(payload[0]?.value).toFixed(2)}
                      </p>
                    </div>
                  )
                }}
              />
              <Area
                type="monotone"
                dataKey="close"
                stroke={isPositive ? 'var(--chart-1)' : 'var(--chart-2)'}
                strokeWidth={2}
                fill="url(#colorPrice)"
              />
            </AreaChart>
          </ResponsiveContainer>
        </ChartContainer>
      </CardContent>
    </Card>
  )
}
