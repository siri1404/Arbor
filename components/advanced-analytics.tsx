'use client'

import React from "react"

import { useState, useMemo } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { ChartContainer, ChartTooltip } from '@/components/ui/chart'
import {
  LineChart, Line, XAxis, YAxis, ResponsiveContainer, ReferenceLine,
  AreaChart, Area, BarChart, Bar, ComposedChart, CartesianGrid, Tooltip
} from 'recharts'
import type { TimeSeriesDataPoint, StockQuote } from '@/lib/types'
import { TrendingUp, TrendingDown, Activity, BarChart3, Zap, Target, AlertTriangle, Gauge } from 'lucide-react'
import { cn } from '@/lib/utils'

interface AdvancedAnalyticsProps {
  symbol: string
  data: TimeSeriesDataPoint[]
  quote: StockQuote | null
}

// Color palette that works with dark theme
const COLORS = {
  primary: '#22c55e',
  secondary: '#3b82f6', 
  accent: '#f59e0b',
  destructive: '#ef4444',
  purple: '#a855f7',
  cyan: '#06b6d4',
  pink: '#ec4899',
  muted: '#6b7280',
}

// Calculate SMA
function calculateSMA(data: number[], period: number): (number | undefined)[] {
  const result: (number | undefined)[] = []
  for (let i = 0; i < data.length; i++) {
    if (i < period - 1) result.push(undefined)
    else {
      const sum = data.slice(i - period + 1, i + 1).reduce((a, b) => a + b, 0)
      result.push(sum / period)
    }
  }
  return result
}

// Calculate EMA
function calculateEMA(data: number[], period: number): (number | undefined)[] {
  const result: (number | undefined)[] = []
  const multiplier = 2 / (period + 1)
  for (let i = 0; i < data.length; i++) {
    if (i < period - 1) result.push(undefined)
    else if (i === period - 1) {
      result.push(data.slice(0, period).reduce((a, b) => a + b, 0) / period)
    } else {
      const prev = result[i - 1]
      if (prev !== undefined) result.push((data[i] - prev) * multiplier + prev)
    }
  }
  return result
}

// Calculate RSI
function calculateRSI(data: number[], period: number = 14): (number | undefined)[] {
  const result: (number | undefined)[] = []
  const gains: number[] = []
  const losses: number[] = []
  for (let i = 1; i < data.length; i++) {
    const change = data[i] - data[i - 1]
    gains.push(change > 0 ? change : 0)
    losses.push(change < 0 ? Math.abs(change) : 0)
  }
  for (let i = 0; i < data.length; i++) {
    if (i < period) result.push(undefined)
    else {
      const avgGain = gains.slice(i - period, i).reduce((a, b) => a + b, 0) / period
      const avgLoss = losses.slice(i - period, i).reduce((a, b) => a + b, 0) / period
      if (avgLoss === 0) result.push(100)
      else result.push(100 - (100 / (1 + avgGain / avgLoss)))
    }
  }
  return result
}

// Calculate Bollinger Bands
function calculateBollingerBands(data: number[], period: number = 20, stdDev: number = 2) {
  const sma = calculateSMA(data, period)
  const upper: (number | undefined)[] = []
  const lower: (number | undefined)[] = []
  for (let i = 0; i < data.length; i++) {
    if (i < period - 1 || sma[i] === undefined) {
      upper.push(undefined)
      lower.push(undefined)
    } else {
      const slice = data.slice(i - period + 1, i + 1)
      const mean = sma[i]!
      const variance = slice.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / period
      const std = Math.sqrt(variance)
      upper.push(mean + stdDev * std)
      lower.push(mean - stdDev * std)
    }
  }
  return { middle: sma, upper, lower }
}

// Calculate returns
function calculateReturns(data: number[]): number[] {
  const returns: number[] = [0]
  for (let i = 1; i < data.length; i++) {
    returns.push(((data[i] - data[i-1]) / data[i-1]) * 100)
  }
  return returns
}

// Calculate cumulative returns
function calculateCumulativeReturns(data: number[]): number[] {
  if (data.length === 0) return []
  const initial = data[0]
  return data.map(d => ((d - initial) / initial) * 100)
}

// Calculate volatility (annualized)
function calculateVolatility(returns: number[]): number {
  const mean = returns.reduce((a, b) => a + b, 0) / returns.length
  const variance = returns.reduce((sum, r) => sum + Math.pow(r - mean, 2), 0) / returns.length
  return Math.sqrt(variance) * Math.sqrt(252) // Annualized
}

// Calculate Sharpe Ratio
function calculateSharpeRatio(returns: number[], riskFreeRate: number = 0.02): number {
  const meanReturn = returns.reduce((a, b) => a + b, 0) / returns.length
  const annualizedReturn = meanReturn * 252
  const vol = calculateVolatility(returns)
  if (vol === 0) return 0
  return (annualizedReturn - riskFreeRate) / vol
}

// Calculate Max Drawdown
function calculateMaxDrawdown(data: number[]): number {
  let maxDrawdown = 0
  let peak = data[0]
  for (const value of data) {
    if (value > peak) peak = value
    const drawdown = (peak - value) / peak
    if (drawdown > maxDrawdown) maxDrawdown = drawdown
  }
  return maxDrawdown * 100
}

export function AdvancedAnalytics({ symbol, data, quote }: AdvancedAnalyticsProps) {
  const [hoveredMetric, setHoveredMetric] = useState<string | null>(null)

  const analytics = useMemo(() => {
    if (!data || data.length === 0) return null

    const closes = data.map(d => d.close)
    const volumes = data.map(d => d.volume)
    const highs = data.map(d => d.high)
    const lows = data.map(d => d.low)
    
    const sma20 = calculateSMA(closes, 20)
    const sma50 = calculateSMA(closes, 50)
    const sma200 = calculateSMA(closes, Math.min(200, closes.length))
    const ema12 = calculateEMA(closes, 12)
    const ema26 = calculateEMA(closes, 26)
    const rsi = calculateRSI(closes, 14)
    const bollinger = calculateBollingerBands(closes, 20, 2)
    const returns = calculateReturns(closes)
    const cumulativeReturns = calculateCumulativeReturns(closes)
    
    // MACD
    const macd = ema12.map((e12, i) => {
      const e26 = ema26[i]
      if (e12 === undefined || e26 === undefined) return undefined
      return e12 - e26
    })
    const macdValues = macd.filter((v): v is number => v !== undefined)
    const signalRaw = calculateEMA(macdValues, 9)
    let signalIndex = 0
    const signal = macd.map(m => {
      if (m === undefined) return undefined
      return signalRaw[signalIndex++]
    })
    const histogram = macd.map((m, i) => {
      if (m === undefined || signal[i] === undefined) return undefined
      return m - signal[i]!
    })

    // Volume analysis
    const avgVolume = volumes.reduce((a, b) => a + b, 0) / volumes.length
    const volumeChange = ((volumes[volumes.length - 1] - avgVolume) / avgVolume) * 100

    // Metrics
    const totalReturn = cumulativeReturns[cumulativeReturns.length - 1] || 0
    const volatility = calculateVolatility(returns)
    const sharpeRatio = calculateSharpeRatio(returns)
    const maxDrawdown = calculateMaxDrawdown(closes)

    // Price action
    const currentPrice = closes[closes.length - 1]
    const priceVsSMA20 = sma20[sma20.length - 1] ? ((currentPrice - sma20[sma20.length - 1]!) / sma20[sma20.length - 1]!) * 100 : 0
    const priceVsSMA50 = sma50[sma50.length - 1] ? ((currentPrice - sma50[sma50.length - 1]!) / sma50[sma50.length - 1]!) * 100 : 0

    // Build chart data
    const chartData = data.map((d, i) => ({
      date: d.date,
      displayDate: new Date(d.date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
      close: d.close,
      open: d.open,
      high: d.high,
      low: d.low,
      volume: d.volume,
      sma20: sma20[i],
      sma50: sma50[i],
      sma200: sma200[i],
      rsi: rsi[i],
      macd: macd[i],
      signal: signal[i],
      histogram: histogram[i],
      upperBand: bollinger.upper[i],
      middleBand: bollinger.middle[i],
      lowerBand: bollinger.lower[i],
      returns: returns[i],
      cumulativeReturns: cumulativeReturns[i],
    }))

    return {
      chartData,
      metrics: {
        totalReturn,
        volatility,
        sharpeRatio,
        maxDrawdown,
        volumeChange,
        avgVolume,
        priceVsSMA20,
        priceVsSMA50,
        currentRSI: rsi[rsi.length - 1] || 0,
        currentMACD: macd[macd.length - 1] || 0,
      }
    }
  }, [data])

  if (!analytics) {
    return (
      <Card className="bg-card/50 border-border/50">
        <CardContent className="p-8 text-center text-muted-foreground">
          Select a stock to view advanced analytics
        </CardContent>
      </Card>
    )
  }

  const { chartData, metrics } = analytics

  const MetricCard = ({ 
    title, value, subtitle, icon: Icon, trend, color = 'primary' 
  }: { 
    title: string
    value: string
    subtitle?: string
    icon: React.ElementType
    trend?: 'up' | 'down' | 'neutral'
    color?: 'primary' | 'destructive' | 'accent' | 'secondary'
  }) => (
    <div 
      className={cn(
        "p-4 rounded-lg border border-border/50 bg-card/30 transition-all duration-300 cursor-pointer",
        "hover:bg-card/60 hover:border-primary/30 hover:shadow-lg hover:shadow-primary/5",
        hoveredMetric === title && "bg-card/60 border-primary/30 shadow-lg shadow-primary/5"
      )}
      onMouseEnter={() => setHoveredMetric(title)}
      onMouseLeave={() => setHoveredMetric(null)}
    >
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs text-muted-foreground uppercase tracking-wide">{title}</span>
        <Icon className={cn("h-4 w-4", color === 'primary' && "text-primary", color === 'destructive' && "text-destructive", color === 'accent' && "text-accent", color === 'secondary' && "text-secondary-foreground")} />
      </div>
      <div className="flex items-baseline gap-2">
        <span className={cn(
          "text-2xl font-bold transition-colors duration-300",
          trend === 'up' && "text-primary",
          trend === 'down' && "text-destructive",
          !trend && "text-foreground"
        )}>{value}</span>
        {trend && (
          <span className={cn("flex items-center text-xs", trend === 'up' ? "text-primary" : "text-destructive")}>
            {trend === 'up' ? <TrendingUp className="h-3 w-3" /> : <TrendingDown className="h-3 w-3" />}
          </span>
        )}
      </div>
      {subtitle && <p className="text-xs text-muted-foreground mt-1">{subtitle}</p>}
    </div>
  )

  const chartConfig = {
    close: { label: 'Price', color: COLORS.primary },
    sma20: { label: 'SMA 20', color: COLORS.secondary },
    sma50: { label: 'SMA 50', color: COLORS.accent },
    rsi: { label: 'RSI', color: COLORS.purple },
    macd: { label: 'MACD', color: COLORS.cyan },
    signal: { label: 'Signal', color: COLORS.pink },
  }

  return (
    <div className="space-y-4">
      {/* Top Metrics Row */}
      <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3">
        <MetricCard
          title="Total Return"
          value={`${metrics.totalReturn >= 0 ? '+' : ''}${metrics.totalReturn.toFixed(2)}%`}
          icon={TrendingUp}
          trend={metrics.totalReturn >= 0 ? 'up' : 'down'}
          subtitle="Period performance"
        />
        <MetricCard
          title="Volatility"
          value={`${metrics.volatility.toFixed(1)}%`}
          icon={Activity}
          color="accent"
          subtitle="Annualized"
        />
        <MetricCard
          title="Sharpe Ratio"
          value={metrics.sharpeRatio.toFixed(2)}
          icon={Target}
          trend={metrics.sharpeRatio > 1 ? 'up' : metrics.sharpeRatio < 0 ? 'down' : 'neutral'}
          subtitle="Risk-adjusted"
        />
        <MetricCard
          title="Max Drawdown"
          value={`-${metrics.maxDrawdown.toFixed(2)}%`}
          icon={AlertTriangle}
          color="destructive"
          subtitle="Peak to trough"
        />
        <MetricCard
          title="RSI (14)"
          value={metrics.currentRSI.toFixed(1)}
          icon={Gauge}
          trend={metrics.currentRSI < 30 ? 'up' : metrics.currentRSI > 70 ? 'down' : 'neutral'}
          subtitle={metrics.currentRSI < 30 ? 'Oversold' : metrics.currentRSI > 70 ? 'Overbought' : 'Neutral'}
        />
        <MetricCard
          title="vs SMA 20"
          value={`${metrics.priceVsSMA20 >= 0 ? '+' : ''}${metrics.priceVsSMA20.toFixed(2)}%`}
          icon={BarChart3}
          trend={metrics.priceVsSMA20 >= 0 ? 'up' : 'down'}
          subtitle={metrics.priceVsSMA20 >= 0 ? 'Above' : 'Below'}
        />
      </div>

      {/* Main Charts Grid - Dense Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Price + Moving Averages + Bollinger */}
        <Card className="bg-card/50 border-border/50">
          <CardHeader className="pb-2">
            <div className="flex items-center justify-between">
              <CardTitle className="text-sm font-medium text-foreground">{symbol} Price Action</CardTitle>
              <div className="flex gap-2">
                <Badge variant="outline" className="text-xs" style={{ borderColor: COLORS.primary, color: COLORS.primary }}>Price</Badge>
                <Badge variant="outline" className="text-xs" style={{ borderColor: COLORS.secondary, color: COLORS.secondary }}>SMA20</Badge>
                <Badge variant="outline" className="text-xs" style={{ borderColor: COLORS.accent, color: COLORS.accent }}>SMA50</Badge>
              </div>
            </div>
          </CardHeader>
          <CardContent className="pt-0">
            <ChartContainer config={chartConfig} className="h-[200px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={chartData} margin={{ top: 5, right: 5, left: 0, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.3} />
                  <XAxis dataKey="displayDate" tick={{ fontSize: 9, fill: COLORS.muted }} axisLine={false} tickLine={false} interval="preserveStartEnd" />
                  <YAxis domain={['auto', 'auto']} tick={{ fontSize: 9, fill: COLORS.muted }} axisLine={false} tickLine={false} tickFormatter={(v) => `$${v}`} width={50} />
                  <Tooltip content={({ active, payload, label }) => {
                    if (!active || !payload?.length) return null
                    return (
                      <div className="rounded-lg border border-border/50 bg-background/95 backdrop-blur px-3 py-2 shadow-xl text-xs">
                        <p className="text-muted-foreground mb-1 font-medium">{label}</p>
                        {payload.map((p, i) => (
                          <p key={i} style={{ color: p.color }} className="flex justify-between gap-4">
                            <span>{p.name}:</span>
                            <span className="font-mono">${Number(p.value).toFixed(2)}</span>
                          </p>
                        ))}
                      </div>
                    )
                  }} />
                  <Area type="monotone" dataKey="upperBand" stroke="none" fill={COLORS.muted} fillOpacity={0.1} name="Upper Band" />
                  <Area type="monotone" dataKey="lowerBand" stroke="none" fill="transparent" name="Lower Band" />
                  <Line type="monotone" dataKey="upperBand" stroke={COLORS.muted} strokeWidth={1} dot={false} strokeDasharray="2 2" name="BB Upper" />
                  <Line type="monotone" dataKey="lowerBand" stroke={COLORS.muted} strokeWidth={1} dot={false} strokeDasharray="2 2" name="BB Lower" />
                  <Line type="monotone" dataKey="sma50" stroke={COLORS.accent} strokeWidth={1.5} dot={false} name="SMA 50" />
                  <Line type="monotone" dataKey="sma20" stroke={COLORS.secondary} strokeWidth={1.5} dot={false} name="SMA 20" />
                  <Line type="monotone" dataKey="close" stroke={COLORS.primary} strokeWidth={2} dot={false} name="Price" />
                </ComposedChart>
              </ResponsiveContainer>
            </ChartContainer>
          </CardContent>
        </Card>

        {/* Volume Chart */}
        <Card className="bg-card/50 border-border/50">
          <CardHeader className="pb-2">
            <div className="flex items-center justify-between">
              <CardTitle className="text-sm font-medium text-foreground">Volume Analysis</CardTitle>
              <Badge variant="outline" className="text-xs" style={{ borderColor: metrics.volumeChange >= 0 ? COLORS.primary : COLORS.destructive, color: metrics.volumeChange >= 0 ? COLORS.primary : COLORS.destructive }}>
                {metrics.volumeChange >= 0 ? '+' : ''}{metrics.volumeChange.toFixed(1)}% vs avg
              </Badge>
            </div>
          </CardHeader>
          <CardContent className="pt-0">
            <ChartContainer config={chartConfig} className="h-[200px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartData} margin={{ top: 5, right: 5, left: 0, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.3} />
                  <XAxis dataKey="displayDate" tick={{ fontSize: 9, fill: COLORS.muted }} axisLine={false} tickLine={false} interval="preserveStartEnd" />
                  <YAxis tick={{ fontSize: 9, fill: COLORS.muted }} axisLine={false} tickLine={false} tickFormatter={(v) => `${(v / 1000000).toFixed(0)}M`} width={40} />
                  <Tooltip content={({ active, payload, label }) => {
                    if (!active || !payload?.length) return null
                    return (
                      <div className="rounded-lg border border-border/50 bg-background/95 backdrop-blur px-3 py-2 shadow-xl text-xs">
                        <p className="text-muted-foreground mb-1 font-medium">{label}</p>
                        <p style={{ color: COLORS.secondary }}>Volume: {(Number(payload[0]?.value) / 1000000).toFixed(2)}M</p>
                      </div>
                    )
                  }} />
                  <ReferenceLine y={metrics.avgVolume} stroke={COLORS.accent} strokeDasharray="3 3" />
                  <Bar 
                    dataKey="volume" 
                    fill={COLORS.secondary}
                    opacity={0.8}
                    radius={[2, 2, 0, 0]}
                  />
                </BarChart>
              </ResponsiveContainer>
            </ChartContainer>
          </CardContent>
        </Card>

        {/* RSI Chart */}
        <Card className="bg-card/50 border-border/50">
          <CardHeader className="pb-2">
            <div className="flex items-center justify-between">
              <CardTitle className="text-sm font-medium text-foreground">RSI (14)</CardTitle>
              <Badge variant={metrics.currentRSI < 30 ? 'default' : metrics.currentRSI > 70 ? 'destructive' : 'secondary'} className="text-xs">
                {metrics.currentRSI.toFixed(1)} - {metrics.currentRSI < 30 ? 'Oversold' : metrics.currentRSI > 70 ? 'Overbought' : 'Neutral'}
              </Badge>
            </div>
          </CardHeader>
          <CardContent className="pt-0">
            <ChartContainer config={chartConfig} className="h-[150px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={chartData} margin={{ top: 5, right: 5, left: 0, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.3} />
                  <XAxis dataKey="displayDate" tick={{ fontSize: 9, fill: COLORS.muted }} axisLine={false} tickLine={false} interval="preserveStartEnd" />
                  <YAxis domain={[0, 100]} tick={{ fontSize: 9, fill: COLORS.muted }} axisLine={false} tickLine={false} width={30} />
                  <Tooltip content={({ active, payload, label }) => {
                    if (!active || !payload?.length) return null
                    const rsiVal = Number(payload[0]?.value)
                    return (
                      <div className="rounded-lg border border-border/50 bg-background/95 backdrop-blur px-3 py-2 shadow-xl text-xs">
                        <p className="text-muted-foreground mb-1 font-medium">{label}</p>
                        <p style={{ color: rsiVal < 30 ? COLORS.primary : rsiVal > 70 ? COLORS.destructive : COLORS.purple }}>
                          RSI: {rsiVal.toFixed(2)}
                        </p>
                      </div>
                    )
                  }} />
                  <ReferenceLine y={70} stroke={COLORS.destructive} strokeDasharray="3 3" strokeOpacity={0.7} />
                  <ReferenceLine y={30} stroke={COLORS.primary} strokeDasharray="3 3" strokeOpacity={0.7} />
                  <ReferenceLine y={50} stroke={COLORS.muted} strokeDasharray="3 3" strokeOpacity={0.5} />
                  <defs>
                    <linearGradient id="rsiGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor={COLORS.purple} stopOpacity={0.3} />
                      <stop offset="95%" stopColor={COLORS.purple} stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <Area type="monotone" dataKey="rsi" stroke={COLORS.purple} strokeWidth={2} fill="url(#rsiGradient)" dot={false} name="RSI" />
                </AreaChart>
              </ResponsiveContainer>
            </ChartContainer>
          </CardContent>
        </Card>

        {/* MACD Chart */}
        <Card className="bg-card/50 border-border/50">
          <CardHeader className="pb-2">
            <div className="flex items-center justify-between">
              <CardTitle className="text-sm font-medium text-foreground">MACD</CardTitle>
              <div className="flex gap-1">
                <Badge variant="outline" className="text-xs" style={{ borderColor: COLORS.cyan, color: COLORS.cyan }}>MACD</Badge>
                <Badge variant="outline" className="text-xs" style={{ borderColor: COLORS.pink, color: COLORS.pink }}>Signal</Badge>
              </div>
            </div>
          </CardHeader>
          <CardContent className="pt-0">
            <ChartContainer config={chartConfig} className="h-[150px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={chartData} margin={{ top: 5, right: 5, left: 0, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.3} />
                  <XAxis dataKey="displayDate" tick={{ fontSize: 9, fill: COLORS.muted }} axisLine={false} tickLine={false} interval="preserveStartEnd" />
                  <YAxis tick={{ fontSize: 9, fill: COLORS.muted }} axisLine={false} tickLine={false} width={40} />
                  <Tooltip content={({ active, payload, label }) => {
                    if (!active || !payload?.length) return null
                    return (
                      <div className="rounded-lg border border-border/50 bg-background/95 backdrop-blur px-3 py-2 shadow-xl text-xs">
                        <p className="text-muted-foreground mb-1 font-medium">{label}</p>
                        {payload.map((p, i) => (
                          <p key={i} style={{ color: p.color }}>{p.name}: {Number(p.value).toFixed(4)}</p>
                        ))}
                      </div>
                    )
                  }} />
                  <ReferenceLine y={0} stroke={COLORS.muted} />
                  <Bar dataKey="histogram" fill={COLORS.secondary} opacity={0.4} radius={[1, 1, 0, 0]} name="Histogram" />
                  <Line type="monotone" dataKey="macd" stroke={COLORS.cyan} strokeWidth={2} dot={false} name="MACD" />
                  <Line type="monotone" dataKey="signal" stroke={COLORS.pink} strokeWidth={1.5} dot={false} name="Signal" />
                </ComposedChart>
              </ResponsiveContainer>
            </ChartContainer>
          </CardContent>
        </Card>
      </div>

      {/* Cumulative Returns */}
      <Card className="bg-card/50 border-border/50">
        <CardHeader className="pb-2">
          <div className="flex items-center justify-between">
            <CardTitle className="text-sm font-medium text-foreground">Cumulative Returns</CardTitle>
            <Badge variant={metrics.totalReturn >= 0 ? 'default' : 'destructive'} className="text-xs">
              {metrics.totalReturn >= 0 ? '+' : ''}{metrics.totalReturn.toFixed(2)}%
            </Badge>
          </div>
        </CardHeader>
        <CardContent className="pt-0">
          <ChartContainer config={chartConfig} className="h-[150px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={chartData} margin={{ top: 5, right: 5, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.3} />
                <XAxis dataKey="displayDate" tick={{ fontSize: 9, fill: COLORS.muted }} axisLine={false} tickLine={false} interval="preserveStartEnd" />
                <YAxis tick={{ fontSize: 9, fill: COLORS.muted }} axisLine={false} tickLine={false} tickFormatter={(v) => `${v.toFixed(0)}%`} width={40} />
                <Tooltip content={({ active, payload, label }) => {
                  if (!active || !payload?.length) return null
                  const val = Number(payload[0]?.value)
                  return (
                    <div className="rounded-lg border border-border/50 bg-background/95 backdrop-blur px-3 py-2 shadow-xl text-xs">
                      <p className="text-muted-foreground mb-1 font-medium">{label}</p>
                      <p style={{ color: val >= 0 ? COLORS.primary : COLORS.destructive }}>Return: {val >= 0 ? '+' : ''}{val.toFixed(2)}%</p>
                    </div>
                  )
                }} />
                <ReferenceLine y={0} stroke={COLORS.muted} />
                <defs>
                  <linearGradient id="returnGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor={metrics.totalReturn >= 0 ? COLORS.primary : COLORS.destructive} stopOpacity={0.3} />
                    <stop offset="95%" stopColor={metrics.totalReturn >= 0 ? COLORS.primary : COLORS.destructive} stopOpacity={0} />
                  </linearGradient>
                </defs>
                <Area 
                  type="monotone" 
                  dataKey="cumulativeReturns" 
                  stroke={metrics.totalReturn >= 0 ? COLORS.primary : COLORS.destructive} 
                  strokeWidth={2} 
                  fill="url(#returnGradient)" 
                  dot={false} 
                  name="Cumulative Return" 
                />
              </AreaChart>
            </ResponsiveContainer>
          </ChartContainer>
        </CardContent>
      </Card>

      {/* Daily Returns Distribution */}
      <Card className="bg-card/50 border-border/50">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-medium text-foreground">Daily Returns</CardTitle>
        </CardHeader>
        <CardContent className="pt-0">
          <ChartContainer config={chartConfig} className="h-[120px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData.slice(-60)} margin={{ top: 5, right: 5, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.3} />
                <XAxis dataKey="displayDate" tick={{ fontSize: 8, fill: COLORS.muted }} axisLine={false} tickLine={false} interval={9} />
                <YAxis tick={{ fontSize: 9, fill: COLORS.muted }} axisLine={false} tickLine={false} tickFormatter={(v) => `${v.toFixed(1)}%`} width={35} />
                <Tooltip content={({ active, payload, label }) => {
                  if (!active || !payload?.length) return null
                  const val = Number(payload[0]?.value)
                  return (
                    <div className="rounded-lg border border-border/50 bg-background/95 backdrop-blur px-3 py-2 shadow-xl text-xs">
                      <p className="text-muted-foreground mb-1 font-medium">{label}</p>
                      <p style={{ color: val >= 0 ? COLORS.primary : COLORS.destructive }}>
                        Return: {val >= 0 ? '+' : ''}{val.toFixed(2)}%
                      </p>
                    </div>
                  )
                }} />
                <ReferenceLine y={0} stroke={COLORS.muted} />
                <Bar 
                  dataKey="returns" 
                  radius={[1, 1, 0, 0]}
                  name="Daily Return"
                >
                  {chartData.slice(-60).map((entry, index) => (
                    <rect 
                      key={`bar-${index}`}
                      fill={(entry.returns || 0) >= 0 ? COLORS.primary : COLORS.destructive}
                      opacity={0.8}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </ChartContainer>
        </CardContent>
      </Card>
    </div>
  )
}
