'use client'

import { useState, useEffect, useMemo } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { ChartContainer, ChartTooltip } from '@/components/ui/chart'
import { LineChart, Line, XAxis, YAxis, ResponsiveContainer, ReferenceLine, Area, ComposedChart } from 'recharts'
import type { TimeSeriesDataPoint } from '@/lib/types'
import { TrendingUp, TrendingDown, Minus } from 'lucide-react'

interface TechnicalIndicatorsProps {
  symbol: string
  data: TimeSeriesDataPoint[]
}

interface IndicatorData {
  date: string
  close: number
  sma20?: number
  sma50?: number
  ema12?: number
  ema26?: number
  rsi?: number
  macd?: number
  signal?: number
  histogram?: number
  upperBand?: number
  lowerBand?: number
  middleBand?: number
}

// Calculate SMA (Simple Moving Average)
function calculateSMA(data: number[], period: number): (number | undefined)[] {
  const result: (number | undefined)[] = []
  for (let i = 0; i < data.length; i++) {
    if (i < period - 1) {
      result.push(undefined)
    } else {
      const sum = data.slice(i - period + 1, i + 1).reduce((a, b) => a + b, 0)
      result.push(sum / period)
    }
  }
  return result
}

// Calculate EMA (Exponential Moving Average)
function calculateEMA(data: number[], period: number): (number | undefined)[] {
  const result: (number | undefined)[] = []
  const multiplier = 2 / (period + 1)
  
  for (let i = 0; i < data.length; i++) {
    if (i < period - 1) {
      result.push(undefined)
    } else if (i === period - 1) {
      const sum = data.slice(0, period).reduce((a, b) => a + b, 0)
      result.push(sum / period)
    } else {
      const prev = result[i - 1]
      if (prev !== undefined) {
        result.push((data[i] - prev) * multiplier + prev)
      }
    }
  }
  return result
}

// Calculate RSI (Relative Strength Index)
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
    if (i < period) {
      result.push(undefined)
    } else {
      const avgGain = gains.slice(i - period, i).reduce((a, b) => a + b, 0) / period
      const avgLoss = losses.slice(i - period, i).reduce((a, b) => a + b, 0) / period
      
      if (avgLoss === 0) {
        result.push(100)
      } else {
        const rs = avgGain / avgLoss
        result.push(100 - (100 / (1 + rs)))
      }
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

export function TechnicalIndicators({ symbol, data }: TechnicalIndicatorsProps) {
  const [activeIndicator, setActiveIndicator] = useState<'overview' | 'rsi' | 'macd' | 'bollinger'>('overview')

  const indicatorData = useMemo<IndicatorData[]>(() => {
    if (!data || data.length === 0) return []

    const closes = data.map(d => d.close)
    const sma20 = calculateSMA(closes, 20)
    const sma50 = calculateSMA(closes, 50)
    const ema12 = calculateEMA(closes, 12)
    const ema26 = calculateEMA(closes, 26)
    const rsi = calculateRSI(closes, 14)
    const bollinger = calculateBollingerBands(closes, 20, 2)

    // MACD = EMA12 - EMA26
    const macd = ema12.map((e12, i) => {
      const e26 = ema26[i]
      if (e12 === undefined || e26 === undefined) return undefined
      return e12 - e26
    })

    // Signal line = 9-period EMA of MACD
    const macdValues = macd.filter((v): v is number => v !== undefined)
    const signalRaw = calculateEMA(macdValues, 9)
    let signalIndex = 0
    const signal = macd.map(m => {
      if (m === undefined) return undefined
      return signalRaw[signalIndex++]
    })

    return data.map((d, i) => ({
      date: d.date,
      close: d.close,
      sma20: sma20[i],
      sma50: sma50[i],
      ema12: ema12[i],
      ema26: ema26[i],
      rsi: rsi[i],
      macd: macd[i],
      signal: signal[i],
      histogram: macd[i] !== undefined && signal[i] !== undefined ? macd[i]! - signal[i]! : undefined,
      upperBand: bollinger.upper[i],
      lowerBand: bollinger.lower[i],
      middleBand: bollinger.middle[i],
    }))
  }, [data])

  const latestData = indicatorData[indicatorData.length - 1]
  const prevData = indicatorData[indicatorData.length - 2]

  // Determine signals
  const signals = useMemo(() => {
    if (!latestData || !prevData) return { rsi: 'neutral', macd: 'neutral', trend: 'neutral' }

    // RSI Signal
    let rsiSignal: 'bullish' | 'bearish' | 'neutral' = 'neutral'
    if (latestData.rsi !== undefined) {
      if (latestData.rsi < 30) rsiSignal = 'bullish' // Oversold
      else if (latestData.rsi > 70) rsiSignal = 'bearish' // Overbought
    }

    // MACD Signal
    let macdSignal: 'bullish' | 'bearish' | 'neutral' = 'neutral'
    if (latestData.macd !== undefined && latestData.signal !== undefined && prevData.macd !== undefined && prevData.signal !== undefined) {
      const prevDiff = prevData.macd - prevData.signal
      const currDiff = latestData.macd - latestData.signal
      if (prevDiff < 0 && currDiff > 0) macdSignal = 'bullish' // Bullish crossover
      else if (prevDiff > 0 && currDiff < 0) macdSignal = 'bearish' // Bearish crossover
      else if (currDiff > 0) macdSignal = 'bullish'
      else if (currDiff < 0) macdSignal = 'bearish'
    }

    // Trend Signal (SMA crossover)
    let trendSignal: 'bullish' | 'bearish' | 'neutral' = 'neutral'
    if (latestData.sma20 !== undefined && latestData.sma50 !== undefined) {
      if (latestData.sma20 > latestData.sma50) trendSignal = 'bullish'
      else if (latestData.sma20 < latestData.sma50) trendSignal = 'bearish'
    }

    return { rsi: rsiSignal, macd: macdSignal, trend: trendSignal }
  }, [latestData, prevData])

  const chartConfig = {
    close: { label: 'Price', color: 'hsl(var(--foreground))' },
    sma20: { label: 'SMA 20', color: 'hsl(var(--primary))' },
    sma50: { label: 'SMA 50', color: 'hsl(var(--accent))' },
    rsi: { label: 'RSI', color: 'hsl(var(--primary))' },
    macd: { label: 'MACD', color: 'hsl(var(--primary))' },
    signal: { label: 'Signal', color: 'hsl(var(--destructive))' },
    upperBand: { label: 'Upper Band', color: 'hsl(var(--muted-foreground))' },
    lowerBand: { label: 'Lower Band', color: 'hsl(var(--muted-foreground))' },
  }

  if (!data || data.length === 0) {
    return (
      <Card className="bg-card/50 border-border/50">
        <CardContent className="p-8 text-center text-muted-foreground">
          Select a stock to view technical indicators
        </CardContent>
      </Card>
    )
  }

  const SignalBadge = ({ signal, label }: { signal: 'bullish' | 'bearish' | 'neutral', label: string }) => (
    <div className="flex items-center gap-2">
      <span className="text-muted-foreground text-sm">{label}:</span>
      <Badge variant={signal === 'bullish' ? 'default' : signal === 'bearish' ? 'destructive' : 'secondary'} className="gap-1">
        {signal === 'bullish' && <TrendingUp className="h-3 w-3" />}
        {signal === 'bearish' && <TrendingDown className="h-3 w-3" />}
        {signal === 'neutral' && <Minus className="h-3 w-3" />}
        {signal.charAt(0).toUpperCase() + signal.slice(1)}
      </Badge>
    </div>
  )

  return (
    <Card className="bg-card/50 border-border/50">
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="text-foreground">Technical Analysis - {symbol}</CardTitle>
          <div className="flex items-center gap-4">
            <SignalBadge signal={signals.trend} label="Trend" />
            <SignalBadge signal={signals.rsi} label="RSI" />
            <SignalBadge signal={signals.macd} label="MACD" />
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <Tabs value={activeIndicator} onValueChange={(v) => setActiveIndicator(v as typeof activeIndicator)}>
          <TabsList className="mb-4">
            <TabsTrigger value="overview">Price + MAs</TabsTrigger>
            <TabsTrigger value="rsi">RSI</TabsTrigger>
            <TabsTrigger value="macd">MACD</TabsTrigger>
            <TabsTrigger value="bollinger">Bollinger</TabsTrigger>
          </TabsList>

          <ChartContainer config={chartConfig} className="h-[300px] w-full">
            {activeIndicator === 'overview' && (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={indicatorData}>
                  <XAxis dataKey="date" tick={{ fontSize: 10 }} tickFormatter={(v) => v.slice(5)} stroke="hsl(var(--muted-foreground))" />
                  <YAxis domain={['auto', 'auto']} tick={{ fontSize: 10 }} stroke="hsl(var(--muted-foreground))" />
                  <ChartTooltip content={({ active, payload, label }) => {
                    if (!active || !payload?.length) return null
                    return (
                      <div className="rounded-lg border border-border/50 bg-background px-3 py-2 shadow-xl text-sm">
                        <p className="text-muted-foreground mb-1">{label}</p>
                        {payload.map((p, i) => (
                          <p key={i} style={{ color: p.color }}>{p.name}: ${Number(p.value).toFixed(2)}</p>
                        ))}
                      </div>
                    )
                  }} />
                  <Line type="monotone" dataKey="close" stroke="hsl(var(--foreground))" strokeWidth={2} dot={false} name="Price" />
                  <Line type="monotone" dataKey="sma20" stroke="hsl(var(--primary))" strokeWidth={1.5} dot={false} name="SMA 20" strokeDasharray="5 5" />
                  <Line type="monotone" dataKey="sma50" stroke="hsl(var(--accent))" strokeWidth={1.5} dot={false} name="SMA 50" strokeDasharray="5 5" />
                </LineChart>
              </ResponsiveContainer>
            )}

            {activeIndicator === 'rsi' && (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={indicatorData}>
                  <XAxis dataKey="date" tick={{ fontSize: 10 }} tickFormatter={(v) => v.slice(5)} stroke="hsl(var(--muted-foreground))" />
                  <YAxis domain={[0, 100]} tick={{ fontSize: 10 }} stroke="hsl(var(--muted-foreground))" />
                  <ChartTooltip content={({ active, payload, label }) => {
                    if (!active || !payload?.length) return null
                    return (
                      <div className="rounded-lg border border-border/50 bg-background px-3 py-2 shadow-xl text-sm">
                        <p className="text-muted-foreground mb-1">{label}</p>
                        <p className="text-primary">RSI: {Number(payload[0]?.value).toFixed(2)}</p>
                      </div>
                    )
                  }} />
                  <ReferenceLine y={70} stroke="hsl(var(--destructive))" strokeDasharray="3 3" label={{ value: 'Overbought', fill: 'hsl(var(--destructive))', fontSize: 10 }} />
                  <ReferenceLine y={30} stroke="hsl(var(--primary))" strokeDasharray="3 3" label={{ value: 'Oversold', fill: 'hsl(var(--primary))', fontSize: 10 }} />
                  <ReferenceLine y={50} stroke="hsl(var(--muted-foreground))" strokeDasharray="3 3" />
                  <Line type="monotone" dataKey="rsi" stroke="hsl(var(--primary))" strokeWidth={2} dot={false} name="RSI" />
                </LineChart>
              </ResponsiveContainer>
            )}

            {activeIndicator === 'macd' && (
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={indicatorData}>
                  <XAxis dataKey="date" tick={{ fontSize: 10 }} tickFormatter={(v) => v.slice(5)} stroke="hsl(var(--muted-foreground))" />
                  <YAxis tick={{ fontSize: 10 }} stroke="hsl(var(--muted-foreground))" />
                  <ChartTooltip content={({ active, payload, label }) => {
                    if (!active || !payload?.length) return null
                    return (
                      <div className="rounded-lg border border-border/50 bg-background px-3 py-2 shadow-xl text-sm">
                        <p className="text-muted-foreground mb-1">{label}</p>
                        {payload.map((p, i) => (
                          <p key={i} style={{ color: p.color }}>{p.name}: {Number(p.value).toFixed(4)}</p>
                        ))}
                      </div>
                    )
                  }} />
                  <ReferenceLine y={0} stroke="hsl(var(--muted-foreground))" />
                  <Line type="monotone" dataKey="macd" stroke="hsl(var(--primary))" strokeWidth={2} dot={false} name="MACD" />
                  <Line type="monotone" dataKey="signal" stroke="hsl(var(--destructive))" strokeWidth={1.5} dot={false} name="Signal" />
                </ComposedChart>
              </ResponsiveContainer>
            )}

            {activeIndicator === 'bollinger' && (
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={indicatorData}>
                  <XAxis dataKey="date" tick={{ fontSize: 10 }} tickFormatter={(v) => v.slice(5)} stroke="hsl(var(--muted-foreground))" />
                  <YAxis domain={['auto', 'auto']} tick={{ fontSize: 10 }} stroke="hsl(var(--muted-foreground))" />
                  <ChartTooltip content={({ active, payload, label }) => {
                    if (!active || !payload?.length) return null
                    return (
                      <div className="rounded-lg border border-border/50 bg-background px-3 py-2 shadow-xl text-sm">
                        <p className="text-muted-foreground mb-1">{label}</p>
                        {payload.map((p, i) => (
                          <p key={i} style={{ color: p.color }}>{p.name}: ${Number(p.value).toFixed(2)}</p>
                        ))}
                      </div>
                    )
                  }} />
                  <Area type="monotone" dataKey="upperBand" stroke="none" fill="hsl(var(--muted))" fillOpacity={0.3} name="Upper Band" />
                  <Area type="monotone" dataKey="lowerBand" stroke="none" fill="hsl(var(--background))" name="Lower Band" />
                  <Line type="monotone" dataKey="upperBand" stroke="hsl(var(--muted-foreground))" strokeWidth={1} dot={false} strokeDasharray="3 3" name="Upper" />
                  <Line type="monotone" dataKey="middleBand" stroke="hsl(var(--accent))" strokeWidth={1} dot={false} name="Middle" />
                  <Line type="monotone" dataKey="lowerBand" stroke="hsl(var(--muted-foreground))" strokeWidth={1} dot={false} strokeDasharray="3 3" name="Lower" />
                  <Line type="monotone" dataKey="close" stroke="hsl(var(--foreground))" strokeWidth={2} dot={false} name="Price" />
                </ComposedChart>
              </ResponsiveContainer>
            )}
          </ChartContainer>
        </Tabs>

        {/* Key Metrics */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-4 pt-4 border-t border-border/50">
          <div>
            <p className="text-muted-foreground text-xs">RSI (14)</p>
            <p className={`text-lg font-bold ${latestData?.rsi && latestData.rsi < 30 ? 'text-primary' : latestData?.rsi && latestData.rsi > 70 ? 'text-destructive' : 'text-foreground'}`}>
              {latestData?.rsi?.toFixed(2) || 'N/A'}
            </p>
          </div>
          <div>
            <p className="text-muted-foreground text-xs">MACD</p>
            <p className={`text-lg font-bold ${latestData?.macd && latestData.macd > 0 ? 'text-primary' : 'text-destructive'}`}>
              {latestData?.macd?.toFixed(4) || 'N/A'}
            </p>
          </div>
          <div>
            <p className="text-muted-foreground text-xs">SMA 20</p>
            <p className="text-lg font-bold text-foreground">${latestData?.sma20?.toFixed(2) || 'N/A'}</p>
          </div>
          <div>
            <p className="text-muted-foreground text-xs">SMA 50</p>
            <p className="text-lg font-bold text-foreground">${latestData?.sma50?.toFixed(2) || 'N/A'}</p>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
