'use client'

import { useState, useMemo, useEffect } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Slider } from '@/components/ui/slider'
import { Badge } from '@/components/ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { cn } from '@/lib/utils'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend, AreaChart, Area, ReferenceLine } from 'recharts'
import { Calculator, TrendingUp, Clock, Activity, Zap, Target } from 'lucide-react'

type OptionType = 'CALL' | 'PUT'

interface Greeks {
  delta: number
  gamma: number
  theta: number
  vega: number
  rho: number
}

interface PricingResult {
  price: number
  intrinsicValue: number
  timeValue: number
  greeks: Greeks
  calculationTimeNs: number
}

interface OptionsPricerProps {
  initialSpot?: number
  symbol?: string
}

export function OptionsPricer({ initialSpot = 150, symbol = 'AAPL' }: OptionsPricerProps) {
  const [spotPrice, setSpotPrice] = useState(initialSpot)
  const [strikePrice, setStrikePrice] = useState(initialSpot)
  const [timeToExpiry, setTimeToExpiry] = useState(30)
  const [riskFreeRate, setRiskFreeRate] = useState(5)
  const [volatility, setVolatility] = useState(25)
  const [optionType, setOptionType] = useState<OptionType>('CALL')
  const [marketPrice, setMarketPrice] = useState('')
  const [result, setResult] = useState<PricingResult | null>(null)
  const [ivResult, setIvResult] = useState<{ iv: number; converged: boolean } | null>(null)
  const [optionChain, setOptionChain] = useState<any[]>([])

  useEffect(() => {
    setSpotPrice(initialSpot)
    setStrikePrice(initialSpot)
    setMarketPrice('')
  }, [initialSpot, symbol])

  // Calculate option price via API
  useEffect(() => {
    const calculate = async () => {
      try {
        const response = await fetch('/api/compute/options', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            action: 'price',
            S: spotPrice,
            K: strikePrice,
            T: timeToExpiry / 365,
            r: riskFreeRate / 100,
            sigma: volatility / 100,
            type: optionType
          })
        })
        const data = await response.json()
        setResult(data)
      } catch (error) {
        console.error('Pricing failed:', error)
      }
    }
    calculate()
  }, [spotPrice, strikePrice, timeToExpiry, riskFreeRate, volatility, optionType])

  // Calculate IV when market price provided
  useEffect(() => {
    const mp = parseFloat(marketPrice)
    if (isNaN(mp) || mp <= 0) {
      setIvResult(null)
      return
    }
    
    const calculate = async () => {
      try {
        const response = await fetch('/api/compute/options', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            action: 'implied_volatility',
            marketPrice: mp,
            S: spotPrice,
            K: strikePrice,
            T: timeToExpiry / 365,
            r: riskFreeRate / 100,
            type: optionType
          })
        })
        const data = await response.json()
        setIvResult(data)
      } catch (error) {
        console.error('IV calculation failed:', error)
      }
    }
    calculate()
  }, [marketPrice, spotPrice, strikePrice, timeToExpiry, riskFreeRate, optionType])

  // Generate option chain
  useEffect(() => {
    const generate = async () => {
      const strikes = []
      for (let i = -10; i <= 10; i++) {
        strikes.push(Math.round((spotPrice + i * 2.5) * 100) / 100)
      }
      
      try {
        const response = await fetch('/api/compute/options', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            action: 'option_chain',
            S: spotPrice,
            T: timeToExpiry / 365,
            r: riskFreeRate / 100,
            sigma: volatility / 100,
            strikes
          })
        })
        const data = await response.json()
        setOptionChain(data.chain || [])
      } catch (error) {
        console.error('Chain failed:', error)
      }
    }
    generate()
  }, [spotPrice, timeToExpiry, riskFreeRate, volatility])

  // Payoff diagram data
  const payoffData = useMemo(() => {
    const data = []
    for (let s = spotPrice * 0.7; s <= spotPrice * 1.3; s += spotPrice * 0.02) {
      const intrinsic = optionType === 'CALL' ? Math.max(0, s - strikePrice) : Math.max(0, strikePrice - s)
      const profit = intrinsic - (result?.price || 0)
      data.push({ spot: s, payoff: intrinsic, profit })
    }
    return data
  }, [spotPrice, strikePrice, optionType, result])

  const formatGreek = (value: number, precision: number = 4) => {
    if (Math.abs(value) < 0.0001) return '~0'
    return value.toFixed(precision)
  }

  return (
    <Card className="w-full">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <CardTitle className="text-lg font-semibold">Options Pricer</CardTitle>
            <Badge variant="outline" className="font-mono">{symbol}</Badge>
            <Badge variant="secondary" className="text-xs"><Zap className="w-3 h-3 mr-1" />Black-Scholes</Badge>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="pricer" className="w-full">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="pricer">Pricer</TabsTrigger>
            <TabsTrigger value="chain">Option Chain</TabsTrigger>
            <TabsTrigger value="payoff">Payoff</TabsTrigger>
          </TabsList>
          
          <TabsContent value="pricer" className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-4">
                <div>
                  <Label className="text-xs">Option Type</Label>
                  <div className="flex gap-2 mt-1">
                    <Button variant={optionType === 'CALL' ? 'default' : 'outline'} size="sm" className={cn(optionType === 'CALL' && 'bg-green-600')} onClick={() => setOptionType('CALL')}>Call</Button>
                    <Button variant={optionType === 'PUT' ? 'default' : 'outline'} size="sm" className={cn(optionType === 'PUT' && 'bg-red-600')} onClick={() => setOptionType('PUT')}>Put</Button>
                  </div>
                </div>
                
                <div>
                  <Label className="text-xs">Spot Price: ${spotPrice.toFixed(2)}</Label>
                  <Slider value={[spotPrice]} onValueChange={([v]) => setSpotPrice(v)} min={50} max={300} step={0.5} className="mt-2" />
                </div>
                
                <div>
                  <Label className="text-xs">Strike Price: ${strikePrice.toFixed(2)}</Label>
                  <Slider value={[strikePrice]} onValueChange={([v]) => setStrikePrice(v)} min={50} max={300} step={0.5} className="mt-2" />
                </div>
                
                <div>
                  <Label className="text-xs">Days to Expiry: {timeToExpiry}</Label>
                  <Slider value={[timeToExpiry]} onValueChange={([v]) => setTimeToExpiry(v)} min={1} max={365} step={1} className="mt-2" />
                </div>
                
                <div>
                  <Label className="text-xs">Volatility: {volatility}%</Label>
                  <Slider value={[volatility]} onValueChange={([v]) => setVolatility(v)} min={5} max={100} step={1} className="mt-2" />
                </div>
                
                <div>
                  <Label className="text-xs">Risk-Free Rate: {riskFreeRate}%</Label>
                  <Slider value={[riskFreeRate]} onValueChange={([v]) => setRiskFreeRate(v)} min={0} max={15} step={0.25} className="mt-2" />
                </div>
              </div>
              
              <div className="space-y-4">
                <div className="bg-muted/50 rounded-lg p-4">
                  <div className="text-xs text-muted-foreground mb-1">Option Price</div>
                  <div className="text-3xl font-mono font-bold">${result?.price.toFixed(2) || '0.00'}</div>
                  <div className="text-xs text-muted-foreground mt-1">
                    Intrinsic: ${result?.intrinsicValue.toFixed(2) || '0.00'} | Time: ${result?.timeValue.toFixed(2) || '0.00'}
                  </div>
                </div>
                
                <div className="grid grid-cols-2 gap-2">
                  <div className="bg-muted/50 rounded p-3">
                    <div className="text-xs text-muted-foreground">Delta (Δ)</div>
                    <div className="font-mono font-semibold">{formatGreek(result?.greeks.delta || 0)}</div>
                  </div>
                  <div className="bg-muted/50 rounded p-3">
                    <div className="text-xs text-muted-foreground">Gamma (Γ)</div>
                    <div className="font-mono font-semibold">{formatGreek(result?.greeks.gamma || 0)}</div>
                  </div>
                  <div className="bg-muted/50 rounded p-3">
                    <div className="text-xs text-muted-foreground">Theta (Θ)</div>
                    <div className="font-mono font-semibold">{formatGreek(result?.greeks.theta || 0)}</div>
                  </div>
                  <div className="bg-muted/50 rounded p-3">
                    <div className="text-xs text-muted-foreground">Vega (ν)</div>
                    <div className="font-mono font-semibold">{formatGreek(result?.greeks.vega || 0)}</div>
                  </div>
                </div>
                
                <div className="border-t pt-4">
                  <Label className="text-xs">Implied Volatility Solver</Label>
                  <div className="flex gap-2 mt-1">
                    <Input type="number" placeholder="Market Price" value={marketPrice} onChange={(e) => setMarketPrice(e.target.value)} className="h-8" step="0.01" />
                    {ivResult && (
                      <div className="bg-blue-500/10 rounded px-3 py-1 flex items-center">
                        <span className="text-sm font-mono">{(ivResult.iv * 100).toFixed(2)}%</span>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </div>
          </TabsContent>
          
          <TabsContent value="chain">
            <div className="max-h-80 overflow-auto">
              <table className="w-full text-sm">
                <thead className="sticky top-0 bg-background">
                  <tr className="border-b">
                    <th className="text-left p-2">Strike</th>
                    <th className="text-right p-2 text-green-600">Call</th>
                    <th className="text-right p-2 text-green-600">Δ</th>
                    <th className="text-right p-2 text-red-600">Put</th>
                    <th className="text-right p-2 text-red-600">Δ</th>
                  </tr>
                </thead>
                <tbody>
                  {optionChain.map((row, i) => (
                    <tr key={i} className={cn("border-b hover:bg-muted/50", row.strike === Math.round(spotPrice) && "bg-yellow-500/10")}>
                      <td className="p-2 font-mono">${row.strike.toFixed(2)}</td>
                      <td className="p-2 font-mono text-right text-green-600">${row.call?.price.toFixed(2)}</td>
                      <td className="p-2 font-mono text-right text-muted-foreground">{row.call?.greeks.delta.toFixed(2)}</td>
                      <td className="p-2 font-mono text-right text-red-600">${row.put?.price.toFixed(2)}</td>
                      <td className="p-2 font-mono text-right text-muted-foreground">{row.put?.greeks.delta.toFixed(2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </TabsContent>
          
          <TabsContent value="payoff">
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={payoffData}>
                  <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                  <XAxis dataKey="spot" tickFormatter={(v) => `$${v.toFixed(0)}`} className="text-xs" />
                  <YAxis tickFormatter={(v) => `$${v.toFixed(0)}`} className="text-xs" />
                  <Tooltip formatter={(v: number) => `$${v.toFixed(2)}`} />
                  <Legend />
                  <ReferenceLine x={spotPrice} stroke="#888" strokeDasharray="3 3" />
                  <ReferenceLine y={0} stroke="#888" />
                  <Area type="monotone" dataKey="profit" stroke={optionType === 'CALL' ? '#22c55e' : '#ef4444'} fill={optionType === 'CALL' ? '#22c55e20' : '#ef444420'} name="P&L" />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  )
}
