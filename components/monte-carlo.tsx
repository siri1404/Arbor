'use client'

import { useState, useCallback, useEffect, useRef } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Label } from '@/components/ui/label'
import { Slider } from '@/components/ui/slider'
import { Badge } from '@/components/ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { cn } from '@/lib/utils'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend, AreaChart, Area, ReferenceLine, BarChart, Bar } from 'recharts'
import { Play, RotateCcw, TrendingUp, TrendingDown, Activity, Zap, AlertTriangle, Target } from 'lucide-react'

interface SimulationStats {
  meanFinalPrice: number
  stdFinalPrice: number
  minFinalPrice: number
  maxFinalPrice: number
  medianFinalPrice: number
  expectedReturn: number
  realizedVolatility: number
  maxDrawdown: number
}

interface VaRResult {
  var95: number
  var99: number
  cvar95: number
  cvar99: number
}

interface MonteCarloProps {
  initialPrice?: number
  symbol?: string
}

export function MonteCarlo({ initialPrice = 150, symbol = 'AAPL' }: MonteCarloProps) {
  const [spotPrice, setSpotPrice] = useState(initialPrice)
  const [drift, setDrift] = useState(8)
  const [volatility, setVolatility] = useState(25)
  const [timeHorizon, setTimeHorizon] = useState(1)
  const [numPaths, setNumPaths] = useState(1000)
  const [numSteps, setNumSteps] = useState(252)
  const [isRunning, setIsRunning] = useState(false)
  
  const [stats, setStats] = useState<SimulationStats | null>(null)
  const [varResult, setVarResult] = useState<VaRResult | null>(null)
  const [displayPaths, setDisplayPaths] = useState<number[][]>([])
  const [finalPrices, setFinalPrices] = useState<number[]>([])
  const [calculationTime, setCalculationTime] = useState(0)

  useEffect(() => {
    setSpotPrice(initialPrice)
    setStats(null)
    setVarResult(null)
    setDisplayPaths([])
    setFinalPrices([])
  }, [initialPrice, symbol])

  const runSimulation = useCallback(async () => {
    setIsRunning(true)
    
    try {
      const response = await fetch('/api/compute/montecarlo', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          action: 'simulate',
          params: {
            S0: spotPrice,
            mu: drift / 100,
            sigma: volatility / 100,
            T: timeHorizon,
            dt: timeHorizon / numSteps,
            numPaths,
            seed: Date.now()
          }
        })
      })
      
      const data = await response.json()
      
      setStats(data.stats)
      setVarResult(data.var)
      setDisplayPaths(data.paths.slice(0, 50))
      setFinalPrices(data.finalPrices)
      setCalculationTime(data.calculationTimeMs)
    } catch (error) {
      console.error('Simulation failed:', error)
    }
    
    setIsRunning(false)
  }, [spotPrice, drift, volatility, timeHorizon, numPaths, numSteps])

  const reset = () => {
    setStats(null)
    setVarResult(null)
    setDisplayPaths([])
    setFinalPrices([])
    setCalculationTime(0)
  }

  // Prepare chart data
  const pathsData = displayPaths.length > 0 ? Array.from({ length: displayPaths[0].length }, (_, t) => {
    const point: any = { time: t / numSteps * timeHorizon }
    displayPaths.slice(0, 20).forEach((path, i) => {
      point[`path${i}`] = path[t]
    })
    return point
  }) : []

  // Distribution histogram
  const histogramData = finalPrices.length > 0 ? (() => {
    const min = Math.min(...finalPrices)
    const max = Math.max(...finalPrices)
    const bins = 30
    const binWidth = (max - min) / bins
    const histogram: { range: string; count: number; price: number }[] = []
    
    for (let i = 0; i < bins; i++) {
      const low = min + i * binWidth
      const high = low + binWidth
      const count = finalPrices.filter(p => p >= low && p < high).length
      histogram.push({
        range: `$${low.toFixed(0)}`,
        count,
        price: (low + high) / 2
      })
    }
    return histogram
  })() : []

  return (
    <Card className="w-full">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <CardTitle className="text-lg font-semibold">Monte Carlo Simulation</CardTitle>
            <Badge variant="outline" className="font-mono">{symbol}</Badge>
            <Badge variant="secondary" className="text-xs"><Zap className="w-3 h-3 mr-1" />GBM Engine</Badge>
          </div>
          <div className="flex items-center gap-2">
            <Button variant="default" size="sm" onClick={runSimulation} disabled={isRunning}>
              <Play className="w-4 h-4 mr-1" />{isRunning ? 'Running...' : 'Run'}
            </Button>
            <Button variant="outline" size="sm" onClick={reset}><RotateCcw className="w-4 h-4" /></Button>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="simulation" className="w-full">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="simulation">Simulation</TabsTrigger>
            <TabsTrigger value="distribution">Distribution</TabsTrigger>
            <TabsTrigger value="risk">Risk Metrics</TabsTrigger>
          </TabsList>
          
          <TabsContent value="simulation" className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-4">
                <div>
                  <Label className="text-xs">Initial Price: ${spotPrice.toFixed(2)}</Label>
                  <Slider value={[spotPrice]} onValueChange={([v]) => setSpotPrice(v)} min={50} max={300} step={1} className="mt-2" />
                </div>
                <div>
                  <Label className="text-xs">Expected Return: {drift}%</Label>
                  <Slider value={[drift]} onValueChange={([v]) => setDrift(v)} min={-20} max={50} step={1} className="mt-2" />
                </div>
                <div>
                  <Label className="text-xs">Volatility: {volatility}%</Label>
                  <Slider value={[volatility]} onValueChange={([v]) => setVolatility(v)} min={5} max={100} step={1} className="mt-2" />
                </div>
                <div>
                  <Label className="text-xs">Time Horizon: {timeHorizon} year(s)</Label>
                  <Slider value={[timeHorizon]} onValueChange={([v]) => setTimeHorizon(v)} min={0.25} max={5} step={0.25} className="mt-2" />
                </div>
                <div>
                  <Label className="text-xs">Number of Paths: {numPaths.toLocaleString()}</Label>
                  <Slider value={[numPaths]} onValueChange={([v]) => setNumPaths(v)} min={100} max={10000} step={100} className="mt-2" />
                </div>
              </div>
              
              <div className="space-y-2">
                {stats && (
                  <>
                    <div className="grid grid-cols-2 gap-2">
                      <div className="bg-muted/50 rounded p-3">
                        <div className="text-xs text-muted-foreground">Mean Final</div>
                        <div className="font-mono font-semibold">${stats.meanFinalPrice.toFixed(2)}</div>
                      </div>
                      <div className="bg-muted/50 rounded p-3">
                        <div className="text-xs text-muted-foreground">Std Dev</div>
                        <div className="font-mono font-semibold">${stats.stdFinalPrice.toFixed(2)}</div>
                      </div>
                      <div className="bg-muted/50 rounded p-3">
                        <div className="text-xs text-muted-foreground">Min</div>
                        <div className="font-mono font-semibold text-red-500">${stats.minFinalPrice.toFixed(2)}</div>
                      </div>
                      <div className="bg-muted/50 rounded p-3">
                        <div className="text-xs text-muted-foreground">Max</div>
                        <div className="font-mono font-semibold text-green-500">${stats.maxFinalPrice.toFixed(2)}</div>
                      </div>
                    </div>
                    <div className="bg-blue-500/10 rounded p-3 border border-blue-500/20">
                      <div className="text-xs text-blue-500 mb-1">Performance</div>
                      <div className="font-mono text-sm">
                        {numPaths.toLocaleString()} paths in {calculationTime.toFixed(1)}ms
                        <span className="text-muted-foreground"> ({(numPaths / calculationTime * 1000).toFixed(0)} paths/sec)</span>
                      </div>
                    </div>
                  </>
                )}
                {!stats && <div className="text-center text-muted-foreground py-8">Run simulation to see results</div>}
              </div>
            </div>
            
            {displayPaths.length > 0 && (
              <div className="h-48">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={pathsData}>
                    <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                    <XAxis dataKey="time" tickFormatter={(v) => `${v.toFixed(1)}y`} />
                    <YAxis tickFormatter={(v) => `$${v.toFixed(0)}`} domain={['auto', 'auto']} />
                    <Tooltip formatter={(v: number) => `$${v.toFixed(2)}`} />
                    {displayPaths.slice(0, 20).map((_, i) => (
                      <Line key={i} type="monotone" dataKey={`path${i}`} stroke={`hsl(${i * 18}, 70%, 50%)`} dot={false} strokeWidth={1} opacity={0.6} />
                    ))}
                    <ReferenceLine y={spotPrice} stroke="#888" strokeDasharray="3 3" />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            )}
          </TabsContent>
          
          <TabsContent value="distribution">
            {histogramData.length > 0 ? (
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={histogramData}>
                    <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                    <XAxis dataKey="range" angle={-45} textAnchor="end" height={60} />
                    <YAxis />
                    <Tooltip />
                    <Bar dataKey="count" fill="#3b82f6" />
                    <ReferenceLine x={`$${spotPrice.toFixed(0)}`} stroke="#ef4444" strokeDasharray="3 3" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            ) : (
              <div className="text-center text-muted-foreground py-16">Run simulation to see distribution</div>
            )}
          </TabsContent>
          
          <TabsContent value="risk">
            {varResult ? (
              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-yellow-500/10 rounded-lg p-4 border border-yellow-500/20">
                    <div className="flex items-center gap-2 text-yellow-500 mb-2">
                      <AlertTriangle className="w-4 h-4" />
                      <span className="font-semibold">VaR (95%)</span>
                    </div>
                    <div className="text-2xl font-mono font-bold">${varResult.var95.toFixed(2)}</div>
                    <div className="text-xs text-muted-foreground mt-1">5% chance of losing more</div>
                  </div>
                  <div className="bg-red-500/10 rounded-lg p-4 border border-red-500/20">
                    <div className="flex items-center gap-2 text-red-500 mb-2">
                      <AlertTriangle className="w-4 h-4" />
                      <span className="font-semibold">VaR (99%)</span>
                    </div>
                    <div className="text-2xl font-mono font-bold">${varResult.var99.toFixed(2)}</div>
                    <div className="text-xs text-muted-foreground mt-1">1% chance of losing more</div>
                  </div>
                  <div className="bg-orange-500/10 rounded-lg p-4 border border-orange-500/20">
                    <div className="flex items-center gap-2 text-orange-500 mb-2">
                      <Target className="w-4 h-4" />
                      <span className="font-semibold">CVaR (95%)</span>
                    </div>
                    <div className="text-2xl font-mono font-bold">${varResult.cvar95.toFixed(2)}</div>
                    <div className="text-xs text-muted-foreground mt-1">Expected loss in worst 5%</div>
                  </div>
                  <div className="bg-purple-500/10 rounded-lg p-4 border border-purple-500/20">
                    <div className="flex items-center gap-2 text-purple-500 mb-2">
                      <Target className="w-4 h-4" />
                      <span className="font-semibold">CVaR (99%)</span>
                    </div>
                    <div className="text-2xl font-mono font-bold">${varResult.cvar99.toFixed(2)}</div>
                    <div className="text-xs text-muted-foreground mt-1">Expected loss in worst 1%</div>
                  </div>
                </div>
                {stats && (
                  <div className="bg-muted/50 rounded-lg p-4">
                    <div className="text-sm font-semibold mb-2">Additional Metrics</div>
                    <div className="grid grid-cols-3 gap-4 text-sm">
                      <div><span className="text-muted-foreground">Expected Return:</span> <span className="font-mono">{(stats.expectedReturn * 100).toFixed(1)}%</span></div>
                      <div><span className="text-muted-foreground">Realized Vol:</span> <span className="font-mono">{(stats.realizedVolatility * 100).toFixed(1)}%</span></div>
                      <div><span className="text-muted-foreground">Max Drawdown:</span> <span className="font-mono text-red-500">{(stats.maxDrawdown * 100).toFixed(1)}%</span></div>
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="text-center text-muted-foreground py-16">Run simulation to see risk metrics</div>
            )}
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  )
}
