'use client'

import { useState, useEffect, useCallback, useRef } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Badge } from '@/components/ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { cn } from '@/lib/utils'
import { 
  Activity, 
  Zap, 
  Play,
  Pause,
  RotateCcw,
} from 'lucide-react'

type Side = 'BUY' | 'SELL'
type OrderType = 'LIMIT' | 'MARKET'

interface PriceLevel {
  price: number
  quantity: number
  orders: number
}

interface Trade {
  price: number
  quantity: number
  buyOrderId: string
  sellOrderId: string
  timestamp: number
  latencyNs: number
}

interface OrderBookSnapshot {
  bids: PriceLevel[]
  asks: PriceLevel[]
  spread: number
  midPrice: number
}

interface LatencyMetrics {
  avgLatencyNs: number
  minLatencyNs: number
  maxLatencyNs: number
  p99LatencyNs: number
  orderCount: number
}

interface OrderBookProps {
  symbol?: string
  initialPrice?: number
}

export function OrderBook({ symbol = 'AAPL', initialPrice = 150.00 }: OrderBookProps) {
  const [snapshot, setSnapshot] = useState<OrderBookSnapshot | null>(null)
  const [trades, setTrades] = useState<Trade[]>([])
  const [metrics, setMetrics] = useState<LatencyMetrics>({
    avgLatencyNs: 0, minLatencyNs: Infinity, maxLatencyNs: 0, p99LatencyNs: 0, orderCount: 0
  })
  const [isRunning, setIsRunning] = useState(false)
  const [orderPrice, setOrderPrice] = useState(initialPrice.toString())
  const [orderQty, setOrderQty] = useState('100')
  const [orderSide, setOrderSide] = useState<Side>('BUY')
  const [orderType, setOrderType] = useState<OrderType>('LIMIT')
  const [throughput, setThroughput] = useState(0)
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const orderCountRef = useRef(0)
  const lastSecondRef = useRef(Date.now())
  const latenciesRef = useRef<number[]>([])

  const addOrder = useCallback(async (side: Side, type: OrderType, price: number, quantity: number) => {
    try {
      const response = await fetch('/api/compute/orderbook', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'add_order', side, type, price, quantity })
      })
      const data = await response.json()
      setSnapshot(data.snapshot)
      
      if (data.trades?.length > 0) {
        setTrades(prev => [...data.trades, ...prev].slice(0, 10))
        for (const trade of data.trades) {
          latenciesRef.current.push(trade.latencyNs)
          if (latenciesRef.current.length > 1000) latenciesRef.current = latenciesRef.current.slice(-1000)
        }
        const sorted = [...latenciesRef.current].sort((a, b) => a - b)
        setMetrics({
          avgLatencyNs: sorted.reduce((a, b) => a + b, 0) / sorted.length,
          minLatencyNs: sorted[0] || 0,
          maxLatencyNs: sorted[sorted.length - 1] || 0,
          p99LatencyNs: sorted[Math.floor(sorted.length * 0.99)] || 0,
          orderCount: latenciesRef.current.length
        })
      }
      return data
    } catch (error) {
      console.error('Order failed:', error)
    }
  }, [])

  const simulateMarketActivity = useCallback(async () => {
    const midPrice = snapshot?.midPrice || initialPrice
    const side: Side = Math.random() > 0.5 ? 'BUY' : 'SELL'
    const type: OrderType = Math.random() > 0.3 ? 'LIMIT' : 'MARKET'
    let price = side === 'BUY' ? midPrice - 0.05 * Math.random() : midPrice + 0.05 * Math.random()
    const quantity = Math.floor(Math.random() * 500) + 50
    
    await addOrder(side, type, Math.round(price * 100) / 100, quantity)
    orderCountRef.current++
    
    const now = Date.now()
    if (now - lastSecondRef.current >= 1000) {
      setThroughput(orderCountRef.current)
      orderCountRef.current = 0
      lastSecondRef.current = now
    }
  }, [addOrder, snapshot, initialPrice])

  const toggleSimulation = useCallback(() => {
    if (isRunning) {
      if (intervalRef.current) clearInterval(intervalRef.current)
      intervalRef.current = null
    } else {
      intervalRef.current = setInterval(simulateMarketActivity, 50)
    }
    setIsRunning(!isRunning)
  }, [isRunning, simulateMarketActivity])

  useEffect(() => {
    const init = async () => {
      const mid = initialPrice
      for (let i = 1; i <= 10; i++) {
        await addOrder('BUY', 'LIMIT', Math.round((mid - i * 0.05) * 100) / 100, Math.floor(Math.random() * 1000) + 100)
        await addOrder('SELL', 'LIMIT', Math.round((mid + i * 0.05) * 100) / 100, Math.floor(Math.random() * 1000) + 100)
      }
    }
    init()
    return () => { if (intervalRef.current) clearInterval(intervalRef.current) }
  }, [initialPrice, addOrder])

  const submitOrder = async () => {
    const price = parseFloat(orderPrice)
    const qty = parseInt(orderQty)
    if (isNaN(price) || isNaN(qty) || qty <= 0) return
    await addOrder(orderSide, orderType, price, qty)
  }

  const resetBook = async () => {
    if (intervalRef.current) clearInterval(intervalRef.current)
    setIsRunning(false)
    await fetch('/api/compute/orderbook', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ action: 'reset' }) })
    const mid = initialPrice
    for (let i = 1; i <= 10; i++) {
      await addOrder('BUY', 'LIMIT', Math.round((mid - i * 0.05) * 100) / 100, Math.floor(Math.random() * 1000) + 100)
      await addOrder('SELL', 'LIMIT', Math.round((mid + i * 0.05) * 100) / 100, Math.floor(Math.random() * 1000) + 100)
    }
    setThroughput(0)
    orderCountRef.current = 0
    latenciesRef.current = []
    setMetrics({ avgLatencyNs: 0, minLatencyNs: Infinity, maxLatencyNs: 0, p99LatencyNs: 0, orderCount: 0 })
  }

  const maxVolume = Math.max(...(snapshot?.bids.map(b => b.quantity) || [1]), ...(snapshot?.asks.map(a => a.quantity) || [1]))
  const formatLatency = (ns: number) => {
    if (!isFinite(ns) || ns === 0) return '-'
    if (ns < 1000) return `${ns.toFixed(0)}ns`
    if (ns < 1000000) return `${(ns / 1000).toFixed(1)}µs`
    return `${(ns / 1000000).toFixed(2)}ms`
  }

  return (
    <Card className="w-full">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <CardTitle className="text-lg font-semibold">Limit Order Book</CardTitle>
            <Badge variant="outline" className="font-mono">{symbol}</Badge>
            <Badge variant="secondary" className="text-xs"><Zap className="w-3 h-3 mr-1" />C++ Engine</Badge>
          </div>
          <div className="flex items-center gap-2">
            <Button variant="outline" size="sm" onClick={toggleSimulation}>
              {isRunning ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
            </Button>
            <Button variant="outline" size="sm" onClick={resetBook}><RotateCcw className="w-4 h-4" /></Button>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="book" className="w-full">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="book">Order Book</TabsTrigger>
            <TabsTrigger value="trades">Trades</TabsTrigger>
            <TabsTrigger value="metrics">Latency</TabsTrigger>
          </TabsList>
          
          <TabsContent value="book" className="space-y-4">
            <div className="grid grid-cols-4 gap-2 text-sm">
              <div className="bg-muted/50 rounded p-2 text-center">
                <div className="text-muted-foreground text-xs">Spread</div>
                <div className="font-mono font-semibold">${snapshot?.spread.toFixed(2) || '0.00'}</div>
              </div>
              <div className="bg-muted/50 rounded p-2 text-center">
                <div className="text-muted-foreground text-xs">Mid Price</div>
                <div className="font-mono font-semibold">${snapshot?.midPrice.toFixed(2) || '0.00'}</div>
              </div>
              <div className="bg-muted/50 rounded p-2 text-center">
                <div className="text-muted-foreground text-xs">Orders/sec</div>
                <div className="font-mono font-semibold">{throughput}</div>
              </div>
              <div className="bg-muted/50 rounded p-2 text-center">
                <div className="text-muted-foreground text-xs">Avg Latency</div>
                <div className="font-mono font-semibold text-green-500">{formatLatency(metrics.avgLatencyNs)}</div>
              </div>
            </div>
            
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-1">
                <div className="text-xs font-medium text-muted-foreground flex justify-between px-2"><span>Bid</span><span>Size</span></div>
                {snapshot?.bids.slice(0, 10).map((level, i) => (
                  <div key={i} className="relative flex justify-between items-center px-2 py-1 text-sm">
                    <div className="absolute inset-0 bg-green-500/20 rounded" style={{ width: `${(level.quantity / maxVolume) * 100}%` }} />
                    <span className="relative font-mono text-green-600">${level.price.toFixed(2)}</span>
                    <span className="relative font-mono">{level.quantity.toLocaleString()}</span>
                  </div>
                ))}
              </div>
              <div className="space-y-1">
                <div className="text-xs font-medium text-muted-foreground flex justify-between px-2"><span>Ask</span><span>Size</span></div>
                {snapshot?.asks.slice(0, 10).map((level, i) => (
                  <div key={i} className="relative flex justify-between items-center px-2 py-1 text-sm">
                    <div className="absolute inset-0 bg-red-500/20 rounded right-0" style={{ width: `${(level.quantity / maxVolume) * 100}%`, marginLeft: 'auto' }} />
                    <span className="relative font-mono text-red-600">${level.price.toFixed(2)}</span>
                    <span className="relative font-mono">{level.quantity.toLocaleString()}</span>
                  </div>
                ))}
              </div>
            </div>
            
            <div className="border-t pt-4 grid grid-cols-5 gap-2 items-end">
              <div>
                <Label className="text-xs">Side</Label>
                <div className="flex gap-1">
                  <Button variant={orderSide === 'BUY' ? 'default' : 'outline'} size="sm" className={cn(orderSide === 'BUY' && 'bg-green-600 hover:bg-green-700')} onClick={() => setOrderSide('BUY')}>Buy</Button>
                  <Button variant={orderSide === 'SELL' ? 'default' : 'outline'} size="sm" className={cn(orderSide === 'SELL' && 'bg-red-600 hover:bg-red-700')} onClick={() => setOrderSide('SELL')}>Sell</Button>
                </div>
              </div>
              <div>
                <Label className="text-xs">Type</Label>
                <div className="flex gap-1">
                  <Button variant={orderType === 'LIMIT' ? 'default' : 'outline'} size="sm" onClick={() => setOrderType('LIMIT')}>Limit</Button>
                  <Button variant={orderType === 'MARKET' ? 'default' : 'outline'} size="sm" onClick={() => setOrderType('MARKET')}>Market</Button>
                </div>
              </div>
              <div><Label className="text-xs">Price</Label><Input type="number" value={orderPrice} onChange={(e) => setOrderPrice(e.target.value)} className="h-8" step="0.01" /></div>
              <div><Label className="text-xs">Quantity</Label><Input type="number" value={orderQty} onChange={(e) => setOrderQty(e.target.value)} className="h-8" /></div>
              <Button onClick={submitOrder} className="h-8">Submit</Button>
            </div>
          </TabsContent>
          
          <TabsContent value="trades">
            <div className="space-y-1">
              <div className="text-xs font-medium text-muted-foreground flex justify-between px-2 py-1 border-b"><span>Price</span><span>Size</span><span>Latency</span></div>
              {trades.map((trade, i) => (
                <div key={i} className="flex justify-between items-center px-2 py-1 text-sm hover:bg-muted/50 rounded">
                  <span className="font-mono">${trade.price.toFixed(2)}</span>
                  <span className="font-mono">{trade.quantity.toLocaleString()}</span>
                  <span className="font-mono text-xs text-green-500">{formatLatency(trade.latencyNs)}</span>
                </div>
              ))}
              {trades.length === 0 && <div className="text-center text-muted-foreground py-8">No trades yet</div>}
            </div>
          </TabsContent>
          
          <TabsContent value="metrics">
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-4">
                <div className="bg-muted/50 rounded-lg p-4"><div className="text-xs text-muted-foreground mb-1">Average Latency</div><div className="text-2xl font-mono font-bold text-green-500">{formatLatency(metrics.avgLatencyNs)}</div></div>
                <div className="bg-muted/50 rounded-lg p-4"><div className="text-xs text-muted-foreground mb-1">Min Latency</div><div className="text-2xl font-mono font-bold">{formatLatency(metrics.minLatencyNs)}</div></div>
              </div>
              <div className="space-y-4">
                <div className="bg-muted/50 rounded-lg p-4"><div className="text-xs text-muted-foreground mb-1">P99 Latency</div><div className="text-2xl font-mono font-bold text-yellow-500">{formatLatency(metrics.p99LatencyNs)}</div></div>
                <div className="bg-muted/50 rounded-lg p-4"><div className="text-xs text-muted-foreground mb-1">Max Latency</div><div className="text-2xl font-mono font-bold text-red-500">{formatLatency(metrics.maxLatencyNs)}</div></div>
              </div>
            </div>
            <div className="mt-4 p-4 bg-blue-500/10 rounded-lg border border-blue-500/20">
              <div className="flex items-center gap-2 text-blue-500 mb-2"><Zap className="w-4 h-4" /><span className="font-semibold">C++ Engine Performance</span></div>
              <p className="text-sm text-muted-foreground">Production C++ engine achieves <strong className="text-foreground">sub-10μs</strong> matching latency with memory pool allocation and cache-line aligned data structures.</p>
            </div>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  )
}
