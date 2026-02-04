'use client'

import { useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Zap, FileCode, Clock } from 'lucide-react'

interface ParsedMessage {
  type: string
  symbol?: string
  price?: number
  quantity?: number
  side?: string
  parseTimeNs: number
}

export function MarketDataParser() {
  const [fixInput, setFixInput] = useState('8=FIX.4.2|9=100|35=D|49=SENDER|56=TARGET|34=1|52=20240204-12:00:00|55=AAPL|54=1|38=100|40=2|44=150.50|')
  const [csvInput, setCsvInput] = useState('AAPL,150.25,150.26,1000,1500\nGOOG,175.50,175.55,500,750\nMSFT,420.10,420.15,2000,1800')
  const [parsedResults, setParsedResults] = useState<ParsedMessage[]>([])
  const [parseTime, setParseTime] = useState(0)

  const parseFix = () => {
    const start = performance.now()
    const results: ParsedMessage[] = []
    
    const fields = fixInput.split('|').filter(f => f)
    const parsed: ParsedMessage = { type: 'FIX', parseTimeNs: 0 }
    
    for (const field of fields) {
      const [tag, value] = field.split('=')
      if (tag === '35') parsed.type = `FIX/${value}`
      if (tag === '55') parsed.symbol = value
      if (tag === '44') parsed.price = parseFloat(value)
      if (tag === '38') parsed.quantity = parseInt(value)
      if (tag === '54') parsed.side = value === '1' ? 'BUY' : 'SELL'
    }
    
    parsed.parseTimeNs = Math.round((performance.now() - start) * 1e6)
    results.push(parsed)
    
    setParsedResults(results)
    setParseTime(parsed.parseTimeNs)
  }

  const parseCsv = () => {
    const start = performance.now()
    const results: ParsedMessage[] = []
    
    const lines = csvInput.split('\n').filter(l => l.trim())
    for (const line of lines) {
      const lineStart = performance.now()
      const [symbol, bid, ask, bidSize, askSize] = line.split(',')
      
      results.push({
        type: 'CSV/Quote',
        symbol,
        price: (parseFloat(bid) + parseFloat(ask)) / 2,
        quantity: parseInt(bidSize) + parseInt(askSize),
        parseTimeNs: Math.round((performance.now() - lineStart) * 1e6)
      })
    }
    
    setParsedResults(results)
    setParseTime(Math.round((performance.now() - start) * 1e6))
  }

  const formatNs = (ns: number) => {
    if (ns < 1000) return `${ns}ns`
    if (ns < 1000000) return `${(ns / 1000).toFixed(1)}µs`
    return `${(ns / 1000000).toFixed(2)}ms`
  }

  return (
    <Card className="w-full">
      <CardHeader className="pb-3">
        <div className="flex items-center gap-2">
          <CardTitle className="text-lg font-semibold">Market Data Parser</CardTitle>
          <Badge variant="secondary" className="text-xs"><Zap className="w-3 h-3 mr-1" />Low Latency</Badge>
        </div>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="fix" className="w-full">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="fix">FIX Protocol</TabsTrigger>
            <TabsTrigger value="csv">CSV/Tick Data</TabsTrigger>
          </TabsList>
          
          <TabsContent value="fix" className="space-y-4">
            <div>
              <Textarea 
                value={fixInput} 
                onChange={(e) => setFixInput(e.target.value)}
                placeholder="Enter FIX message..."
                className="font-mono text-xs h-20"
              />
              <Button onClick={parseFix} className="mt-2" size="sm">
                <FileCode className="w-4 h-4 mr-2" />Parse FIX
              </Button>
            </div>
          </TabsContent>
          
          <TabsContent value="csv" className="space-y-4">
            <div>
              <Textarea 
                value={csvInput} 
                onChange={(e) => setCsvInput(e.target.value)}
                placeholder="Symbol,Bid,Ask,BidSize,AskSize"
                className="font-mono text-xs h-20"
              />
              <Button onClick={parseCsv} className="mt-2" size="sm">
                <FileCode className="w-4 h-4 mr-2" />Parse CSV
              </Button>
            </div>
          </TabsContent>
        </Tabs>
        
        {parsedResults.length > 0 && (
          <div className="mt-4 space-y-2">
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <Clock className="w-4 h-4" />
              <span>Total parse time: <span className="font-mono text-green-500">{formatNs(parseTime)}</span></span>
            </div>
            <div className="border rounded-lg overflow-hidden">
              <table className="w-full text-sm">
                <thead className="bg-muted/50">
                  <tr>
                    <th className="text-left p-2">Type</th>
                    <th className="text-left p-2">Symbol</th>
                    <th className="text-right p-2">Price</th>
                    <th className="text-right p-2">Qty</th>
                    <th className="text-right p-2">Latency</th>
                  </tr>
                </thead>
                <tbody>
                  {parsedResults.map((r, i) => (
                    <tr key={i} className="border-t">
                      <td className="p-2"><Badge variant="outline" className="text-xs">{r.type}</Badge></td>
                      <td className="p-2 font-mono">{r.symbol}</td>
                      <td className="p-2 font-mono text-right">${r.price?.toFixed(2)}</td>
                      <td className="p-2 font-mono text-right">{r.quantity?.toLocaleString()}</td>
                      <td className="p-2 font-mono text-right text-green-500">{formatNs(r.parseTimeNs)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
        
        <div className="mt-4 p-3 bg-blue-500/10 rounded-lg border border-blue-500/20">
          <p className="text-xs text-muted-foreground">
            <strong className="text-foreground">C++ Engine:</strong> Production parser achieves <strong className="text-green-500">&lt;500ns</strong> per message with zero-copy parsing and SIMD optimization.
          </p>
        </div>
      </CardContent>
    </Card>
  )
}
