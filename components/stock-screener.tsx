'use client'

import { useState, useEffect } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Badge } from '@/components/ui/badge'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Search, Filter, TrendingUp, TrendingDown, ArrowUpDown } from 'lucide-react'
import type { StockQuote, CompanyOverview } from '@/lib/types'

interface ScreenerStock {
  symbol: string
  name: string
  price: number
  change: number
  changePercent: number
  marketCap: number
  peRatio: number
  sector: string
  volume: number
}

const SCREENER_STOCKS = [
  'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'AMD', 'INTC', 'NFLX',
  'JPM', 'BAC', 'WFC', 'GS', 'MS', 'V', 'MA', 'PYPL',
  'JNJ', 'PFE', 'UNH', 'MRK', 'ABBV', 'LLY',
  'XOM', 'CVX', 'COP',
  'DIS', 'CMCSA', 'T', 'VZ',
  'WMT', 'COST', 'HD', 'TGT', 'NKE',
]

const SECTORS = ['All', 'Technology', 'Healthcare', 'Financial Services', 'Consumer Cyclical', 'Energy', 'Communication Services', 'Consumer Defensive', 'Industrials']

export function StockScreener() {
  const [stocks, setStocks] = useState<ScreenerStock[]>([])
  const [filteredStocks, setFilteredStocks] = useState<ScreenerStock[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [searchTerm, setSearchTerm] = useState('')
  const [sortBy, setSortBy] = useState<keyof ScreenerStock>('marketCap')
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('desc')
  const [filters, setFilters] = useState({
    sector: 'All',
    minPrice: '',
    maxPrice: '',
    minPE: '',
    maxPE: '',
    minMarketCap: '',
  })

  useEffect(() => {
    const fetchStocks = async () => {
      setIsLoading(true)
      const results: ScreenerStock[] = []

      // Fetch data for all screener stocks in parallel batches
      const batchSize = 5
      for (let i = 0; i < SCREENER_STOCKS.length; i += batchSize) {
        const batch = SCREENER_STOCKS.slice(i, i + batchSize)
        const batchResults = await Promise.all(
          batch.map(async (symbol) => {
            try {
              const [quoteRes, overviewRes] = await Promise.all([
                fetch(`/api/stock/quote?symbol=${symbol}`),
                fetch(`/api/stock/overview?symbol=${symbol}`),
              ])

              if (quoteRes.ok && overviewRes.ok) {
                const quote: StockQuote = await quoteRes.json()
                const overview: CompanyOverview = await overviewRes.json()
                return {
                  symbol,
                  name: overview.name || symbol,
                  price: quote.price,
                  change: quote.change,
                  changePercent: quote.changePercent,
                  marketCap: overview.marketCap || 0,
                  peRatio: overview.peRatio || 0,
                  sector: overview.sector || 'Unknown',
                  volume: quote.volume,
                }
              }
            } catch {
              // Skip failed fetches
            }
            return null
          })
        )
        results.push(...batchResults.filter((r): r is ScreenerStock => r !== null))
      }

      setStocks(results)
      setFilteredStocks(results)
      setIsLoading(false)
    }

    fetchStocks()
  }, [])

  useEffect(() => {
    let filtered = [...stocks]

    // Apply search filter
    if (searchTerm) {
      const term = searchTerm.toLowerCase()
      filtered = filtered.filter(
        (s) => s.symbol.toLowerCase().includes(term) || s.name.toLowerCase().includes(term)
      )
    }

    // Apply sector filter
    if (filters.sector !== 'All') {
      filtered = filtered.filter((s) => s.sector === filters.sector)
    }

    // Apply price filters
    if (filters.minPrice) {
      filtered = filtered.filter((s) => s.price >= parseFloat(filters.minPrice))
    }
    if (filters.maxPrice) {
      filtered = filtered.filter((s) => s.price <= parseFloat(filters.maxPrice))
    }

    // Apply P/E filters
    if (filters.minPE) {
      filtered = filtered.filter((s) => s.peRatio >= parseFloat(filters.minPE))
    }
    if (filters.maxPE) {
      filtered = filtered.filter((s) => s.peRatio <= parseFloat(filters.maxPE))
    }

    // Apply market cap filter
    if (filters.minMarketCap) {
      const minCap = parseFloat(filters.minMarketCap) * 1e9 // Convert billions to actual
      filtered = filtered.filter((s) => s.marketCap >= minCap)
    }

    // Apply sorting
    filtered.sort((a, b) => {
      const aVal = a[sortBy]
      const bVal = b[sortBy]
      if (typeof aVal === 'number' && typeof bVal === 'number') {
        return sortDir === 'asc' ? aVal - bVal : bVal - aVal
      }
      if (typeof aVal === 'string' && typeof bVal === 'string') {
        return sortDir === 'asc' ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal)
      }
      return 0
    })

    setFilteredStocks(filtered)
  }, [stocks, searchTerm, filters, sortBy, sortDir])

  const handleSort = (column: keyof ScreenerStock) => {
    if (sortBy === column) {
      setSortDir(sortDir === 'asc' ? 'desc' : 'asc')
    } else {
      setSortBy(column)
      setSortDir('desc')
    }
  }

  const formatMarketCap = (cap: number) => {
    if (cap >= 1e12) return `$${(cap / 1e12).toFixed(2)}T`
    if (cap >= 1e9) return `$${(cap / 1e9).toFixed(2)}B`
    if (cap >= 1e6) return `$${(cap / 1e6).toFixed(2)}M`
    return `$${cap.toLocaleString()}`
  }

  const formatVolume = (vol: number) => {
    if (vol >= 1e6) return `${(vol / 1e6).toFixed(2)}M`
    if (vol >= 1e3) return `${(vol / 1e3).toFixed(2)}K`
    return vol.toString()
  }

  const resetFilters = () => {
    setFilters({
      sector: 'All',
      minPrice: '',
      maxPrice: '',
      minPE: '',
      maxPE: '',
      minMarketCap: '',
    })
    setSearchTerm('')
  }

  return (
    <div className="space-y-4">
      {/* Filters */}
      <Card className="bg-card/50 border-border/50">
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle className="text-foreground flex items-center gap-2">
              <Filter className="h-5 w-5" /> Screener Filters
            </CardTitle>
            <Button variant="outline" size="sm" onClick={resetFilters}>
              Reset
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="lg:col-span-2">
              <Label className="text-muted-foreground">Search</Label>
              <div className="relative mt-1">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                <Input
                  placeholder="Search symbol or name..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="pl-9 bg-input border-border"
                />
              </div>
            </div>
            <div>
              <Label className="text-muted-foreground">Sector</Label>
              <Select value={filters.sector} onValueChange={(v) => setFilters({ ...filters, sector: v })}>
                <SelectTrigger className="mt-1 bg-input border-border">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {SECTORS.map((sector) => (
                    <SelectItem key={sector} value={sector}>{sector}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div>
              <Label className="text-muted-foreground">Min Market Cap (B)</Label>
              <Input
                type="number"
                placeholder="e.g. 100"
                value={filters.minMarketCap}
                onChange={(e) => setFilters({ ...filters, minMarketCap: e.target.value })}
                className="mt-1 bg-input border-border"
              />
            </div>
            <div>
              <Label className="text-muted-foreground">Min Price</Label>
              <Input
                type="number"
                placeholder="$0"
                value={filters.minPrice}
                onChange={(e) => setFilters({ ...filters, minPrice: e.target.value })}
                className="mt-1 bg-input border-border"
              />
            </div>
            <div>
              <Label className="text-muted-foreground">Max Price</Label>
              <Input
                type="number"
                placeholder="$1000"
                value={filters.maxPrice}
                onChange={(e) => setFilters({ ...filters, maxPrice: e.target.value })}
                className="mt-1 bg-input border-border"
              />
            </div>
            <div>
              <Label className="text-muted-foreground">Min P/E Ratio</Label>
              <Input
                type="number"
                placeholder="0"
                value={filters.minPE}
                onChange={(e) => setFilters({ ...filters, minPE: e.target.value })}
                className="mt-1 bg-input border-border"
              />
            </div>
            <div>
              <Label className="text-muted-foreground">Max P/E Ratio</Label>
              <Input
                type="number"
                placeholder="100"
                value={filters.maxPE}
                onChange={(e) => setFilters({ ...filters, maxPE: e.target.value })}
                className="mt-1 bg-input border-border"
              />
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Results */}
      <Card className="bg-card/50 border-border/50">
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle className="text-foreground">
              {isLoading ? 'Loading stocks...' : `${filteredStocks.length} Stocks Found`}
            </CardTitle>
          </div>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <div className="flex items-center justify-center py-12">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary" />
            </div>
          ) : filteredStocks.length === 0 ? (
            <p className="text-muted-foreground text-center py-8">No stocks match your criteria.</p>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-border/50 text-muted-foreground text-sm">
                    <th className="text-left py-2 px-2 cursor-pointer hover:text-foreground" onClick={() => handleSort('symbol')}>
                      <div className="flex items-center gap-1">Symbol <ArrowUpDown className="h-3 w-3" /></div>
                    </th>
                    <th className="text-left py-2 px-2">Name</th>
                    <th className="text-right py-2 px-2 cursor-pointer hover:text-foreground" onClick={() => handleSort('price')}>
                      <div className="flex items-center justify-end gap-1">Price <ArrowUpDown className="h-3 w-3" /></div>
                    </th>
                    <th className="text-right py-2 px-2 cursor-pointer hover:text-foreground" onClick={() => handleSort('changePercent')}>
                      <div className="flex items-center justify-end gap-1">Change <ArrowUpDown className="h-3 w-3" /></div>
                    </th>
                    <th className="text-right py-2 px-2 cursor-pointer hover:text-foreground" onClick={() => handleSort('marketCap')}>
                      <div className="flex items-center justify-end gap-1">Market Cap <ArrowUpDown className="h-3 w-3" /></div>
                    </th>
                    <th className="text-right py-2 px-2 cursor-pointer hover:text-foreground" onClick={() => handleSort('peRatio')}>
                      <div className="flex items-center justify-end gap-1">P/E <ArrowUpDown className="h-3 w-3" /></div>
                    </th>
                    <th className="text-right py-2 px-2 cursor-pointer hover:text-foreground" onClick={() => handleSort('volume')}>
                      <div className="flex items-center justify-end gap-1">Volume <ArrowUpDown className="h-3 w-3" /></div>
                    </th>
                    <th className="text-left py-2 px-2">Sector</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredStocks.map((stock) => (
                    <tr key={stock.symbol} className="border-b border-border/30 hover:bg-muted/20 transition-colors">
                      <td className="py-3 px-2">
                        <span className="font-bold text-foreground">{stock.symbol}</span>
                      </td>
                      <td className="py-3 px-2 text-foreground max-w-[200px] truncate">{stock.name}</td>
                      <td className="text-right py-3 px-2 text-foreground font-medium">${stock.price.toFixed(2)}</td>
                      <td className="text-right py-3 px-2">
                        <div className={`flex items-center justify-end gap-1 ${stock.change >= 0 ? 'text-primary' : 'text-destructive'}`}>
                          {stock.change >= 0 ? <TrendingUp className="h-4 w-4" /> : <TrendingDown className="h-4 w-4" />}
                          <span>{stock.change >= 0 ? '+' : ''}{stock.changePercent.toFixed(2)}%</span>
                        </div>
                      </td>
                      <td className="text-right py-3 px-2 text-foreground">{formatMarketCap(stock.marketCap)}</td>
                      <td className="text-right py-3 px-2 text-foreground">{stock.peRatio > 0 ? stock.peRatio.toFixed(2) : 'N/A'}</td>
                      <td className="text-right py-3 px-2 text-foreground">{formatVolume(stock.volume)}</td>
                      <td className="py-3 px-2">
                        <Badge variant="outline" className="text-xs">{stock.sector}</Badge>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
