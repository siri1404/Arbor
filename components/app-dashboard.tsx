'use client'

import { useState, useEffect, useCallback } from 'react'
import useSWR from 'swr'
import { StockCard } from '@/components/stock-card'
import { StockChart } from '@/components/stock-chart'
import { PortfolioTracker } from '@/components/portfolio-tracker'
import { StockScreener } from '@/components/stock-screener'

import { AdvancedAnalytics } from '@/components/advanced-analytics'
import { OrderBook } from '@/components/order-book'
import { OptionsPricer } from '@/components/options-pricer'
import { MonteCarlo } from '@/components/monte-carlo'
import { MarketDataParser } from '@/components/market-data-parser'
import { DEFAULT_STOCKS } from '@/lib/types'
import type { StockQuote, StockTimeSeries, CompanyOverview } from '@/lib/types'
import { RefreshCw, AlertCircle, LogOut, TrendingUp, Search, Briefcase, LineChart, Activity, Calculator, Cpu, FileCode } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Badge } from '@/components/ui/badge'
import { createClient } from '@/lib/supabase/client'
import type { User } from '@supabase/supabase-js'
import { useRouter } from 'next/navigation'

const fetcher = (url: string) => fetch(url).then((res) => {
  if (!res.ok) throw new Error('Failed to fetch')
  return res.json()
})

export function AppDashboard() {
  const router = useRouter()
  const [user, setUser] = useState<User | null>(null)
  const [isLoadingAuth, setIsLoadingAuth] = useState(false)
  const [selectedSymbol, setSelectedSymbol] = useState<string | null>('AAPL')
  const [chartInterval, setChartInterval] = useState('daily')
  const [quotesMap, setQuotesMap] = useState<Record<string, StockQuote>>({})
  const [quotesLoading, setQuotesLoading] = useState<Record<string, boolean>>({})
  const [quotesError, setQuotesError] = useState<string | null>(null)
  const [activeTab, setActiveTab] = useState('market')
  const supabase = createClient()

  useEffect(() => {
    const { data: { subscription } } = supabase.auth.onAuthStateChange((_, session) => {
      setUser(session?.user ?? null)
      if (!session?.user) {
        router.push('/')
      }
    })

    return () => subscription.unsubscribe()
  }, [supabase.auth, router])

  const { data: timeseries, isLoading: timeseriesLoading } = useSWR<StockTimeSeries>(
    selectedSymbol ? `/api/stock/timeseries?symbol=${selectedSymbol}&interval=${chartInterval}` : null,
    fetcher,
    { revalidateOnFocus: false }
  )

  const { data: overview, isLoading: overviewLoading } = useSWR<CompanyOverview>(
    selectedSymbol ? `/api/stock/overview?symbol=${selectedSymbol}` : null,
    fetcher,
    { revalidateOnFocus: false }
  )

  const fetchQuotes = useCallback(async () => {
    setQuotesError(null)
    const loadingState: Record<string, boolean> = {}
    DEFAULT_STOCKS.forEach(symbol => { loadingState[symbol] = true })
    setQuotesLoading(loadingState)

    let successCount = 0

    const results = await Promise.all(
      DEFAULT_STOCKS.map(async (symbol) => {
        try {
          const res = await fetch(`/api/stock/quote?symbol=${symbol}`)
          if (res.ok) {
            const data = await res.json()
            return { symbol, data, success: true }
          } else if (res.status === 429) {
            return { symbol, data: null, success: false, rateLimited: true }
          }
          return { symbol, data: null, success: false }
        } catch {
          return { symbol, data: null, success: false }
        }
      })
    )

    const newQuotesMap: Record<string, StockQuote> = {}
    let rateLimitHit = false

    for (const result of results) {
      if (result.success && result.data) {
        newQuotesMap[result.symbol] = result.data
        successCount++
      }
      if (result.rateLimited) {
        rateLimitHit = true
      }
    }

    setQuotesMap(newQuotesMap)
    setQuotesLoading({})

    if (rateLimitHit) {
      setQuotesError(`Rate limit reached. Loaded ${successCount} of ${DEFAULT_STOCKS.length} stocks.`)
    } else if (successCount === 0) {
      setQuotesError('Unable to fetch stock data. Please verify your FINNHUB_API_KEY is set correctly.')
    }
  }, [])

  useEffect(() => {
    fetchQuotes()
  }, [fetchQuotes])

  const handleStockSelect = (symbol: string) => {
    setSelectedSymbol(symbol)
  }

  const handleRefresh = () => {
    setQuotesMap({})
    fetchQuotes()
  }

  const handleLogout = async () => {
    await supabase.auth.signOut()
    router.push('/')
  }

  const currentPrice = quotesMap[selectedSymbol || '']?.c || overview?.price || 150

  if (isLoadingAuth) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary" />
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-background">
      <header className="border-b border-border bg-card/50 backdrop-blur-sm sticky top-0 z-50">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              {/* Geometric Tree Logo - same as landing page */}
              <div className="relative w-8 h-8 flex items-center justify-center">
                <svg width="32" height="32" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <circle cx="16" cy="26" r="2" fill="currentColor" className="text-primary" />
                  <line x1="16" y1="24" x2="16" y2="14" stroke="currentColor" strokeWidth="1.5" className="text-primary" />
                  <circle cx="8" cy="12" r="2" fill="currentColor" className="text-primary" />
                  <circle cx="24" cy="12" r="2" fill="currentColor" className="text-primary" />
                  <circle cx="16" cy="8" r="2.5" fill="currentColor" className="text-primary" />
                  <line x1="16" y1="14" x2="8" y2="12" stroke="currentColor" strokeWidth="1.5" className="text-primary" />
                  <line x1="16" y1="14" x2="24" y2="12" stroke="currentColor" strokeWidth="1.5" className="text-primary" />
                  <line x1="16" y1="14" x2="16" y2="10.5" stroke="currentColor" strokeWidth="1.5" className="text-primary" />
                  <circle cx="4" cy="6" r="1.5" fill="currentColor" opacity="0.8" className="text-primary" />
                  <circle cx="12" cy="4" r="1.5" fill="currentColor" opacity="0.8" className="text-primary" />
                  <circle cx="20" cy="4" r="1.5" fill="currentColor" opacity="0.8" className="text-primary" />
                  <circle cx="28" cy="6" r="1.5" fill="currentColor" opacity="0.8" className="text-primary" />
                  <line x1="8" y1="12" x2="4" y2="6" stroke="currentColor" strokeWidth="1" opacity="0.6" className="text-primary" />
                  <line x1="8" y1="12" x2="12" y2="4" stroke="currentColor" strokeWidth="1" opacity="0.6" className="text-primary" />
                  <line x1="24" y1="12" x2="20" y2="4" stroke="currentColor" strokeWidth="1" opacity="0.6" className="text-primary" />
                  <line x1="24" y1="12" x2="28" y2="6" stroke="currentColor" strokeWidth="1" opacity="0.6" className="text-primary" />
                </svg>
              </div>
              
              <div>
                <h1 className="text-2xl font-bold text-foreground tracking-tight font-sans">ARBOR</h1>
                <p className="text-xs text-muted-foreground">Quantitative Trading Systems</p>
              </div>
            </div>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-4 py-6">
        {quotesError && (
          <Alert variant="destructive" className="bg-destructive/10 border-destructive/30 mb-6">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>{quotesError}</AlertDescription>
          </Alert>
        )}

        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
          <TabsList className="bg-card/50 border border-border/50 p-1 flex-wrap h-auto">
            <TabsTrigger value="orderbook" className="gap-2 data-[state=active]:bg-primary data-[state=active]:text-primary-foreground">
              <Activity className="h-4 w-4" />
              <span className="hidden sm:inline">Order Book</span>
            </TabsTrigger>
            <TabsTrigger value="options" className="gap-2 data-[state=active]:bg-primary data-[state=active]:text-primary-foreground">
              <Calculator className="h-4 w-4" />
              <span className="hidden sm:inline">Options Pricer</span>
            </TabsTrigger>
            <TabsTrigger value="montecarlo" className="gap-2 data-[state=active]:bg-primary data-[state=active]:text-primary-foreground">
              <Cpu className="h-4 w-4" />
              <span className="hidden sm:inline">Monte Carlo</span>
            </TabsTrigger>
            <TabsTrigger value="parser" className="gap-2 data-[state=active]:bg-primary data-[state=active]:text-primary-foreground">
              <FileCode className="h-4 w-4" />
              <span className="hidden sm:inline">Data Parser</span>
            </TabsTrigger>
            <TabsTrigger value="market" className="gap-2 data-[state=active]:bg-primary data-[state=active]:text-primary-foreground">
              <TrendingUp className="h-4 w-4" />
              <span className="hidden sm:inline">Market</span>
            </TabsTrigger>

            <TabsTrigger value="analytics" className="gap-2 data-[state=active]:bg-primary data-[state=active]:text-primary-foreground">
              <LineChart className="h-4 w-4" />
              <span className="hidden sm:inline">Analytics</span>
            </TabsTrigger>
            <TabsTrigger value="screener" className="gap-2 data-[state=active]:bg-primary data-[state=active]:text-primary-foreground">
              <Search className="h-4 w-4" />
              <span className="hidden sm:inline">Screener</span>
            </TabsTrigger>
            {user && (
              <TabsTrigger value="portfolio" className="gap-2 data-[state=active]:bg-primary data-[state=active]:text-primary-foreground">
                <Briefcase className="h-4 w-4" />
                <span className="hidden sm:inline">Portfolio</span>
              </TabsTrigger>
            )}
          </TabsList>

          {/* Order Book Tab - NEW */}
          <TabsContent value="orderbook" className="space-y-6">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-xl font-bold">Limit Order Book Engine</h2>
                <p className="text-sm text-muted-foreground">Price-Time Priority Matching with Nanosecond Latency Tracking</p>
              </div>
              <div className="flex flex-wrap gap-2">
                {DEFAULT_STOCKS.map((symbol) => (
                  <Button
                    key={symbol}
                    variant={selectedSymbol === symbol ? 'default' : 'outline'}
                    size="sm"
                    onClick={() => handleStockSelect(symbol)}
                  >
                    {symbol}
                  </Button>
                ))}
              </div>
            </div>
            <OrderBook key={`${selectedSymbol}-${currentPrice}`} symbol={selectedSymbol || 'AAPL'} initialPrice={currentPrice} />
          </TabsContent>

          {/* Options Pricing Tab - NEW */}
          <TabsContent value="options" className="space-y-6">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-xl font-bold">Black-Scholes Options Pricer</h2>
                <p className="text-sm text-muted-foreground">Full Greeks Calculation (Delta, Gamma, Theta, Vega, Rho) with IV Solver</p>
              </div>
              <div className="flex flex-wrap gap-2">
                {DEFAULT_STOCKS.map((symbol) => (
                  <Button
                    key={symbol}
                    variant={selectedSymbol === symbol ? 'default' : 'outline'}
                    size="sm"
                    onClick={() => handleStockSelect(symbol)}
                  >
                    {symbol}
                  </Button>
                ))}
              </div>
            </div>
            <OptionsPricer key={`${selectedSymbol}-${currentPrice}`} initialSpot={currentPrice} symbol={selectedSymbol || 'AAPL'} />
          </TabsContent>

          {/* Monte Carlo Tab - NEW */}
          <TabsContent value="montecarlo" className="space-y-6">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-xl font-bold">Monte Carlo Simulation Engine</h2>
                <p className="text-sm text-muted-foreground">GBM Price Simulation, VaR, Expected Shortfall (CVaR)</p>
              </div>
              <div className="flex flex-wrap gap-2">
                {DEFAULT_STOCKS.map((symbol) => (
                  <Button
                    key={symbol}
                    variant={selectedSymbol === symbol ? 'default' : 'outline'}
                    size="sm"
                    onClick={() => handleStockSelect(symbol)}
                  >
                    {symbol}
                  </Button>
                ))}
              </div>
            </div>
            <MonteCarlo key={`${selectedSymbol}-${currentPrice}`} initialPrice={currentPrice} symbol={selectedSymbol || 'AAPL'} />
          </TabsContent>

          {/* Market Data Parser Tab - NEW */}
          <TabsContent value="parser" className="space-y-6">
            <div>
              <h2 className="text-xl font-bold">High-Performance Market Data Parser</h2>
              <p className="text-sm text-muted-foreground">FIX Protocol Parser, CSV/Tick Data Processing with Latency Benchmarks</p>
            </div>
            <MarketDataParser />
          </TabsContent>

          {/* Market Tab */}
          <TabsContent value="market" className="space-y-6">
            <section>
              <h2 className="text-lg font-semibold mb-4 text-foreground">S&P 500 Leaders</h2>
              <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-7 gap-4">
                {DEFAULT_STOCKS.map((symbol) => (
                  <StockCard
                    key={symbol}
                    quote={quotesMap[symbol] || null}
                    isLoading={quotesLoading[symbol]}
                    isSelected={selectedSymbol === symbol}
                    onClick={() => handleStockSelect(symbol)}
                  />
                ))}
              </div>
            </section>

            <div className="grid lg:grid-cols-2 gap-6">
              <StockChart
                data={timeseries || null}
                isLoading={timeseriesLoading}
                symbol={selectedSymbol || undefined}
                interval={chartInterval}
                onIntervalChange={setChartInterval}
              />
              <div className="space-y-4">
                <div className="p-4 bg-card rounded-lg border">
                  <h3 className="font-semibold mb-2">Quick Links</h3>
                  <div className="grid grid-cols-2 gap-2">
                    <Button variant="outline" size="sm" onClick={() => setActiveTab('orderbook')}>
                      Order Book
                    </Button>
                    <Button variant="outline" size="sm" onClick={() => setActiveTab('options')}>
                      Options Pricer
                    </Button>
                    <Button variant="outline" size="sm" onClick={() => setActiveTab('montecarlo')}>
                      Monte Carlo
                    </Button>
                    <Button variant="outline" size="sm" onClick={() => setActiveTab('parser')}>
                      Data Parser
                    </Button>
                  </div>
                </div>
              </div>
            </div>
          </TabsContent>

          {/* Advanced Analytics Tab */}
          <TabsContent value="analytics" className="space-y-6">
            <section>
              <h3 className="text-lg font-semibold mb-4 text-foreground">Select Stock for Analysis</h3>
              <div className="flex flex-wrap gap-2">
                {DEFAULT_STOCKS.map((symbol) => (
                  <Button
                    key={symbol}
                    variant={selectedSymbol === symbol ? 'default' : 'outline'}
                    size="sm"
                    onClick={() => handleStockSelect(symbol)}
                    className="transition-all duration-200 hover:scale-105"
                  >
                    {symbol}
                  </Button>
                ))}
              </div>
            </section>
            {selectedSymbol && timeseries ? (
              <AdvancedAnalytics 
                symbol={selectedSymbol} 
                data={timeseries.data} 
                quote={quotesMap[selectedSymbol] || null}
              />
            ) : (
              <div className="text-center py-12 text-muted-foreground">
                Select a stock above to view advanced analytics
              </div>
            )}
          </TabsContent>

          {/* Stock Screener Tab */}
          <TabsContent value="screener">
            <StockScreener />
          </TabsContent>

          {/* Portfolio Tab (Authenticated Only) */}
          {user && (
            <TabsContent value="portfolio">
              <PortfolioTracker />
            </TabsContent>
          )}
        </Tabs>
      </main>

      <footer className="border-t border-border bg-card/30 mt-12">
        <div className="container mx-auto px-4 py-6">
          <p className="text-center text-sm text-muted-foreground">
            ARBOR QUANT - Quantitative Trading Systems
            <br />
            <span className="text-xs">Order Book | Black-Scholes | Monte Carlo | FIX Protocol | Market Data Parsing</span>
          </p>
        </div>
      </footer>
    </div>
  )
}
