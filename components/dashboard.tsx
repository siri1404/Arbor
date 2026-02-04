'use client'

import { useState, useEffect, useCallback } from 'react'
import useSWR from 'swr'
import { StockCard } from '@/components/stock-card'
import { StockChart } from '@/components/stock-chart'
import { AgentPanel } from '@/components/agent-panel'
import { ChatInterface } from '@/components/chat-interface'
import { CompanyInfo } from '@/components/company-info'
import { DEFAULT_STOCKS } from '@/lib/types'
import type { StockQuote, StockTimeSeries, CompanyOverview, OrchestratorResult } from '@/lib/types'
import { RefreshCw, AlertCircle } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Alert, AlertDescription } from '@/components/ui/alert'

const fetcher = (url: string) => fetch(url).then((res) => {
  if (!res.ok) throw new Error('Failed to fetch')
  return res.json()
})

export function Dashboard() {
  const [selectedSymbol, setSelectedSymbol] = useState<string | null>(null)
  const [chartInterval, setChartInterval] = useState('daily')
  const [analysisResult, setAnalysisResult] = useState<OrchestratorResult | null>(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [quotesMap, setQuotesMap] = useState<Record<string, StockQuote>>({})
  const [quotesLoading, setQuotesLoading] = useState<Record<string, boolean>>({})
  const [quotesError, setQuotesError] = useState<string | null>(null)

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

    // Finnhub allows 60 calls/min - fetch all stocks in parallel
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
      setQuotesError('Unable to fetch stock data. Please verify your FINNHUB_API_KEY is set correctly in the Vars sidebar.')
    }
  }, [])

  useEffect(() => {
    fetchQuotes()
  }, [fetchQuotes])

  const handleAnalysisComplete = (result: OrchestratorResult) => {
    setAnalysisResult(result)
    setIsAnalyzing(false)
    if (result.symbol !== selectedSymbol) {
      setSelectedSymbol(result.symbol)
    }
  }

  const handleAnalysisStart = () => {
    setIsAnalyzing(true)
  }

  const handleStockSelect = (symbol: string) => {
    setSelectedSymbol(symbol)
    setAnalysisResult(null)
  }

  const handleRefresh = () => {
    setQuotesMap({})
    fetchQuotes()
  }

  return (
    <div className="min-h-screen bg-background">
      <header className="border-b border-border bg-card/50 backdrop-blur-sm sticky top-0 z-50">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-foreground">MarketProphet</h1>
              <p className="text-sm text-muted-foreground">AI-Powered Financial Intelligence</p>
            </div>
            <Button variant="outline" size="sm" onClick={handleRefresh}>
              <RefreshCw className="h-4 w-4 mr-2" />
              Refresh Data
            </Button>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-4 py-6 space-y-6">
        {quotesError && (
          <Alert variant="destructive" className="bg-destructive/10 border-destructive/30">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>{quotesError}</AlertDescription>
          </Alert>
        )}

        {/* Stock Cards Grid */}
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

        {/* Main Content Grid */}
        <div className="grid lg:grid-cols-2 gap-6">
          {/* Left Column: Chart + Company Info */}
          <div className="space-y-6">
            <StockChart
              data={timeseries || null}
              isLoading={timeseriesLoading}
              symbol={selectedSymbol || undefined}
              interval={chartInterval}
              onIntervalChange={setChartInterval}
            />
            <CompanyInfo overview={overview || null} isLoading={overviewLoading} />
          </div>

          {/* Right Column: Chat Interface */}
          <div>
            <ChatInterface
              selectedSymbol={selectedSymbol}
              onAnalysisComplete={handleAnalysisComplete}
              onAnalysisStart={handleAnalysisStart}
            />
          </div>
        </div>

        {/* Agent Analysis Panel */}
        <section>
          <AgentPanel result={analysisResult} isLoading={isAnalyzing} />
        </section>
      </main>

      <footer className="border-t border-border bg-card/30 mt-12">
        <div className="container mx-auto px-4 py-6">
          <p className="text-center text-sm text-muted-foreground">
            MarketProphet uses AI-powered multi-agent analysis. Data provided by Finnhub. AI analysis by Kimi-K2.5.
            <br />
            <span className="text-xs">This is not financial advice. Always do your own research before investing.</span>
          </p>
        </div>
      </footer>
    </div>
  )
}
