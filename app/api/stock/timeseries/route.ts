import { NextRequest, NextResponse } from 'next/server'
import type { StockTimeSeries, TimeSeriesDataPoint } from '@/lib/types'
import { getFromCache, setInCache } from '@/lib/cache'

const CACHE_TTL = 300 // 5 minutes cache for time series

export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams
  const symbol = searchParams.get('symbol')?.toUpperCase()
  const interval = searchParams.get('interval') || 'daily'

  if (!symbol) {
    return NextResponse.json({ error: 'Symbol is required' }, { status: 400 })
  }

  // Check cache first
  const cacheKey = `timeseries:${symbol}:${interval}`
  const cached = getFromCache<StockTimeSeries>(cacheKey)
  if (cached) {
    return NextResponse.json(cached)
  }

  try {
    // Yahoo Finance parameters
    let yahooInterval: string
    let range: string

    switch (interval) {
      case 'intraday':
        yahooInterval = '5m'
        range = '1d'
        break
      case 'weekly':
        yahooInterval = '1wk'
        range = '1y'
        break
      case 'monthly':
        yahooInterval = '1mo'
        range = '5y'
        break
      case 'daily':
      default:
        yahooInterval = '1d'
        range = '3mo'
        break
    }

    // Yahoo Finance v8 API (free, no key required)
    const url = `https://query1.finance.yahoo.com/v8/finance/chart/${symbol}?interval=${yahooInterval}&range=${range}`
    const response = await fetch(url, {
      headers: {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
      }
    })
    
    if (!response.ok) {
      console.error('Yahoo Finance API error:', response.status, response.statusText)
      return NextResponse.json({ error: 'Failed to fetch data from Yahoo Finance' }, { status: response.status })
    }

    const text = await response.text()
    let data
    try {
      data = JSON.parse(text)
    } catch {
      console.error('Yahoo Finance invalid JSON:', text.substring(0, 200))
      return NextResponse.json({ error: 'Invalid response from Yahoo Finance' }, { status: 500 })
    }
    const result = data.chart?.result?.[0]

    if (!result || !result.timestamp) {
      return NextResponse.json({ error: 'No data found for symbol' }, { status: 404 })
    }

    const timestamps = result.timestamp
    const quote = result.indicators?.quote?.[0]

    if (!quote) {
      return NextResponse.json({ error: 'No quote data found' }, { status: 404 })
    }

    const dataPoints: TimeSeriesDataPoint[] = timestamps
      .map((timestamp: number, i: number) => {
        if (quote.close[i] === null) return null
        return {
          date: new Date(timestamp * 1000).toISOString().split('T')[0],
          open: quote.open[i] || 0,
          high: quote.high[i] || 0,
          low: quote.low[i] || 0,
          close: quote.close[i] || 0,
          volume: quote.volume[i] || 0,
        }
      })
      .filter((point: TimeSeriesDataPoint | null): point is TimeSeriesDataPoint => point !== null)

    const timeSeriesResult: StockTimeSeries = {
      symbol,
      data: dataPoints,
    }

    // Cache the result
    setInCache(cacheKey, timeSeriesResult, CACHE_TTL)

    return NextResponse.json(timeSeriesResult)
  } catch (error) {
    console.error('Yahoo Finance API error:', error)
    return NextResponse.json({ error: 'Failed to fetch time series data' }, { status: 500 })
  }
}
