import { NextRequest, NextResponse } from 'next/server'
import type { StockQuote } from '@/lib/types'
import { getFromCache, setInCache } from '@/lib/cache'

const FINNHUB_API_KEY = process.env.FINNHUB_API_KEY
const CACHE_TTL = 300 // 5 minute cache to reduce API calls and avoid rate limiting

export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams
  const symbol = searchParams.get('symbol')?.toUpperCase()

  if (!symbol) {
    return NextResponse.json({ error: 'Symbol is required' }, { status: 400 })
  }

  // Check cache first
  const cacheKey = `quote:${symbol}`
  const cached = getFromCache<StockQuote>(cacheKey)
  if (cached) {
    const response = NextResponse.json(cached)
    response.headers.set('x-cache-status', 'HIT')
    return response
  }

  if (!FINNHUB_API_KEY) {
    return NextResponse.json({ error: 'Finnhub API key not configured' }, { status: 500 })
  }

  try {
    const url = `https://finnhub.io/api/v1/quote?symbol=${symbol}&token=${FINNHUB_API_KEY}`
    const response = await fetch(url)
    
    // Handle rate limiting
    if (response.status === 429) {
      return NextResponse.json({ error: 'Rate limit exceeded. Please try again later.' }, { status: 429 })
    }
    
    const text = await response.text()
    let data
    try {
      data = JSON.parse(text)
    } catch {
      console.error('Finnhub API error:', text)
      return NextResponse.json({ error: 'Rate limit exceeded or invalid response' }, { status: 429 })
    }

    // Finnhub returns { c, d, dp, h, l, o, pc, t } 
    // c = current price, d = change, dp = percent change, h = high, l = low, o = open, pc = previous close, t = timestamp
    if (!data || data.c === 0 || data.c === null) {
      if (data.error) {
        return NextResponse.json({ error: data.error }, { status: 429 })
      }
      return NextResponse.json({ error: 'No data found for symbol' }, { status: 404 })
    }

    const stockQuote: StockQuote = {
      symbol: symbol,
      price: data.c,
      change: data.d || 0,
      changePercent: data.dp || 0,
      volume: 0,
      high: data.h,
      low: data.l,
      open: data.o,
      previousClose: data.pc,
      timestamp: new Date(data.t * 1000).toISOString().split('T')[0],
    }

    // Cache the result
    setInCache(cacheKey, stockQuote, CACHE_TTL)

    return NextResponse.json(stockQuote)
  } catch (error) {
    console.error('Finnhub API error:', error)
    return NextResponse.json({ error: 'Failed to fetch stock data' }, { status: 500 })
  }
}
