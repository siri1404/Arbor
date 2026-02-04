import { NextRequest, NextResponse } from 'next/server'
import type { CompanyOverview } from '@/lib/types'
import { getFromCache, setInCache } from '@/lib/cache'

const FINNHUB_API_KEY = process.env.FINNHUB_API_KEY
const CACHE_TTL = 3600 // 1 hour cache for company overview (rarely changes)

export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams
  const symbol = searchParams.get('symbol')?.toUpperCase()

  if (!symbol) {
    return NextResponse.json({ error: 'Symbol is required' }, { status: 400 })
  }

  // Check cache first
  const cacheKey = `overview:${symbol}`
  const cached = getFromCache<CompanyOverview>(cacheKey)
  if (cached) {
    return NextResponse.json(cached)
  }

  if (!FINNHUB_API_KEY) {
    return NextResponse.json({ error: 'Finnhub API key not configured' }, { status: 500 })
  }

  try {
    // Fetch both company profile and basic financials in parallel
    const [profileRes, metricsRes] = await Promise.all([
      fetch(`https://finnhub.io/api/v1/stock/profile2?symbol=${symbol}&token=${FINNHUB_API_KEY}`),
      fetch(`https://finnhub.io/api/v1/stock/metric?symbol=${symbol}&metric=all&token=${FINNHUB_API_KEY}`)
    ])

    // Handle rate limiting
    if (profileRes.status === 429 || metricsRes.status === 429) {
      return NextResponse.json({ error: 'Rate limit exceeded. Please try again later.' }, { status: 429 })
    }

    const profileText = await profileRes.text()
    const metricsText = await metricsRes.text()
    
    let profile, metrics
    try {
      profile = JSON.parse(profileText)
      metrics = JSON.parse(metricsText)
    } catch {
      console.error('Finnhub API error: Invalid JSON response')
      return NextResponse.json({ error: 'Rate limit exceeded or invalid response' }, { status: 429 })
    }

    if (!profile || !profile.name) {
      return NextResponse.json({ error: 'No company data found' }, { status: 404 })
    }

    const metric = metrics.metric || {}

    const overview: CompanyOverview = {
      symbol: profile.ticker || symbol,
      name: profile.name,
      description: `${profile.name} is a company in the ${profile.finnhubIndustry || 'N/A'} industry, headquartered in ${profile.country || 'N/A'}.`,
      sector: profile.finnhubIndustry || 'N/A',
      industry: profile.finnhubIndustry || 'N/A',
      marketCap: profile.marketCapitalization ? profile.marketCapitalization * 1000000 : 0,
      peRatio: metric.peBasicExclExtraTTM || metric.peNormalizedAnnual || 0,
      eps: metric.epsBasicExclExtraItemsTTM || metric.epsGrowth3Y || 0,
      dividendYield: metric.dividendYieldIndicatedAnnual || 0,
      fiftyTwoWeekHigh: metric['52WeekHigh'] || 0,
      fiftyTwoWeekLow: metric['52WeekLow'] || 0,
      beta: metric.beta || 0,
    }

    // Cache the result
    setInCache(cacheKey, overview, CACHE_TTL)

    return NextResponse.json(overview)
  } catch (error) {
    console.error('Finnhub API error:', error)
    return NextResponse.json({ error: 'Failed to fetch company data' }, { status: 500 })
  }
}
