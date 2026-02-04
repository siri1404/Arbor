import { NextRequest, NextResponse } from 'next/server'
import type { OrchestratorResult, AgentAnalysis, StockQuote, StockTimeSeries, CompanyOverview } from '@/lib/types'

const OPENROUTER_API_KEY = process.env.OPENROUTER_API_KEY
const FINNHUB_API_KEY = process.env.FINNHUB_API_KEY
const ALPHA_VANTAGE_API_KEY = process.env.ALPHA_VANTAGE_API_KEY

interface AgentConfig {
  id: string
  name: string
  systemPrompt: string
}

const agents: AgentConfig[] = [
  {
    id: 'research',
    name: 'Research Agent',
    systemPrompt: `You are a fundamental research analyst. Analyze the company's financial health, competitive position, and growth prospects. Focus on:
- Revenue and earnings trends
- Market position and competitive advantages
- Management quality and strategy
- Industry dynamics and tailwinds/headwinds
Provide a clear buy/sell/hold recommendation with confidence level (0-100).`,
  },
  {
    id: 'technical',
    name: 'Technical Agent',
    systemPrompt: `You are a technical analysis expert. Analyze price patterns, trends, and momentum indicators. Focus on:
- Price trends and moving averages
- Support and resistance levels
- Volume patterns
- Momentum indicators (RSI, MACD concepts)
Provide a clear buy/sell/hold recommendation with confidence level (0-100).`,
  },
  {
    id: 'sentiment',
    name: 'Sentiment Agent',
    systemPrompt: `You are a market sentiment analyst. Analyze the overall market sentiment and investor psychology. Focus on:
- Recent price momentum and what it indicates about sentiment
- Sector rotation and market trends
- Institutional vs retail sentiment indicators
- Fear/greed indicators based on volatility
Provide a clear buy/sell/hold recommendation with confidence level (0-100).`,
  },
  {
    id: 'risk',
    name: 'Risk Agent',
    systemPrompt: `You are a risk management specialist. Analyze potential risks and downside scenarios. Focus on:
- Volatility and beta analysis
- Downside risk scenarios
- Valuation risk (overvalued/undervalued)
- Macro and sector-specific risks
Provide a clear buy/sell/hold recommendation with confidence level (0-100).`,
  },
]

async function fetchStockData(symbol: string, baseUrl: string) {
  const [quoteRes, timeseriesRes, overviewRes] = await Promise.all([
    fetch(`${baseUrl}/api/stock/quote?symbol=${symbol}`),
    fetch(`${baseUrl}/api/stock/timeseries?symbol=${symbol}`),
    fetch(`${baseUrl}/api/stock/overview?symbol=${symbol}`),
  ])

  const quote: StockQuote | null = quoteRes.ok ? await quoteRes.json() : null
  const timeseries: StockTimeSeries | null = timeseriesRes.ok ? await timeseriesRes.json() : null
  const overview: CompanyOverview | null = overviewRes.ok ? await overviewRes.json() : null

  return { quote, timeseries, overview }
}

async function runAgent(
  agent: AgentConfig,
  symbol: string,
  query: string,
  stockData: { quote: StockQuote | null; timeseries: StockTimeSeries | null; overview: CompanyOverview | null }
): Promise<AgentAnalysis> {
  const { quote, timeseries, overview } = stockData

  const dataContext = `
Stock: ${symbol}
${quote ? `Current Price: $${quote.price.toFixed(2)}
Change: ${quote.change > 0 ? '+' : ''}${quote.change.toFixed(2)} (${quote.changePercent.toFixed(2)}%)
Volume: ${quote.volume.toLocaleString()}
Day Range: $${quote.low.toFixed(2)} - $${quote.high.toFixed(2)}
Previous Close: $${quote.previousClose.toFixed(2)}` : 'Quote data unavailable'}

${overview ? `Company: ${overview.name}
Sector: ${overview.sector}
Industry: ${overview.industry}
Market Cap: $${(overview.marketCap / 1e9).toFixed(2)}B
P/E Ratio: ${overview.peRatio.toFixed(2)}
EPS: $${overview.eps.toFixed(2)}
52-Week Range: $${overview.fiftyTwoWeekLow.toFixed(2)} - $${overview.fiftyTwoWeekHigh.toFixed(2)}
Beta: ${overview.beta.toFixed(2)}
Description: ${overview.description?.slice(0, 500) || 'N/A'}` : 'Company overview unavailable'}

${timeseries ? `Recent Price History (last 10 days):
${timeseries.data.slice(-10).map(d => `${d.date}: Open $${d.open.toFixed(2)}, Close $${d.close.toFixed(2)}, Volume ${d.volume.toLocaleString()}`).join('\n')}` : 'Time series data unavailable'}
`

  const response = await fetch('https://openrouter.ai/api/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${OPENROUTER_API_KEY}`,
      'HTTP-Referer': 'https://arbor-ai.vercel.app',
      'X-Title': 'ARBOR',
    },
    body: JSON.stringify({
      model: 'moonshotai/kimi-k2.5',
      messages: [
        {
          role: 'system',
          content: `${agent.systemPrompt}

IMPORTANT: You must respond in valid JSON format with this exact structure:
{
  "analysis": "Your detailed analysis text here",
  "confidence": 75,
  "keyPoints": ["Point 1", "Point 2", "Point 3"],
  "recommendation": "buy" | "sell" | "hold" | "neutral"
}`,
        },
        {
          role: 'user',
          content: `User query: "${query}"

Stock data:
${dataContext}

Provide your analysis as the ${agent.name}. Remember to respond in JSON format.`,
        },
      ],
      temperature: 0.7,
      max_tokens: 1500,
    }),
  })

  if (!response.ok) {
    const errorText = await response.text()
    console.error(`Agent ${agent.id} error:`, errorText)
    throw new Error(`Agent ${agent.id} failed: ${response.status}`)
  }

  const data = await response.json()
  const content = data.choices?.[0]?.message?.content || ''

  let parsed: { analysis: string; confidence: number; keyPoints: string[]; recommendation: string }
  try {
    const jsonMatch = content.match(/\{[\s\S]*\}/)
    if (jsonMatch) {
      parsed = JSON.parse(jsonMatch[0])
    } else {
      throw new Error('No JSON found')
    }
  } catch {
    parsed = {
      analysis: content,
      confidence: 50,
      keyPoints: ['Analysis completed'],
      recommendation: 'neutral',
    }
  }

  return {
    agentId: agent.id,
    agentName: agent.name,
    analysis: parsed.analysis,
    confidence: Math.min(100, Math.max(0, parsed.confidence)),
    keyPoints: Array.isArray(parsed.keyPoints) ? parsed.keyPoints.slice(0, 5) : ['Analysis completed'],
    recommendation: ['buy', 'sell', 'hold', 'neutral'].includes(parsed.recommendation)
      ? (parsed.recommendation as 'buy' | 'sell' | 'hold' | 'neutral')
      : 'neutral',
    timestamp: new Date().toISOString(),
  }
}

async function synthesizeResults(
  symbol: string,
  query: string,
  agentResults: AgentAnalysis[]
): Promise<{ synthesis: string; overallRecommendation: 'buy' | 'sell' | 'hold' | 'neutral'; confidenceScore: number }> {
  const agentSummary = agentResults
    .map(
      (a) =>
        `${a.agentName}: ${a.recommendation.toUpperCase()} (${a.confidence}% confidence)
Key points: ${a.keyPoints.join('; ')}`
    )
    .join('\n\n')

  const response = await fetch('https://openrouter.ai/api/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${OPENROUTER_API_KEY}`,
      'HTTP-Referer': 'https://arbor-ai.vercel.app',
      'X-Title': 'ARBOR',
    },
    body: JSON.stringify({
      model: 'moonshotai/kimi-k2.5',
      messages: [
        {
          role: 'system',
          content: `You are the Chief Investment Strategist synthesizing analysis from multiple AI agents. Provide a final recommendation that weighs all perspectives.

IMPORTANT: Respond in valid JSON format:
{
  "synthesis": "Your comprehensive synthesis explaining the collective view and final recommendation",
  "overallRecommendation": "buy" | "sell" | "hold" | "neutral",
  "confidenceScore": 75
}`,
        },
        {
          role: 'user',
          content: `Original query about ${symbol}: "${query}"

Agent analyses:
${agentSummary}

Synthesize these analyses into a final recommendation. Consider areas of agreement and disagreement. Respond in JSON format.`,
        },
      ],
      temperature: 0.5,
      max_tokens: 1000,
    }),
  })

  if (!response.ok) {
    throw new Error('Synthesis failed')
  }

  const data = await response.json()
  const content = data.choices?.[0]?.message?.content || ''

  try {
    const jsonMatch = content.match(/\{[\s\S]*\}/)
    if (jsonMatch) {
      const parsed = JSON.parse(jsonMatch[0])
      return {
        synthesis: parsed.synthesis,
        overallRecommendation: ['buy', 'sell', 'hold', 'neutral'].includes(parsed.overallRecommendation)
          ? parsed.overallRecommendation
          : 'neutral',
        confidenceScore: Math.min(100, Math.max(0, parsed.confidenceScore || 50)),
      }
    }
  } catch {
    // Fall through to default
  }

  const avgConfidence = agentResults.reduce((sum, a) => sum + a.confidence, 0) / agentResults.length
  const recommendations = agentResults.map((a) => a.recommendation)
  const mostCommon = recommendations
    .sort((a, b) => recommendations.filter((v) => v === b).length - recommendations.filter((v) => v === a).length)[0]

  return {
    synthesis: content || 'Analysis complete. Please review individual agent recommendations.',
    overallRecommendation: mostCommon,
    confidenceScore: Math.round(avgConfidence),
  }
}

export async function POST(request: NextRequest) {
  if (!OPENROUTER_API_KEY) {
    return NextResponse.json({ error: 'OpenRouter API key not configured' }, { status: 500 })
  }

  if (!FINNHUB_API_KEY) {
    return NextResponse.json({ error: 'Finnhub API key not configured' }, { status: 500 })
  }

  try {
    const body = await request.json()
    const { symbol, query } = body

    if (!symbol || !query) {
      return NextResponse.json({ error: 'Symbol and query are required' }, { status: 400 })
    }

    const baseUrl = request.nextUrl.origin
    const stockData = await fetchStockData(symbol.toUpperCase(), baseUrl)

    const agentResults = await Promise.all(
      agents.map((agent) => runAgent(agent, symbol.toUpperCase(), query, stockData))
    )

    const { synthesis, overallRecommendation, confidenceScore } = await synthesizeResults(
      symbol.toUpperCase(),
      query,
      agentResults
    )

    const result: OrchestratorResult = {
      symbol: symbol.toUpperCase(),
      query,
      agents: agentResults,
      synthesis,
      overallRecommendation,
      confidenceScore,
      timestamp: new Date().toISOString(),
    }

    return NextResponse.json(result)
  } catch (error) {
    console.error('Analysis error:', error)
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Analysis failed' },
      { status: 500 }
    )
  }
}
