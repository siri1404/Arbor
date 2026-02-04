export interface StockQuote {
  symbol: string
  price: number
  change: number
  changePercent: number
  volume: number
  high: number
  low: number
  open: number
  previousClose: number
  timestamp: string
}

export interface StockTimeSeries {
  symbol: string
  data: TimeSeriesDataPoint[]
}

export interface TimeSeriesDataPoint {
  date: string
  open: number
  high: number
  low: number
  close: number
  volume: number
}

export interface CompanyOverview {
  symbol: string
  name: string
  description: string
  sector: string
  industry: string
  marketCap: number
  peRatio: number
  eps: number
  dividendYield: number
  fiftyTwoWeekHigh: number
  fiftyTwoWeekLow: number
  beta: number
}

export interface AgentType {
  id: string
  name: string
  description: string
  icon: string
  status: 'idle' | 'running' | 'completed' | 'error'
}

export interface AgentAnalysis {
  agentId: string
  agentName: string
  analysis: string
  confidence: number
  keyPoints: string[]
  recommendation: 'buy' | 'sell' | 'hold' | 'neutral'
  timestamp: string
}

export interface OrchestratorResult {
  symbol: string
  query: string
  agents: AgentAnalysis[]
  synthesis: string
  overallRecommendation: 'buy' | 'sell' | 'hold' | 'neutral'
  confidenceScore: number
  timestamp: string
}

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: string
  agentResults?: OrchestratorResult
}

export const DEFAULT_STOCKS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA'] as const
