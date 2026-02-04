'use client'

import React from "react"

import { useState, useRef, useEffect } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import type { ChatMessage, OrchestratorResult } from '@/lib/types'
import { Send, Loader2, MessageSquare, Bot, User } from 'lucide-react'

interface ChatInterfaceProps {
  selectedSymbol: string | null
  onAnalysisComplete: (result: OrchestratorResult) => void
  onAnalysisStart: () => void
}

export function ChatInterface({ selectedSymbol, onAnalysisComplete, onAnalysisStart }: ChatInterfaceProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const extractSymbol = (query: string): string | null => {
    // Known stock symbols to recognize
    const knownSymbols = ['AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'META', 'NVDA', 'TSLA', 'AMD', 'INTC', 'NFLX', 'DIS', 'BA', 'JPM', 'V', 'MA', 'WMT', 'PG', 'JNJ', 'UNH', 'HD', 'CRM', 'ORCL', 'ADBE', 'PYPL', 'COST', 'PEP', 'KO', 'MRK', 'ABT', 'TMO', 'NKE', 'MCD', 'LLY', 'ABBV', 'XOM', 'CVX', 'VZ', 'T', 'CSCO', 'QCOM', 'TXN', 'IBM', 'GS', 'MS', 'AXP', 'BLK', 'C', 'BAC', 'WFC']
    // Common English words to exclude (uppercase)
    const excludeWords = ['I', 'A', 'IS', 'IT', 'TO', 'BE', 'OR', 'IF', 'IN', 'ON', 'AT', 'BY', 'AN', 'AS', 'DO', 'GO', 'HE', 'ME', 'MY', 'NO', 'OF', 'OK', 'SO', 'UP', 'US', 'WE', 'AM', 'ARE', 'THE', 'AND', 'FOR', 'NOT', 'YOU', 'ALL', 'CAN', 'HAD', 'HER', 'WAS', 'ONE', 'OUR', 'OUT', 'HAS', 'HIS', 'HOW', 'ITS', 'LET', 'MAY', 'NEW', 'NOW', 'OLD', 'SEE', 'WAY', 'WHO', 'BOY', 'DID', 'GET', 'HIM', 'OWN', 'SAY', 'SHE', 'TOO', 'USE', 'BUY', 'SELL', 'HOLD', 'WHAT', 'WHEN', 'WITH', 'GOOD', 'RISK', 'HIGH', 'LOW']
    
    const upperQuery = query.toUpperCase()
    
    // First, check for known symbols
    for (const symbol of knownSymbols) {
      if (upperQuery.includes(symbol)) {
        return symbol
      }
    }
    
    // Then try to find any uppercase word that looks like a ticker (2-5 chars, all caps)
    const words = query.split(/\s+/)
    for (const word of words) {
      const clean = word.replace(/[^A-Za-z]/g, '').toUpperCase()
      if (clean.length >= 2 && clean.length <= 5 && !excludeWords.includes(clean)) {
        // Only accept if it's actually written in caps in the original or is a known symbol
        if (word === word.toUpperCase() || knownSymbols.includes(clean)) {
          return clean
        }
      }
    }
    
    return selectedSymbol
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!input.trim() || isLoading) return

    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      role: 'user',
      content: input,
      timestamp: new Date().toISOString(),
    }

    setMessages((prev) => [...prev, userMessage])
    setInput('')
    setIsLoading(true)
    onAnalysisStart()

    const symbol = extractSymbol(input.toUpperCase()) || selectedSymbol

    if (!symbol) {
      const errorMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: 'Please select a stock from the dashboard or include a stock symbol (e.g., AAPL, MSFT) in your question.',
        timestamp: new Date().toISOString(),
      }
      setMessages((prev) => [...prev, errorMessage])
      setIsLoading(false)
      return
    }

    try {
      const response = await fetch('/api/ai/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ symbol, query: input }),
      })

      if (!response.ok) {
        const error = await response.json()
        throw new Error(error.error || 'Analysis failed')
      }

      const result: OrchestratorResult = await response.json()

      const assistantMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: result.synthesis,
        timestamp: new Date().toISOString(),
        agentResults: result,
      }

      setMessages((prev) => [...prev, assistantMessage])
      onAnalysisComplete(result)
    } catch (error) {
      const errorMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: `Analysis failed: ${error instanceof Error ? error.message : 'Unknown error'}. Please check your API keys are configured correctly.`,
        timestamp: new Date().toISOString(),
      }
      setMessages((prev) => [...prev, errorMessage])
    } finally {
      setIsLoading(false)
    }
  }

  const recommendationColors: Record<string, string> = {
    buy: 'bg-success/20 text-success border-success/30',
    sell: 'bg-destructive/20 text-destructive border-destructive/30',
    hold: 'bg-warning/20 text-warning border-warning/30',
    neutral: 'bg-muted text-muted-foreground border-muted',
  }

  return (
    <Card className="h-full flex flex-col">
      <CardHeader className="pb-2 border-b border-border">
        <CardTitle className="flex items-center gap-2 text-base">
          <MessageSquare className="h-5 w-5 text-primary" />
          Natural Language Query
          {selectedSymbol && (
            <Badge variant="outline" className="ml-auto font-mono">
              {selectedSymbol}
            </Badge>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent className="flex-1 flex flex-col p-0">
        <div className="flex-1 overflow-y-auto p-4 space-y-4 max-h-[400px]">
          {messages.length === 0 && (
            <div className="text-center py-8">
              <Bot className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
              <h3 className="font-medium text-foreground mb-2">Ask ARBOR</h3>
              <p className="text-sm text-muted-foreground max-w-md mx-auto">
                Ask any question about stocks and our AI agents will analyze it from multiple perspectives.
              </p>
              <div className="mt-4 flex flex-wrap justify-center gap-2">
                {[
                  'Should I buy AAPL?',
                  'Is NVDA overvalued?',
                  'What are the risks for TSLA?',
                ].map((suggestion) => (
                  <Button
                    key={suggestion}
                    variant="outline"
                    size="sm"
                    className="text-xs bg-transparent"
                    onClick={() => setInput(suggestion)}
                  >
                    {suggestion}
                  </Button>
                ))}
              </div>
            </div>
          )}
          {messages.map((message) => (
            <div
              key={message.id}
              className={`flex gap-3 ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              {message.role === 'assistant' && (
                <div className="flex-shrink-0 w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center">
                  <Bot className="h-4 w-4 text-primary" />
                </div>
              )}
              <div
                className={`max-w-[80%] rounded-lg p-3 ${
                  message.role === 'user'
                    ? 'bg-primary text-primary-foreground'
                    : 'bg-secondary'
                }`}
              >
                <p className="text-sm">{message.content}</p>
                {message.agentResults && (
                  <div className="mt-2 pt-2 border-t border-border/50">
                    <div className="flex items-center gap-2">
                      <Badge className={recommendationColors[message.agentResults.overallRecommendation]}>
                        {message.agentResults.overallRecommendation.toUpperCase()}
                      </Badge>
                      <span className="text-xs text-muted-foreground">
                        {message.agentResults.confidenceScore}% confidence
                      </span>
                    </div>
                  </div>
                )}
              </div>
              {message.role === 'user' && (
                <div className="flex-shrink-0 w-8 h-8 rounded-full bg-primary flex items-center justify-center">
                  <User className="h-4 w-4 text-primary-foreground" />
                </div>
              )}
            </div>
          ))}
          {isLoading && (
            <div className="flex gap-3 justify-start">
              <div className="flex-shrink-0 w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center">
                <Bot className="h-4 w-4 text-primary" />
              </div>
              <div className="bg-secondary rounded-lg p-3">
                <div className="flex items-center gap-2">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  <span className="text-sm text-muted-foreground">
                    Analyzing with AI agents...
                  </span>
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
        <form onSubmit={handleSubmit} className="p-4 border-t border-border">
          <div className="flex gap-2">
            <Input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder={selectedSymbol ? `Ask about ${selectedSymbol}...` : 'Ask about any stock (e.g., Should I buy AAPL?)'}
              disabled={isLoading}
              className="flex-1"
            />
            <Button type="submit" disabled={isLoading || !input.trim()}>
              {isLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
            </Button>
          </div>
        </form>
      </CardContent>
    </Card>
  )
}
