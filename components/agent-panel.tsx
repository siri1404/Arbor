'use client'

import React from "react"

import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import type { AgentAnalysis, OrchestratorResult } from '@/lib/types'
import { Search, LineChart, MessageSquare, Shield, Brain, CheckCircle2, Loader2 } from 'lucide-react'

interface AgentPanelProps {
  result: OrchestratorResult | null
  isLoading?: boolean
  runningAgents?: string[]
}

const agentIcons: Record<string, React.ReactNode> = {
  research: <Search className="h-4 w-4" />,
  technical: <LineChart className="h-4 w-4" />,
  sentiment: <MessageSquare className="h-4 w-4" />,
  risk: <Shield className="h-4 w-4" />,
}

const recommendationColors: Record<string, string> = {
  buy: 'bg-success/20 text-success border-success/30',
  sell: 'bg-destructive/20 text-destructive border-destructive/30',
  hold: 'bg-warning/20 text-warning border-warning/30',
  neutral: 'bg-muted text-muted-foreground border-muted',
}

function AgentCard({ agent, isRunning }: { agent: AgentAnalysis; isRunning?: boolean }) {
  return (
    <Card className="bg-secondary/30">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="p-1.5 rounded-md bg-primary/10 text-primary">
              {agentIcons[agent.agentId] || <Brain className="h-4 w-4" />}
            </div>
            <CardTitle className="text-sm">{agent.agentName}</CardTitle>
          </div>
          {isRunning ? (
            <Loader2 className="h-4 w-4 animate-spin text-primary" />
          ) : (
            <CheckCircle2 className="h-4 w-4 text-success" />
          )}
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="flex items-center justify-between">
          <Badge className={recommendationColors[agent.recommendation]}>
            {agent.recommendation.toUpperCase()}
          </Badge>
          <span className="text-xs text-muted-foreground">
            {agent.confidence}% confidence
          </span>
        </div>
        <Progress value={agent.confidence} className="h-1.5" />
        <div className="space-y-1">
          {agent.keyPoints.slice(0, 3).map((point, i) => (
            <p key={i} className="text-xs text-muted-foreground line-clamp-1">
              • {point}
            </p>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}

export function AgentPanel({ result, isLoading, runningAgents = [] }: AgentPanelProps) {
  if (isLoading && !result) {
    return (
      <Card className="h-full">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Brain className="h-5 w-5 text-primary" />
            AI Agent Analysis
          </CardTitle>
          <CardDescription>Processing your query with specialized agents...</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 md:grid-cols-2">
            {['research', 'technical', 'sentiment', 'risk'].map((agentId) => (
              <Card key={agentId} className="bg-secondary/30">
                <CardHeader className="pb-2">
                  <div className="flex items-center gap-2">
                    <div className="p-1.5 rounded-md bg-primary/10 text-primary">
                      {agentIcons[agentId]}
                    </div>
                    <div className="h-4 w-24 animate-pulse rounded bg-muted" />
                  </div>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="h-6 w-16 animate-pulse rounded bg-muted" />
                  <div className="h-1.5 w-full animate-pulse rounded bg-muted" />
                  <div className="space-y-1">
                    <div className="h-3 w-full animate-pulse rounded bg-muted" />
                    <div className="h-3 w-3/4 animate-pulse rounded bg-muted" />
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </CardContent>
      </Card>
    )
  }

  if (!result) {
    return (
      <Card className="h-full">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Brain className="h-5 w-5 text-primary" />
            AI Agent Analysis
          </CardTitle>
          <CardDescription>
            Ask a question about any stock to activate the multi-agent analysis system
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
            {[
              { id: 'research', name: 'Research Agent', desc: 'Fundamental analysis' },
              { id: 'technical', name: 'Technical Agent', desc: 'Chart patterns & trends' },
              { id: 'sentiment', name: 'Sentiment Agent', desc: 'Market psychology' },
              { id: 'risk', name: 'Risk Agent', desc: 'Risk assessment' },
            ].map((agent) => (
              <Card key={agent.id} className="bg-secondary/30 border-dashed">
                <CardHeader className="pb-2">
                  <div className="flex items-center gap-2">
                    <div className="p-1.5 rounded-md bg-muted text-muted-foreground">
                      {agentIcons[agent.id]}
                    </div>
                    <CardTitle className="text-sm text-muted-foreground">{agent.name}</CardTitle>
                  </div>
                </CardHeader>
                <CardContent>
                  <p className="text-xs text-muted-foreground">{agent.desc}</p>
                </CardContent>
              </Card>
            ))}
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card className="h-full">
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <Brain className="h-5 w-5 text-primary" />
              Analysis for {result.symbol}
            </CardTitle>
            <CardDescription>Query: &quot;{result.query}&quot;</CardDescription>
          </div>
          <div className="text-right">
            <Badge className={`text-sm ${recommendationColors[result.overallRecommendation]}`}>
              {result.overallRecommendation.toUpperCase()}
            </Badge>
            <p className="text-xs text-muted-foreground mt-1">
              {result.confidenceScore}% confidence
            </p>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          {result.agents.map((agent) => (
            <AgentCard key={agent.agentId} agent={agent} isRunning={runningAgents.includes(agent.agentId)} />
          ))}
        </div>
        <Card className="bg-primary/5 border-primary/20">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm flex items-center gap-2">
              <Brain className="h-4 w-4 text-primary" />
              Synthesis
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-foreground/90">{result.synthesis}</p>
          </CardContent>
        </Card>
      </CardContent>
    </Card>
  )
}
