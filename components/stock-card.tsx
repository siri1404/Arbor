'use client'

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import type { StockQuote } from '@/lib/types'
import { TrendingUp, TrendingDown, Minus, Activity } from 'lucide-react'
import { cn } from '@/lib/utils'

interface StockCardProps {
  quote: StockQuote | null
  isLoading?: boolean
  isSelected?: boolean
  onClick?: () => void
}

export function StockCard({ quote, isLoading, isSelected, onClick }: StockCardProps) {
  if (isLoading) {
    return (
      <Card className={cn(
        "cursor-pointer transition-all duration-300 hover:border-primary/50 relative overflow-hidden",
        isSelected && "border-primary"
      )}>
        <div className="absolute inset-0 bg-gradient-to-r from-transparent via-muted/50 to-transparent animate-shimmer" />
        <CardHeader className="pb-2">
          <div className="h-6 w-16 animate-pulse rounded bg-muted" />
        </CardHeader>
        <CardContent>
          <div className="h-8 w-24 animate-pulse rounded bg-muted mb-2" />
          <div className="h-4 w-20 animate-pulse rounded bg-muted" />
        </CardContent>
      </Card>
    )
  }

  if (!quote) {
    return (
      <Card className="cursor-pointer opacity-50 transition-all duration-300">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm text-muted-foreground">No data</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-xs text-muted-foreground">Unable to load</p>
        </CardContent>
      </Card>
    )
  }

  const isPositive = quote.change > 0
  const isNeutral = quote.change === 0

  return (
    <Card
      className={cn(
        "cursor-pointer transition-all duration-300 hover:scale-[1.02] hover:shadow-lg relative overflow-hidden group",
        "hover:border-primary/50 hover:shadow-primary/5",
        isSelected && "border-primary ring-2 ring-primary/20 shadow-lg shadow-primary/10",
        isPositive && "hover:shadow-success/10",
        !isPositive && !isNeutral && "hover:shadow-destructive/10"
      )}
      onClick={onClick}
    >
      {/* Animated background gradient */}
      <div className={cn(
        "absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-500",
        isPositive ? "bg-gradient-to-br from-success/5 to-transparent" : 
        isNeutral ? "bg-gradient-to-br from-muted/5 to-transparent" :
        "bg-gradient-to-br from-destructive/5 to-transparent"
      )} />
      
      <CardHeader className="pb-2 relative">
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg font-bold tracking-tight">{quote.symbol}</CardTitle>
          <Badge
            variant={isNeutral ? 'secondary' : 'outline'}
            className={cn(
              "transition-all duration-300 font-semibold",
              isNeutral
                ? ''
                : isPositive
                  ? 'border-success text-success bg-success/10'
                  : 'border-destructive text-destructive bg-destructive/10'
            )}
          >
            {isNeutral ? (
              <Minus className="mr-1 h-3 w-3" />
            ) : isPositive ? (
              <TrendingUp className="mr-1 h-3 w-3 animate-pulse" />
            ) : (
              <TrendingDown className="mr-1 h-3 w-3 animate-pulse" />
            )}
            {isPositive ? '+' : ''}
            {quote.changePercent.toFixed(2)}%
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="relative">
        <p className="text-2xl font-bold font-mono tracking-tight">${quote.price.toFixed(2)}</p>
        <p className={cn(
          "text-sm font-mono transition-colors duration-300",
          isNeutral ? 'text-muted-foreground' : isPositive ? 'text-success' : 'text-destructive'
        )}>
          {isPositive ? '+' : ''}
          {quote.change.toFixed(2)}
        </p>
        <div className="mt-3 pt-3 border-t border-border/50 flex flex-wrap gap-2 text-xs text-muted-foreground">
          <span className="flex items-center gap-1">
            <Activity className="h-3 w-3" />
            {(quote.volume / 1e6).toFixed(1)}M
          </span>
          <span className="text-success">H: ${quote.high.toFixed(2)}</span>
          <span className="text-destructive">L: ${quote.low.toFixed(2)}</span>
        </div>
      </CardContent>
    </Card>
  )
}
