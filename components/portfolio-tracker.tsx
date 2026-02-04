'use client'

import React from "react"

import { useState, useEffect, useCallback } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Badge } from '@/components/ui/badge'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { createClient } from '@/lib/supabase/client'
import { Plus, TrendingUp, TrendingDown, DollarSign, PieChart, Trash2 } from 'lucide-react'
import type { StockQuote } from '@/lib/types'

interface Holding {
  id: string
  symbol: string
  shares: number
  avg_cost_basis: number
  current_price?: number
  current_value?: number
  gain_loss?: number
  gain_loss_percent?: number
}

interface Transaction {
  id: string
  symbol: string
  transaction_type: 'BUY' | 'SELL'
  shares: number
  price_per_share: number
  total_amount: number
  transaction_date: string
}

export function PortfolioTracker() {
  const [holdings, setHoldings] = useState<Holding[]>([])
  const [transactions, setTransactions] = useState<Transaction[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [dialogOpen, setDialogOpen] = useState(false)
  const [formData, setFormData] = useState({
    symbol: '',
    type: 'BUY' as 'BUY' | 'SELL',
    shares: '',
    price: '',
  })
  const supabase = createClient()

  const fetchHoldings = useCallback(async () => {
    const { data: holdingsData } = await supabase
      .from('holdings')
      .select('*')
      .order('symbol')

    if (holdingsData) {
      // Fetch current prices for all holdings
      const holdingsWithPrices = await Promise.all(
        holdingsData.map(async (holding) => {
          try {
            const res = await fetch(`/api/stock/quote?symbol=${holding.symbol}`)
            if (res.ok) {
              const quote: StockQuote = await res.json()
              const currentValue = Number(holding.shares) * quote.price
              const costBasis = Number(holding.shares) * Number(holding.avg_cost_basis)
              const gainLoss = currentValue - costBasis
              const gainLossPercent = (gainLoss / costBasis) * 100
              return {
                ...holding,
                shares: Number(holding.shares),
                avg_cost_basis: Number(holding.avg_cost_basis),
                current_price: quote.price,
                current_value: currentValue,
                gain_loss: gainLoss,
                gain_loss_percent: gainLossPercent,
              }
            }
          } catch {
            // Skip price fetch errors
          }
          return {
            ...holding,
            shares: Number(holding.shares),
            avg_cost_basis: Number(holding.avg_cost_basis),
          }
        })
      )
      setHoldings(holdingsWithPrices)
    }
  }, [supabase])

  const fetchTransactions = useCallback(async () => {
    const { data } = await supabase
      .from('transactions')
      .select('*')
      .order('transaction_date', { ascending: false })
      .limit(10)

    if (data) {
      setTransactions(data.map(t => ({
        ...t,
        shares: Number(t.shares),
        price_per_share: Number(t.price_per_share),
        total_amount: Number(t.total_amount),
      })))
    }
  }, [supabase])

  useEffect(() => {
    const loadData = async () => {
      setIsLoading(true)
      await Promise.all([fetchHoldings(), fetchTransactions()])
      setIsLoading(false)
    }
    loadData()
  }, [fetchHoldings, fetchTransactions])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    const { data: { user } } = await supabase.auth.getUser()
    if (!user) return

    const shares = parseFloat(formData.shares)
    const price = parseFloat(formData.price)
    const total = shares * price
    const symbol = formData.symbol.toUpperCase()

    // Add transaction
    await supabase.from('transactions').insert({
      user_id: user.id,
      symbol,
      transaction_type: formData.type,
      shares,
      price_per_share: price,
      total_amount: total,
    })

    // Update holdings
    const existingHolding = holdings.find(h => h.symbol === symbol)
    
    if (formData.type === 'BUY') {
      if (existingHolding) {
        const newShares = existingHolding.shares + shares
        const newCostBasis = ((existingHolding.shares * existingHolding.avg_cost_basis) + total) / newShares
        await supabase
          .from('holdings')
          .update({ shares: newShares, avg_cost_basis: newCostBasis, updated_at: new Date().toISOString() })
          .eq('id', existingHolding.id)
      } else {
        await supabase.from('holdings').insert({
          user_id: user.id,
          symbol,
          shares,
          avg_cost_basis: price,
        })
      }
    } else if (formData.type === 'SELL' && existingHolding) {
      const newShares = existingHolding.shares - shares
      if (newShares <= 0) {
        await supabase.from('holdings').delete().eq('id', existingHolding.id)
      } else {
        await supabase
          .from('holdings')
          .update({ shares: newShares, updated_at: new Date().toISOString() })
          .eq('id', existingHolding.id)
      }
    }

    setFormData({ symbol: '', type: 'BUY', shares: '', price: '' })
    setDialogOpen(false)
    await Promise.all([fetchHoldings(), fetchTransactions()])
  }

  const deleteHolding = async (id: string) => {
    await supabase.from('holdings').delete().eq('id', id)
    await fetchHoldings()
  }

  const totalValue = holdings.reduce((sum, h) => sum + (h.current_value || 0), 0)
  const totalCost = holdings.reduce((sum, h) => sum + (h.shares * h.avg_cost_basis), 0)
  const totalGainLoss = totalValue - totalCost
  const totalGainLossPercent = totalCost > 0 ? (totalGainLoss / totalCost) * 100 : 0

  if (isLoading) {
    return (
      <div className="space-y-4">
        <Card className="bg-card/50 border-border/50">
          <CardContent className="p-8">
            <div className="flex items-center justify-center">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary" />
            </div>
          </CardContent>
        </Card>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      {/* Portfolio Summary */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card className="bg-card/50 border-border/50">
          <CardContent className="p-4">
            <div className="flex items-center gap-2 text-muted-foreground text-sm">
              <DollarSign className="h-4 w-4" />
              Portfolio Value
            </div>
            <p className="text-2xl font-bold text-foreground mt-1">
              ${totalValue.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
            </p>
          </CardContent>
        </Card>
        <Card className="bg-card/50 border-border/50">
          <CardContent className="p-4">
            <div className="flex items-center gap-2 text-muted-foreground text-sm">
              <PieChart className="h-4 w-4" />
              Cost Basis
            </div>
            <p className="text-2xl font-bold text-foreground mt-1">
              ${totalCost.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
            </p>
          </CardContent>
        </Card>
        <Card className="bg-card/50 border-border/50">
          <CardContent className="p-4">
            <div className="flex items-center gap-2 text-muted-foreground text-sm">
              {totalGainLoss >= 0 ? <TrendingUp className="h-4 w-4 text-primary" /> : <TrendingDown className="h-4 w-4 text-destructive" />}
              Total Gain/Loss
            </div>
            <p className={`text-2xl font-bold mt-1 ${totalGainLoss >= 0 ? 'text-primary' : 'text-destructive'}`}>
              {totalGainLoss >= 0 ? '+' : ''}${totalGainLoss.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
            </p>
          </CardContent>
        </Card>
        <Card className="bg-card/50 border-border/50">
          <CardContent className="p-4">
            <div className="flex items-center gap-2 text-muted-foreground text-sm">
              {totalGainLossPercent >= 0 ? <TrendingUp className="h-4 w-4 text-primary" /> : <TrendingDown className="h-4 w-4 text-destructive" />}
              Return %
            </div>
            <p className={`text-2xl font-bold mt-1 ${totalGainLossPercent >= 0 ? 'text-primary' : 'text-destructive'}`}>
              {totalGainLossPercent >= 0 ? '+' : ''}{totalGainLossPercent.toFixed(2)}%
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Holdings Table */}
      <Card className="bg-card/50 border-border/50">
        <CardHeader className="flex flex-row items-center justify-between">
          <CardTitle className="text-foreground">Holdings</CardTitle>
          <Dialog open={dialogOpen} onOpenChange={setDialogOpen}>
            <DialogTrigger asChild>
              <Button size="sm" className="gap-1">
                <Plus className="h-4 w-4" /> Add Transaction
              </Button>
            </DialogTrigger>
            <DialogContent className="bg-card border-border">
              <DialogHeader>
                <DialogTitle className="text-foreground">Add Transaction</DialogTitle>
              </DialogHeader>
              <form onSubmit={handleSubmit} className="space-y-4">
                <div className="grid gap-2">
                  <Label htmlFor="symbol">Symbol</Label>
                  <Input
                    id="symbol"
                    placeholder="AAPL"
                    value={formData.symbol}
                    onChange={(e) => setFormData({ ...formData, symbol: e.target.value })}
                    required
                    className="bg-input border-border"
                  />
                </div>
                <div className="grid gap-2">
                  <Label>Type</Label>
                  <Select value={formData.type} onValueChange={(v: 'BUY' | 'SELL') => setFormData({ ...formData, type: v })}>
                    <SelectTrigger className="bg-input border-border">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="BUY">Buy</SelectItem>
                      <SelectItem value="SELL">Sell</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <div className="grid gap-2">
                    <Label htmlFor="shares">Shares</Label>
                    <Input
                      id="shares"
                      type="number"
                      step="0.00000001"
                      placeholder="10"
                      value={formData.shares}
                      onChange={(e) => setFormData({ ...formData, shares: e.target.value })}
                      required
                      className="bg-input border-border"
                    />
                  </div>
                  <div className="grid gap-2">
                    <Label htmlFor="price">Price</Label>
                    <Input
                      id="price"
                      type="number"
                      step="0.01"
                      placeholder="150.00"
                      value={formData.price}
                      onChange={(e) => setFormData({ ...formData, price: e.target.value })}
                      required
                      className="bg-input border-border"
                    />
                  </div>
                </div>
                <Button type="submit" className="w-full">
                  Add {formData.type === 'BUY' ? 'Purchase' : 'Sale'}
                </Button>
              </form>
            </DialogContent>
          </Dialog>
        </CardHeader>
        <CardContent>
          {holdings.length === 0 ? (
            <p className="text-muted-foreground text-center py-8">
              No holdings yet. Add your first transaction to get started.
            </p>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-border/50 text-muted-foreground text-sm">
                    <th className="text-left py-2 px-2">Symbol</th>
                    <th className="text-right py-2 px-2">Shares</th>
                    <th className="text-right py-2 px-2">Avg Cost</th>
                    <th className="text-right py-2 px-2">Current</th>
                    <th className="text-right py-2 px-2">Value</th>
                    <th className="text-right py-2 px-2">Gain/Loss</th>
                    <th className="text-right py-2 px-2"></th>
                  </tr>
                </thead>
                <tbody>
                  {holdings.map((holding) => (
                    <tr key={holding.id} className="border-b border-border/30">
                      <td className="py-3 px-2">
                        <span className="font-medium text-foreground">{holding.symbol}</span>
                      </td>
                      <td className="text-right py-3 px-2 text-foreground">{holding.shares.toFixed(4)}</td>
                      <td className="text-right py-3 px-2 text-foreground">${holding.avg_cost_basis.toFixed(2)}</td>
                      <td className="text-right py-3 px-2 text-foreground">
                        {holding.current_price ? `$${holding.current_price.toFixed(2)}` : '-'}
                      </td>
                      <td className="text-right py-3 px-2 text-foreground">
                        {holding.current_value ? `$${holding.current_value.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}` : '-'}
                      </td>
                      <td className="text-right py-3 px-2">
                        {holding.gain_loss !== undefined ? (
                          <div className={holding.gain_loss >= 0 ? 'text-primary' : 'text-destructive'}>
                            <span>{holding.gain_loss >= 0 ? '+' : ''}${holding.gain_loss.toFixed(2)}</span>
                            <Badge variant={holding.gain_loss >= 0 ? 'default' : 'destructive'} className="ml-2 text-xs">
                              {holding.gain_loss_percent && holding.gain_loss_percent >= 0 ? '+' : ''}{holding.gain_loss_percent?.toFixed(2)}%
                            </Badge>
                          </div>
                        ) : '-'}
                      </td>
                      <td className="text-right py-3 px-2">
                        <Button variant="ghost" size="sm" onClick={() => deleteHolding(holding.id)}>
                          <Trash2 className="h-4 w-4 text-muted-foreground hover:text-destructive" />
                        </Button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Recent Transactions */}
      <Card className="bg-card/50 border-border/50">
        <CardHeader>
          <CardTitle className="text-foreground">Recent Transactions</CardTitle>
        </CardHeader>
        <CardContent>
          {transactions.length === 0 ? (
            <p className="text-muted-foreground text-center py-4">No transactions yet.</p>
          ) : (
            <div className="space-y-2">
              {transactions.map((tx) => (
                <div key={tx.id} className="flex items-center justify-between py-2 border-b border-border/30 last:border-0">
                  <div className="flex items-center gap-3">
                    <Badge variant={tx.transaction_type === 'BUY' ? 'default' : 'destructive'}>
                      {tx.transaction_type}
                    </Badge>
                    <span className="font-medium text-foreground">{tx.symbol}</span>
                    <span className="text-muted-foreground text-sm">
                      {tx.shares} shares @ ${tx.price_per_share.toFixed(2)}
                    </span>
                  </div>
                  <div className="text-right">
                    <p className="font-medium text-foreground">${tx.total_amount.toFixed(2)}</p>
                    <p className="text-xs text-muted-foreground">
                      {new Date(tx.transaction_date).toLocaleDateString()}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
