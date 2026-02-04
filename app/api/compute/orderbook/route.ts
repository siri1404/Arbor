'use server'

import { NextRequest, NextResponse } from 'next/server'

/**
 * Order Book API - Bridge to C++ Engine
 * 
 * In production, this would call the C++ native addon via:
 * const arbor = require('./cpp/build/Release/arbor_addon.node')
 * 
 * For development/demo, we provide optimized TypeScript fallback
 * that demonstrates the same algorithmic concepts
 */

interface Order {
  id: string
  side: 'BUY' | 'SELL'
  type: 'LIMIT' | 'MARKET'
  price: number
  quantity: number
  timestamp: number
}

interface Trade {
  price: number
  quantity: number
  buyOrderId: string
  sellOrderId: string
  timestamp: number
  latencyNs: number
}

// In-memory order book state (would be C++ native module in production)
const bids: Map<number, Order[]> = new Map()
const asks: Map<number, Order[]> = new Map()
let orderIdCounter = 0

function generateOrderId(): string {
  return `ORD-${++orderIdCounter}-${Date.now()}`
}

function matchOrder(order: Order): Trade[] {
  const startTime = performance.now()
  const trades: Trade[] = []
  
  if (order.side === 'BUY') {
    // Match against asks (sorted ascending)
    const sortedAsks = [...asks.keys()].sort((a, b) => a - b)
    
    for (const askPrice of sortedAsks) {
      if (order.type === 'LIMIT' && askPrice > order.price) break
      if (order.quantity <= 0) break
      
      const askOrders = asks.get(askPrice) || []
      while (askOrders.length > 0 && order.quantity > 0) {
        const restingOrder = askOrders[0]
        const fillQty = Math.min(order.quantity, restingOrder.quantity)
        
        trades.push({
          price: askPrice,
          quantity: fillQty,
          buyOrderId: order.id,
          sellOrderId: restingOrder.id,
          timestamp: Date.now(),
          latencyNs: Math.round((performance.now() - startTime) * 1e6)
        })
        
        order.quantity -= fillQty
        restingOrder.quantity -= fillQty
        
        if (restingOrder.quantity === 0) {
          askOrders.shift()
        }
      }
      
      if (askOrders.length === 0) {
        asks.delete(askPrice)
      }
    }
    
    // Add remaining to book
    if (order.quantity > 0 && order.type === 'LIMIT') {
      const existing = bids.get(order.price) || []
      existing.push(order)
      bids.set(order.price, existing)
    }
  } else {
    // Match against bids (sorted descending)
    const sortedBids = [...bids.keys()].sort((a, b) => b - a)
    
    for (const bidPrice of sortedBids) {
      if (order.type === 'LIMIT' && bidPrice < order.price) break
      if (order.quantity <= 0) break
      
      const bidOrders = bids.get(bidPrice) || []
      while (bidOrders.length > 0 && order.quantity > 0) {
        const restingOrder = bidOrders[0]
        const fillQty = Math.min(order.quantity, restingOrder.quantity)
        
        trades.push({
          price: bidPrice,
          quantity: fillQty,
          buyOrderId: restingOrder.id,
          sellOrderId: order.id,
          timestamp: Date.now(),
          latencyNs: Math.round((performance.now() - startTime) * 1e6)
        })
        
        order.quantity -= fillQty
        restingOrder.quantity -= fillQty
        
        if (restingOrder.quantity === 0) {
          bidOrders.shift()
        }
      }
      
      if (bidOrders.length === 0) {
        bids.delete(bidPrice)
      }
    }
    
    // Add remaining to book
    if (order.quantity > 0 && order.type === 'LIMIT') {
      const existing = asks.get(order.price) || []
      existing.push(order)
      asks.set(order.price, existing)
    }
  }
  
  return trades
}

export async function POST(request: NextRequest) {
  const body = await request.json()
  const { action, side, type, price, quantity } = body
  
  if (action === 'add_order') {
    const order: Order = {
      id: generateOrderId(),
      side,
      type,
      price: Math.round(price * 100) / 100,
      quantity,
      timestamp: Date.now()
    }
    
    const trades = matchOrder(order)
    
    // Get snapshot
    const bidLevels = [...bids.entries()]
      .sort((a, b) => b[0] - a[0])
      .slice(0, 15)
      .map(([price, orders]) => ({
        price,
        quantity: orders.reduce((sum, o) => sum + o.quantity, 0),
        orders: orders.length
      }))
    
    const askLevels = [...asks.entries()]
      .sort((a, b) => a[0] - b[0])
      .slice(0, 15)
      .map(([price, orders]) => ({
        price,
        quantity: orders.reduce((sum, o) => sum + o.quantity, 0),
        orders: orders.length
      }))
    
    const bestBid = bidLevels[0]?.price || 0
    const bestAsk = askLevels[0]?.price || 0
    
    return NextResponse.json({
      orderId: order.id,
      trades,
      snapshot: {
        bids: bidLevels,
        asks: askLevels,
        spread: bestAsk - bestBid,
        midPrice: (bestBid + bestAsk) / 2
      }
    })
  }
  
  if (action === 'reset') {
    bids.clear()
    asks.clear()
    orderIdCounter = 0
    return NextResponse.json({ success: true })
  }
  
  return NextResponse.json({ error: 'Invalid action' }, { status: 400 })
}

export async function GET() {
  const bidLevels = [...bids.entries()]
    .sort((a, b) => b[0] - a[0])
    .slice(0, 15)
    .map(([price, orders]) => ({
      price,
      quantity: orders.reduce((sum, o) => sum + o.quantity, 0),
      orders: orders.length
    }))
  
  const askLevels = [...asks.entries()]
    .sort((a, b) => a[0] - b[0])
    .slice(0, 15)
    .map(([price, orders]) => ({
      price,
      quantity: orders.reduce((sum, o) => sum + o.quantity, 0),
      orders: orders.length
    }))
  
  const bestBid = bidLevels[0]?.price || 0
  const bestAsk = askLevels[0]?.price || 0
  
  return NextResponse.json({
    bids: bidLevels,
    asks: askLevels,
    spread: bestAsk - bestBid,
    midPrice: (bestBid + bestAsk) / 2
  })
}
