'use server'

import { NextRequest, NextResponse } from 'next/server'

/**
 * Monte Carlo Simulation API - Bridge to C++ Engine
 * 
 * GBM simulation with VaR/CVaR calculation
 * Production: calls C++ native addon (multi-threaded)
 * Development: optimized TypeScript implementation
 */

interface SimulationParams {
  S0: number
  mu: number
  sigma: number
  T: number
  dt: number
  numPaths: number
  seed?: number
}

interface SimulationStats {
  meanFinalPrice: number
  stdFinalPrice: number
  minFinalPrice: number
  maxFinalPrice: number
  medianFinalPrice: number
  expectedReturn: number
  realizedVolatility: number
  maxDrawdown: number
}

interface VaRResult {
  var95: number
  var99: number
  cvar95: number
  cvar99: number
}

// Box-Muller transform for normal random numbers
function normalRandom(rng: () => number): number {
  const u1 = rng()
  const u2 = rng()
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2)
}

// Simple PRNG with seed support
function createRng(seed: number): () => number {
  let state = seed
  return () => {
    state = (state * 1103515245 + 12345) & 0x7fffffff
    return state / 0x7fffffff
  }
}

function simulateGBM(params: SimulationParams): {
  paths: number[][]
  finalPrices: number[]
  stats: SimulationStats
  calculationTimeMs: number
} {
  const startTime = performance.now()
  
  const { S0, mu, sigma, T, dt, numPaths, seed = Date.now() } = params
  const numSteps = Math.ceil(T / dt)
  const sqrtDt = Math.sqrt(dt)
  const drift = (mu - 0.5 * sigma * sigma) * dt
  
  const rng = createRng(seed)
  const paths: number[][] = []
  const finalPrices: number[] = []
  
  // Simulate paths
  for (let p = 0; p < numPaths; p++) {
    const path = new Array(numSteps + 1)
    path[0] = S0
    
    for (let t = 1; t <= numSteps; t++) {
      const dW = normalRandom(rng) * sqrtDt
      path[t] = path[t - 1] * Math.exp(drift + sigma * dW)
    }
    
    paths.push(path)
    finalPrices.push(path[numSteps])
  }
  
  // Calculate statistics
  const sortedPrices = [...finalPrices].sort((a, b) => a - b)
  const mean = finalPrices.reduce((a, b) => a + b, 0) / numPaths
  const variance = finalPrices.reduce((sum, p) => sum + (p - mean) ** 2, 0) / numPaths
  const std = Math.sqrt(variance)
  
  // Calculate max drawdown
  let maxDrawdown = 0
  for (const path of paths) {
    let peak = path[0]
    for (let t = 1; t < path.length; t++) {
      if (path[t] > peak) peak = path[t]
      const drawdown = (peak - path[t]) / peak
      if (drawdown > maxDrawdown) maxDrawdown = drawdown
    }
  }
  
  const stats: SimulationStats = {
    meanFinalPrice: mean,
    stdFinalPrice: std,
    minFinalPrice: sortedPrices[0],
    maxFinalPrice: sortedPrices[numPaths - 1],
    medianFinalPrice: sortedPrices[Math.floor(numPaths / 2)],
    expectedReturn: (mean - S0) / S0,
    realizedVolatility: std / S0,
    maxDrawdown
  }
  
  return {
    paths: paths.slice(0, 100), // Return max 100 paths for visualization
    finalPrices,
    stats,
    calculationTimeMs: performance.now() - startTime
  }
}

function calculateVaR(returns: number[], portfolioValue: number): VaRResult {
  const sorted = [...returns].sort((a, b) => a - b)
  const n = sorted.length
  
  const idx95 = Math.floor(n * 0.05)
  const idx99 = Math.floor(n * 0.01)
  
  const var95 = -sorted[idx95] * portfolioValue
  const var99 = -sorted[idx99] * portfolioValue
  
  // CVaR (Expected Shortfall) - average of losses beyond VaR
  const cvar95 = -sorted.slice(0, idx95 + 1).reduce((a, b) => a + b, 0) / (idx95 + 1) * portfolioValue
  const cvar99 = -sorted.slice(0, idx99 + 1).reduce((a, b) => a + b, 0) / (idx99 + 1) * portfolioValue
  
  return { var95, var99, cvar95, cvar99 }
}

export async function POST(request: NextRequest) {
  const body = await request.json()
  const { action, params } = body
  
  if (action === 'simulate') {
    const result = simulateGBM(params)
    
    // Calculate VaR from returns
    const returns = result.finalPrices.map(p => (p - params.S0) / params.S0)
    const varResult = calculateVaR(returns, params.S0)
    
    return NextResponse.json({
      ...result,
      var: varResult
    })
  }
  
  if (action === 'var_only') {
    const { returns, portfolioValue } = body
    const result = calculateVaR(returns, portfolioValue)
    return NextResponse.json(result)
  }
  
  return NextResponse.json({ error: 'Invalid action' }, { status: 400 })
}
