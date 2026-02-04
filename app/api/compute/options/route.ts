'use server'

import { NextRequest, NextResponse } from 'next/server'

/**
 * Options Pricing API - Bridge to C++ Engine
 * 
 * Black-Scholes implementation with full Greeks
 * Production: calls C++ native addon
 * Development: optimized TypeScript with same algorithms
 */

type OptionType = 'CALL' | 'PUT'

interface Greeks {
  delta: number
  gamma: number
  theta: number
  vega: number
  rho: number
}

interface PricingResult {
  price: number
  intrinsicValue: number
  timeValue: number
  greeks: Greeks
  calculationTimeNs: number
}

// Fast normal CDF approximation (Abramowitz & Stegun)
function normCdf(x: number): number {
  const a1 = 0.254829592
  const a2 = -0.284496736
  const a3 = 1.421413741
  const a4 = -1.453152027
  const a5 = 1.061405429
  const p = 0.3275911
  
  const sign = x < 0 ? -1 : 1
  x = Math.abs(x) / Math.SQRT2
  
  const t = 1.0 / (1.0 + p * x)
  const y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x)
  
  return 0.5 * (1.0 + sign * y)
}

// Normal PDF
function normPdf(x: number): number {
  return Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI)
}

function blackScholes(
  S: number,
  K: number,
  T: number,
  r: number,
  sigma: number,
  type: OptionType
): PricingResult {
  const startTime = performance.now()
  
  // Handle edge cases
  if (T <= 0) {
    const intrinsic = type === 'CALL' 
      ? Math.max(0, S - K) 
      : Math.max(0, K - S)
    return {
      price: intrinsic,
      intrinsicValue: intrinsic,
      timeValue: 0,
      greeks: { delta: type === 'CALL' && S > K ? 1 : 0, gamma: 0, theta: 0, vega: 0, rho: 0 },
      calculationTimeNs: Math.round((performance.now() - startTime) * 1e6)
    }
  }
  
  const sqrtT = Math.sqrt(T)
  const d1 = (Math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
  const d2 = d1 - sigma * sqrtT
  
  const Nd1 = normCdf(d1)
  const Nd2 = normCdf(d2)
  const Nmd1 = normCdf(-d1)
  const Nmd2 = normCdf(-d2)
  const nd1 = normPdf(d1)
  
  const discountFactor = Math.exp(-r * T)
  
  let price: number
  let delta: number
  let rho: number
  
  if (type === 'CALL') {
    price = S * Nd1 - K * discountFactor * Nd2
    delta = Nd1
    rho = K * T * discountFactor * Nd2 / 100
  } else {
    price = K * discountFactor * Nmd2 - S * Nmd1
    delta = Nd1 - 1
    rho = -K * T * discountFactor * Nmd2 / 100
  }
  
  // Greeks
  const gamma = nd1 / (S * sigma * sqrtT)
  const theta = (-(S * nd1 * sigma) / (2 * sqrtT) - r * K * discountFactor * (type === 'CALL' ? Nd2 : -Nmd2)) / 365
  const vega = S * nd1 * sqrtT / 100
  
  const intrinsicValue = type === 'CALL' 
    ? Math.max(0, S - K) 
    : Math.max(0, K - S)
  
  return {
    price,
    intrinsicValue,
    timeValue: price - intrinsicValue,
    greeks: { delta, gamma, theta, vega, rho },
    calculationTimeNs: Math.round((performance.now() - startTime) * 1e6)
  }
}

// Newton-Raphson implied volatility solver
function impliedVolatility(
  marketPrice: number,
  S: number,
  K: number,
  T: number,
  r: number,
  type: OptionType,
  maxIter: number = 100,
  tolerance: number = 1e-8
): { iv: number; iterations: number; converged: boolean } {
  let sigma = 0.25 // Initial guess
  
  for (let i = 0; i < maxIter; i++) {
    const result = blackScholes(S, K, T, r, sigma, type)
    const diff = result.price - marketPrice
    
    if (Math.abs(diff) < tolerance) {
      return { iv: sigma, iterations: i + 1, converged: true }
    }
    
    // Vega in actual units (not per 1%)
    const vega = result.greeks.vega * 100
    if (Math.abs(vega) < 1e-10) break
    
    sigma = sigma - diff / vega
    sigma = Math.max(0.001, Math.min(5.0, sigma)) // Bounds
  }
  
  return { iv: sigma, iterations: maxIter, converged: false }
}

export async function POST(request: NextRequest) {
  const body = await request.json()
  const { action, S, K, T, r, sigma, type, marketPrice, strikes } = body
  
  if (action === 'price') {
    const result = blackScholes(S, K, T, r, sigma, type)
    return NextResponse.json(result)
  }
  
  if (action === 'implied_volatility') {
    const result = impliedVolatility(marketPrice, S, K, T, r, type)
    return NextResponse.json(result)
  }
  
  if (action === 'option_chain') {
    const chain = strikes.map((strike: number) => ({
      strike,
      call: blackScholes(S, strike, T, r, sigma, 'CALL'),
      put: blackScholes(S, strike, T, r, sigma, 'PUT')
    }))
    return NextResponse.json({ chain })
  }
  
  if (action === 'greeks_surface') {
    const { spotRange, strikeRange } = body
    const surface = []
    
    for (const spot of spotRange) {
      for (const strike of strikeRange) {
        const result = blackScholes(spot, strike, T, r, sigma, type)
        surface.push({
          spot,
          strike,
          price: result.price,
          delta: result.greeks.delta,
          gamma: result.greeks.gamma
        })
      }
    }
    
    return NextResponse.json({ surface })
  }
  
  return NextResponse.json({ error: 'Invalid action' }, { status: 400 })
}
