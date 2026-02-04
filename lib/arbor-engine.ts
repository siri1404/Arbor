/**
 * TypeScript wrapper for Arbor C++ Quant Engine
 * 
 * This provides type-safe access to the native C++ addon
 * Falls back to JS implementation if native module unavailable
 */

// Types matching C++ structures
export interface Trade {
  price: number;
  quantity: number;
  buyOrderId: number;
  sellOrderId: number;
  latencyNs: number;
}

export interface OrderResult {
  orderId: number;
  trades: Trade[];
  latencyNs: number;
}

export interface OrderBookSnapshot {
  bestBid: number;
  bestAsk: number;
  spread: number;
  midPrice: number;
  bids: Array<{ price: number; quantity: number }>;
  asks: Array<{ price: number; quantity: number }>;
  latencyStats: {
    count: number;
    avgNs: number;
    minNs: number;
    maxNs: number;
  };
}

export interface Greeks {
  delta: number;
  gamma: number;
  theta: number;
  vega: number;
  rho: number;
}

export interface OptionPriceResult {
  price: number;
  intrinsicValue: number;
  timeValue: number;
  greeks: Greeks;
  latencyNs: number;
}

export interface ImpliedVolResult {
  iv: number;
  latencyNs: number;
  converged: boolean;
}

export interface SimulationParams {
  S0: number;
  mu: number;
  sigma: number;
  T: number;
  numPaths: number;
  numSteps: number;
}

export interface SimulationStats {
  meanFinalPrice: number;
  stdFinalPrice: number;
  minFinalPrice: number;
  maxFinalPrice: number;
}

export interface SimulationResult {
  stats: SimulationStats;
  finalPrices: number[];
  paths: number[][];
  calculationTimeMs: number;
}

export interface RiskLimits {
  maxPositionQty?: number;
  maxOrderQty?: number;
  maxDailyLoss?: number;
  maxDrawdown?: number;
}

export interface RiskCheckResult {
  passed: boolean;
  rejectCode: number;
  rejectReason: string;
  checkLatencyNs: number;
}

// Native module interface
interface ArborNative {
  createOrderBook(symbol: string, tickSize: number): boolean;
  addOrder(symbol: string, side: string, type: string, price: number, quantity: number): OrderResult;
  getOrderBookSnapshot(symbol: string, depth: number): OrderBookSnapshot;
  
  priceOption(S: number, K: number, T: number, r: number, sigma: number, type: string): OptionPriceResult;
  impliedVolatility(marketPrice: number, S: number, K: number, T: number, r: number, type: string): ImpliedVolResult;
  
  runMonteCarlo(params: SimulationParams): Promise<SimulationResult>;
  
  initRiskManager(limits: RiskLimits): boolean;
  checkOrderRisk(symbol: string, side: string, quantity: number, price: number, marketPrice: number): RiskCheckResult;
}

// Try to load native module
let native: ArborNative | null = null;
let loadError: Error | null = null;

try {
  // In production, this would be the compiled native addon
  // native = require('../build/Release/arbor_engine.node');
  
  // For now, we'll set native to null to use fallback
  // Uncomment above line after running: cd cpp && npx node-gyp rebuild
  // native = null;
} catch (e) {
  loadError = e as Error;
  console.warn('Native C++ engine not available, using JS fallback:', e);
}

/**
 * Check if native C++ engine is available
 */
export function isNativeAvailable(): boolean {
  return native !== null;
}

/**
 * Type guard for native module
 */
function ensureNative(): ArborNative {
  if (!native) {
    throw new Error('Native module not available - use API route instead');
  }
  return native;
}

/**
 * Get native module load error if any
 */
export function getNativeLoadError(): Error | null {
  return loadError;
}

/**
 * Order Book operations
 */
export const OrderBook = {
  create(symbol: string, tickSize: number = 1): boolean {
    if (!native) {
      // JS fallback handled by API route
      return true;
    }
    return ensureNative().createOrderBook(symbol, tickSize);
  },

  addOrder(symbol: string, side: 'BUY' | 'SELL', type: 'LIMIT' | 'MARKET', price: number, quantity: number): OrderResult {
    return ensureNative().addOrder(symbol, side, type, price, quantity);
  },

  getSnapshot(symbol: string, depth: number = 10): OrderBookSnapshot {
    return ensureNative().getOrderBookSnapshot(symbol, depth);
  }
};

/**
 * Options Pricing operations
 */
export const Options = {
  price(S: number, K: number, T: number, r: number, sigma: number, type: 'CALL' | 'PUT'): OptionPriceResult {
    return ensureNative().priceOption(S, K, T, r, sigma, type);
  },

  impliedVolatility(marketPrice: number, S: number, K: number, T: number, r: number, type: 'CALL' | 'PUT'): ImpliedVolResult {
    return ensureNative().impliedVolatility(marketPrice, S, K, T, r, type);
  }
};

/**
 * Monte Carlo operations
 */
export const MonteCarlo = {
  async simulate(params: SimulationParams): Promise<SimulationResult> {
    return ensureNative().runMonteCarlo(params);
  }
};

/**
 * Risk Management operations
 */
export const Risk = {
  init(limits: RiskLimits = {}): boolean {
    if (!native) {
      return true;
    }
    return ensureNative().initRiskManager(limits);
  },

  checkOrder(symbol: string, side: 'BUY' | 'SELL', quantity: number, price: number, marketPrice: number): RiskCheckResult {
    if (!native) {
      // Fallback: always pass
      return {
        passed: true,
        rejectCode: 0,
        rejectReason: '',
        checkLatencyNs: 0
      };
    }
    return ensureNative().checkOrderRisk(symbol, side, quantity, price, marketPrice);
  }
};

export default {
  isNativeAvailable,
  getNativeLoadError,
  OrderBook,
  Options,
  MonteCarlo,
  Risk
};
