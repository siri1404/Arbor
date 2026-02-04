#define NAPI_VERSION 8
#include <node_api.h>

#include "orderbook.hpp"
#include "options_pricing.hpp"
#include "monte_carlo.hpp"
#include "risk_manager.hpp"
#include "lockfree_queue.hpp"

#include <memory>
#include <unordered_map>
#include <string>
#include <chrono>

/**
 * Node.js N-API Bindings for Arbor Quant Engine
 * 
 * Exposes C++ computational core to JavaScript/TypeScript
 * Zero-copy where possible, async for heavy computations
 */

namespace arbor::bindings {

// Global instances (one per symbol for order books)
std::unordered_map<std::string, std::unique_ptr<orderbook::LimitOrderBook>> order_books;
std::unique_ptr<risk::RiskManager> risk_manager;
std::unique_ptr<risk::OrderSequencer> sequencer;

// Helper macros for N-API error handling
#define NAPI_CALL(env, call)                                      \
  do {                                                            \
    napi_status status = (call);                                  \
    if (status != napi_ok) {                                      \
      napi_throw_error(env, nullptr, "N-API call failed");        \
      return nullptr;                                             \
    }                                                             \
  } while(0)

#define NAPI_ASSERT(env, condition, message)                      \
  do {                                                            \
    if (!(condition)) {                                           \
      napi_throw_error(env, nullptr, message);                    \
      return nullptr;                                             \
    }                                                             \
  } while(0)

// Helper to get string from napi_value
std::string get_string(napi_env env, napi_value value) {
    size_t len;
    napi_get_value_string_utf8(env, value, nullptr, 0, &len);
    std::string result(len, '\0');
    napi_get_value_string_utf8(env, value, &result[0], len + 1, &len);
    return result;
}

// Helper to create object with properties
napi_value create_result_object(napi_env env) {
    napi_value obj;
    napi_create_object(env, &obj);
    return obj;
}

void set_property_double(napi_env env, napi_value obj, const char* name, double value) {
    napi_value nval;
    napi_create_double(env, value, &nval);
    napi_set_named_property(env, obj, name, nval);
}

void set_property_int64(napi_env env, napi_value obj, const char* name, int64_t value) {
    napi_value nval;
    napi_create_int64(env, value, &nval);
    napi_set_named_property(env, obj, name, nval);
}

void set_property_bool(napi_env env, napi_value obj, const char* name, bool value) {
    napi_value nval;
    napi_get_boolean(env, value, &nval);
    napi_set_named_property(env, obj, name, nval);
}

void set_property_string(napi_env env, napi_value obj, const char* name, const std::string& value) {
    napi_value nval;
    napi_create_string_utf8(env, value.c_str(), value.length(), &nval);
    napi_set_named_property(env, obj, name, nval);
}

//=============================================================================
// ORDER BOOK BINDINGS
//=============================================================================

/**
 * createOrderBook(symbol: string, tickSize: number): boolean
 */
napi_value CreateOrderBook(napi_env env, napi_callback_info info) {
    size_t argc = 2;
    napi_value args[2];
    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, args, nullptr, nullptr));
    NAPI_ASSERT(env, argc >= 2, "Expected 2 arguments: symbol, tickSize");

    std::string symbol = get_string(env, args[0]);
    
    int64_t tick_size;
    napi_get_value_int64(env, args[1], &tick_size);

    order_books[symbol] = std::make_unique<orderbook::LimitOrderBook>(symbol, tick_size);

    napi_value result;
    napi_get_boolean(env, true, &result);
    return result;
}

/**
 * addOrder(symbol: string, side: string, type: string, price: number, quantity: number)
 * Returns: { orderId, trades: [...], latencyNs }
 */
napi_value AddOrder(napi_env env, napi_callback_info info) {
    size_t argc = 5;
    napi_value args[5];
    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, args, nullptr, nullptr));
    NAPI_ASSERT(env, argc >= 5, "Expected 5 arguments");

    std::string symbol = get_string(env, args[0]);
    std::string side_str = get_string(env, args[1]);
    std::string type_str = get_string(env, args[2]);
    
    double price_d;
    napi_get_value_double(env, args[3], &price_d);
    int64_t price = static_cast<int64_t>(price_d * 100);  // Convert to ticks
    
    int64_t quantity;
    napi_get_value_int64(env, args[4], &quantity);

    auto it = order_books.find(symbol);
    NAPI_ASSERT(env, it != order_books.end(), "Order book not found for symbol");

    auto side = (side_str == "BUY") ? orderbook::Side::BUY : orderbook::Side::SELL;
    auto type = (type_str == "MARKET") ? orderbook::OrderType::MARKET : orderbook::OrderType::LIMIT;

    std::vector<orderbook::Trade> trades;
    
    auto start = std::chrono::steady_clock::now();
    uint64_t order_id = it->second->add_order(side, type, price, quantity, trades);
    auto end = std::chrono::steady_clock::now();
    
    int64_t latency_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    // Build result object
    napi_value result = create_result_object(env);
    set_property_int64(env, result, "orderId", order_id);
    set_property_int64(env, result, "latencyNs", latency_ns);

    // Build trades array
    napi_value trades_array;
    napi_create_array_with_length(env, trades.size(), &trades_array);
    
    for (size_t i = 0; i < trades.size(); i++) {
        napi_value trade = create_result_object(env);
        set_property_double(env, trade, "price", trades[i].price_ticks / 100.0);
        set_property_int64(env, trade, "quantity", trades[i].quantity);
        set_property_int64(env, trade, "buyOrderId", trades[i].buy_order_id);
        set_property_int64(env, trade, "sellOrderId", trades[i].sell_order_id);
        set_property_int64(env, trade, "latencyNs", trades[i].match_latency_ns);
        napi_set_element(env, trades_array, i, trade);
    }
    napi_set_named_property(env, result, "trades", trades_array);

    return result;
}

/**
 * getOrderBookSnapshot(symbol: string, depth: number)
 */
napi_value GetOrderBookSnapshot(napi_env env, napi_callback_info info) {
    size_t argc = 2;
    napi_value args[2];
    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, args, nullptr, nullptr));

    std::string symbol = get_string(env, args[0]);
    int32_t depth;
    napi_get_value_int32(env, args[1], &depth);

    auto it = order_books.find(symbol);
    NAPI_ASSERT(env, it != order_books.end(), "Order book not found");

    napi_value result = create_result_object(env);
    
    set_property_double(env, result, "bestBid", it->second->best_bid() / 100.0);
    set_property_double(env, result, "bestAsk", it->second->best_ask() / 100.0);
    set_property_double(env, result, "spread", it->second->spread() / 100.0);
    set_property_double(env, result, "midPrice", it->second->mid_price() / 100.0);

    // Bids array
    napi_value bids;
    napi_create_array(env, &bids);
    int bid_idx = 0;
    // Would iterate through bid levels here...
    napi_set_named_property(env, result, "bids", bids);

    // Asks array  
    napi_value asks;
    napi_create_array(env, &asks);
    napi_set_named_property(env, result, "asks", asks);

    // Latency stats
    const auto& stats = it->second->get_latency_stats();
    napi_value latency = create_result_object(env);
    set_property_int64(env, latency, "count", stats.count);
    set_property_double(env, latency, "avgNs", stats.avg_ns());
    set_property_int64(env, latency, "minNs", stats.min_ns);
    set_property_int64(env, latency, "maxNs", stats.max_ns);
    napi_set_named_property(env, result, "latencyStats", latency);

    return result;
}

//=============================================================================
// OPTIONS PRICING BINDINGS
//=============================================================================

/**
 * priceOption(S, K, T, r, sigma, type) 
 * Returns: { price, greeks: { delta, gamma, theta, vega, rho }, latencyNs }
 */
napi_value PriceOption(napi_env env, napi_callback_info info) {
    size_t argc = 6;
    napi_value args[6];
    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, args, nullptr, nullptr));
    NAPI_ASSERT(env, argc >= 6, "Expected 6 arguments: S, K, T, r, sigma, type");

    double S, K, T, r, sigma;
    napi_get_value_double(env, args[0], &S);
    napi_get_value_double(env, args[1], &K);
    napi_get_value_double(env, args[2], &T);
    napi_get_value_double(env, args[3], &r);
    napi_get_value_double(env, args[4], &sigma);
    
    std::string type_str = get_string(env, args[5]);
    auto type = (type_str == "PUT") ? options::OptionType::PUT : options::OptionType::CALL;

    auto start = std::chrono::steady_clock::now();
    auto pricing = options::BlackScholesPricer::price(S, K, T, r, sigma, type);
    auto end = std::chrono::steady_clock::now();
    
    int64_t latency_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    napi_value result = create_result_object(env);
    set_property_double(env, result, "price", pricing.price);
    set_property_double(env, result, "intrinsicValue", pricing.intrinsic_value);
    set_property_double(env, result, "timeValue", pricing.time_value);
    set_property_int64(env, result, "latencyNs", latency_ns);

    napi_value greeks = create_result_object(env);
    set_property_double(env, greeks, "delta", pricing.greeks.delta);
    set_property_double(env, greeks, "gamma", pricing.greeks.gamma);
    set_property_double(env, greeks, "theta", pricing.greeks.theta);
    set_property_double(env, greeks, "vega", pricing.greeks.vega);
    set_property_double(env, greeks, "rho", pricing.greeks.rho);
    napi_set_named_property(env, result, "greeks", greeks);

    return result;
}

/**
 * impliedVolatility(marketPrice, S, K, T, r, type)
 */
napi_value ImpliedVolatility(napi_env env, napi_callback_info info) {
    size_t argc = 6;
    napi_value args[6];
    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, args, nullptr, nullptr));

    double marketPrice, S, K, T, r;
    napi_get_value_double(env, args[0], &marketPrice);
    napi_get_value_double(env, args[1], &S);
    napi_get_value_double(env, args[2], &K);
    napi_get_value_double(env, args[3], &T);
    napi_get_value_double(env, args[4], &r);
    
    std::string type_str = get_string(env, args[5]);
    auto type = (type_str == "PUT") ? options::OptionType::PUT : options::OptionType::CALL;

    auto start = std::chrono::steady_clock::now();
    double iv = options::BlackScholesPricer::implied_volatility(marketPrice, S, K, T, r, type);
    auto end = std::chrono::steady_clock::now();
    
    int64_t latency_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    napi_value result = create_result_object(env);
    set_property_double(env, result, "iv", iv);
    set_property_int64(env, result, "latencyNs", latency_ns);
    set_property_bool(env, result, "converged", iv > 0.001 && iv < 5.0);

    return result;
}

//=============================================================================
// MONTE CARLO BINDINGS
//=============================================================================

/**
 * runMonteCarloAsync(params: { S0, mu, sigma, T, numPaths, numSteps, seed })
 * Returns Promise<SimulationResult>
 */

struct MonteCarloWork {
    napi_async_work work;
    napi_deferred deferred;
    napi_ref callback_ref;
    
    // Input
    montecarlo::SimulationParams params;
    
    // Output
    montecarlo::SimulationResult result;
    int64_t latency_ms;
};

void MonteCarloExecute(napi_env env, void* data) {
    auto* work = static_cast<MonteCarloWork*>(data);
    
    auto start = std::chrono::steady_clock::now();
    
    montecarlo::MonteCarloEngine engine(std::thread::hardware_concurrency());
    work->result = engine.simulate_gbm(work->params);
    
    auto end = std::chrono::steady_clock::now();
    work->latency_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
}

void MonteCarloComplete(napi_env env, napi_status status, void* data) {
    auto* work = static_cast<MonteCarloWork*>(data);
    
    napi_value result = create_result_object(env);
    
    // Stats
    napi_value stats = create_result_object(env);
    set_property_double(env, stats, "meanFinalPrice", work->result.stats.mean_final_price);
    set_property_double(env, stats, "stdFinalPrice", work->result.stats.std_final_price);
    set_property_double(env, stats, "minFinalPrice", work->result.stats.min_final_price);
    set_property_double(env, stats, "maxFinalPrice", work->result.stats.max_final_price);
    napi_set_named_property(env, result, "stats", stats);
    
    // Final prices array
    napi_value final_prices;
    napi_create_array_with_length(env, work->result.final_prices.size(), &final_prices);
    for (size_t i = 0; i < work->result.final_prices.size(); i++) {
        napi_value price;
        napi_create_double(env, work->result.final_prices[i], &price);
        napi_set_element(env, final_prices, i, price);
    }
    napi_set_named_property(env, result, "finalPrices", final_prices);
    
    // Sample paths (first 100)
    napi_value paths;
    size_t num_paths = std::min(work->result.paths.size(), size_t(100));
    napi_create_array_with_length(env, num_paths, &paths);
    for (size_t i = 0; i < num_paths; i++) {
        napi_value path;
        napi_create_array_with_length(env, work->result.paths[i].size(), &path);
        for (size_t j = 0; j < work->result.paths[i].size(); j++) {
            napi_value val;
            napi_create_double(env, work->result.paths[i][j], &val);
            napi_set_element(env, path, j, val);
        }
        napi_set_element(env, paths, i, path);
    }
    napi_set_named_property(env, result, "paths", paths);
    
    set_property_int64(env, result, "calculationTimeMs", work->latency_ms);
    
    napi_resolve_deferred(env, work->deferred, result);
    napi_delete_async_work(env, work->work);
    delete work;
}

napi_value RunMonteCarlo(napi_env env, napi_callback_info info) {
    size_t argc = 1;
    napi_value args[1];
    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, args, nullptr, nullptr));

    auto* work = new MonteCarloWork();
    
    // Parse params object
    napi_value param_obj = args[0];
    napi_value temp;
    
    napi_get_named_property(env, param_obj, "S0", &temp);
    napi_get_value_double(env, temp, &work->params.S0);
    
    napi_get_named_property(env, param_obj, "mu", &temp);
    napi_get_value_double(env, temp, &work->params.mu);
    
    napi_get_named_property(env, param_obj, "sigma", &temp);
    napi_get_value_double(env, temp, &work->params.sigma);
    
    napi_get_named_property(env, param_obj, "T", &temp);
    napi_get_value_double(env, temp, &work->params.T);
    
    int32_t num_paths, num_steps;
    napi_get_named_property(env, param_obj, "numPaths", &temp);
    napi_get_value_int32(env, temp, &num_paths);
    work->params.num_paths = num_paths;
    
    napi_get_named_property(env, param_obj, "numSteps", &temp);
    napi_get_value_int32(env, temp, &num_steps);
    work->params.num_steps = num_steps;
    
    work->params.dt = work->params.T / work->params.num_steps;
    work->params.seed = std::chrono::steady_clock::now().time_since_epoch().count();
    
    // Create promise
    napi_value promise;
    napi_create_promise(env, &work->deferred, &promise);
    
    // Create async work
    napi_value resource_name;
    napi_create_string_utf8(env, "MonteCarloSimulation", NAPI_AUTO_LENGTH, &resource_name);
    napi_create_async_work(env, nullptr, resource_name,
                           MonteCarloExecute, MonteCarloComplete, work, &work->work);
    napi_queue_async_work(env, work->work);
    
    return promise;
}

//=============================================================================
// RISK MANAGEMENT BINDINGS
//=============================================================================

/**
 * initRiskManager(limits: RiskLimits)
 */
napi_value InitRiskManager(napi_env env, napi_callback_info info) {
    size_t argc = 1;
    napi_value args[1];
    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, args, nullptr, nullptr));

    risk::RiskLimits limits;
    napi_value temp;
    
    if (argc > 0) {
        napi_get_named_property(env, args[0], "maxPositionQty", &temp);
        int64_t val;
        if (napi_get_value_int64(env, temp, &val) == napi_ok) {
            limits.max_position_qty = val;
        }
        
        napi_get_named_property(env, args[0], "maxOrderQty", &temp);
        int32_t val32;
        if (napi_get_value_int32(env, temp, &val32) == napi_ok) {
            limits.max_order_qty = val32;
        }
        
        napi_get_named_property(env, args[0], "maxDailyLoss", &temp);
        if (napi_get_value_int64(env, temp, &val) == napi_ok) {
            limits.max_daily_loss = val;
        }
    }
    
    risk_manager = std::make_unique<risk::RiskManager>(limits);
    sequencer = std::make_unique<risk::OrderSequencer>();
    
    napi_value result;
    napi_get_boolean(env, true, &result);
    return result;
}

/**
 * checkOrderRisk(symbol, side, quantity, price, marketPrice)
 */
napi_value CheckOrderRisk(napi_env env, napi_callback_info info) {
    size_t argc = 5;
    napi_value args[5];
    NAPI_CALL(env, napi_get_cb_info(env, info, &argc, args, nullptr, nullptr));
    NAPI_ASSERT(env, risk_manager != nullptr, "Risk manager not initialized");
    
    std::string symbol = get_string(env, args[0]);
    std::string side_str = get_string(env, args[1]);
    
    int64_t quantity, price_ticks, market_price_ticks;
    napi_get_value_int64(env, args[2], &quantity);
    
    double price_d, market_d;
    napi_get_value_double(env, args[3], &price_d);
    napi_get_value_double(env, args[4], &market_d);
    price_ticks = risk::to_fixed(price_d);
    market_price_ticks = risk::to_fixed(market_d);
    
    char side = (side_str == "BUY") ? 'B' : 'S';
    
    auto check = risk_manager->check_order(
        symbol.c_str(), side, quantity, price_ticks, market_price_ticks);
    
    napi_value result = create_result_object(env);
    set_property_bool(env, result, "passed", check.passed);
    set_property_int64(env, result, "rejectCode", check.reject_code);
    set_property_string(env, result, "rejectReason", check.reject_reason);
    set_property_int64(env, result, "checkLatencyNs", check.check_latency_ns);
    
    return result;
}

//=============================================================================
// MODULE INITIALIZATION
//=============================================================================

napi_value Init(napi_env env, napi_value exports) {
    // Order book functions
    napi_value fn;
    
    napi_create_function(env, nullptr, 0, CreateOrderBook, nullptr, &fn);
    napi_set_named_property(env, exports, "createOrderBook", fn);
    
    napi_create_function(env, nullptr, 0, AddOrder, nullptr, &fn);
    napi_set_named_property(env, exports, "addOrder", fn);
    
    napi_create_function(env, nullptr, 0, GetOrderBookSnapshot, nullptr, &fn);
    napi_set_named_property(env, exports, "getOrderBookSnapshot", fn);
    
    // Options pricing functions
    napi_create_function(env, nullptr, 0, PriceOption, nullptr, &fn);
    napi_set_named_property(env, exports, "priceOption", fn);
    
    napi_create_function(env, nullptr, 0, ImpliedVolatility, nullptr, &fn);
    napi_set_named_property(env, exports, "impliedVolatility", fn);
    
    // Monte Carlo functions
    napi_create_function(env, nullptr, 0, RunMonteCarlo, nullptr, &fn);
    napi_set_named_property(env, exports, "runMonteCarlo", fn);
    
    // Risk management functions
    napi_create_function(env, nullptr, 0, InitRiskManager, nullptr, &fn);
    napi_set_named_property(env, exports, "initRiskManager", fn);
    
    napi_create_function(env, nullptr, 0, CheckOrderRisk, nullptr, &fn);
    napi_set_named_property(env, exports, "checkOrderRisk", fn);
    
    return exports;
}

NAPI_MODULE(NODE_GYP_MODULE_NAME, Init)

} // namespace arbor::bindings
