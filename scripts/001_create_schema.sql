-- MarketProphet Database Schema
-- Profiles table for user data
CREATE TABLE IF NOT EXISTS public.profiles (
  id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
  email TEXT,
  full_name TEXT,
  avatar_url TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "profiles_select_own" ON public.profiles;
DROP POLICY IF EXISTS "profiles_insert_own" ON public.profiles;
DROP POLICY IF EXISTS "profiles_update_own" ON public.profiles;
CREATE POLICY "profiles_select_own" ON public.profiles FOR SELECT USING (auth.uid() = id);
CREATE POLICY "profiles_insert_own" ON public.profiles FOR INSERT WITH CHECK (auth.uid() = id);
CREATE POLICY "profiles_update_own" ON public.profiles FOR UPDATE USING (auth.uid() = id);

-- Portfolio holdings table
CREATE TABLE IF NOT EXISTS public.holdings (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  symbol TEXT NOT NULL,
  shares DECIMAL(18, 8) NOT NULL,
  avg_cost_basis DECIMAL(18, 4) NOT NULL,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE(user_id, symbol)
);

ALTER TABLE public.holdings ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "holdings_select_own" ON public.holdings;
DROP POLICY IF EXISTS "holdings_insert_own" ON public.holdings;
DROP POLICY IF EXISTS "holdings_update_own" ON public.holdings;
DROP POLICY IF EXISTS "holdings_delete_own" ON public.holdings;
CREATE POLICY "holdings_select_own" ON public.holdings FOR SELECT USING (auth.uid() = user_id);
CREATE POLICY "holdings_insert_own" ON public.holdings FOR INSERT WITH CHECK (auth.uid() = user_id);
CREATE POLICY "holdings_update_own" ON public.holdings FOR UPDATE USING (auth.uid() = user_id);
CREATE POLICY "holdings_delete_own" ON public.holdings FOR DELETE USING (auth.uid() = user_id);

-- Transactions table for buy/sell history
CREATE TABLE IF NOT EXISTS public.transactions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  symbol TEXT NOT NULL,
  type TEXT NOT NULL CHECK (type IN ('BUY', 'SELL')),
  shares DECIMAL(18, 8) NOT NULL,
  price_per_share DECIMAL(18, 4) NOT NULL,
  total_amount DECIMAL(18, 4) NOT NULL,
  executed_at TIMESTAMPTZ DEFAULT NOW(),
  notes TEXT
);

ALTER TABLE public.transactions ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "transactions_select_own" ON public.transactions;
DROP POLICY IF EXISTS "transactions_insert_own" ON public.transactions;
CREATE POLICY "transactions_select_own" ON public.transactions FOR SELECT USING (auth.uid() = user_id);
CREATE POLICY "transactions_insert_own" ON public.transactions FOR INSERT WITH CHECK (auth.uid() = user_id);

-- Watchlist table
CREATE TABLE IF NOT EXISTS public.watchlist (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  symbol TEXT NOT NULL,
  added_at TIMESTAMPTZ DEFAULT NOW(),
  notes TEXT,
  UNIQUE(user_id, symbol)
);

ALTER TABLE public.watchlist ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "watchlist_select_own" ON public.watchlist;
DROP POLICY IF EXISTS "watchlist_insert_own" ON public.watchlist;
DROP POLICY IF EXISTS "watchlist_delete_own" ON public.watchlist;
CREATE POLICY "watchlist_select_own" ON public.watchlist FOR SELECT USING (auth.uid() = user_id);
CREATE POLICY "watchlist_insert_own" ON public.watchlist FOR INSERT WITH CHECK (auth.uid() = user_id);
CREATE POLICY "watchlist_delete_own" ON public.watchlist FOR DELETE USING (auth.uid() = user_id);

-- AI Analysis history table
CREATE TABLE IF NOT EXISTS public.analysis_history (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  symbol TEXT NOT NULL,
  query TEXT NOT NULL,
  recommendation TEXT,
  confidence DECIMAL(5, 2),
  research_summary TEXT,
  technical_summary TEXT,
  sentiment_summary TEXT,
  risk_summary TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

ALTER TABLE public.analysis_history ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "analysis_select_own" ON public.analysis_history;
DROP POLICY IF EXISTS "analysis_insert_own" ON public.analysis_history;
CREATE POLICY "analysis_select_own" ON public.analysis_history FOR SELECT USING (auth.uid() = user_id);
CREATE POLICY "analysis_insert_own" ON public.analysis_history FOR INSERT WITH CHECK (auth.uid() = user_id);
