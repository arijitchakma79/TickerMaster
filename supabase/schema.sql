-- ============================================
-- TickerMaster Complete Database Schema
-- Run in Supabase SQL Editor
-- ============================================

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE IF NOT EXISTS public.profiles (
    id UUID REFERENCES auth.users(id) ON DELETE CASCADE PRIMARY KEY,
    email TEXT,
    display_name TEXT,
    avatar_url TEXT,
    phone_number TEXT,
    poke_enabled BOOLEAN DEFAULT FALSE,
    tutorial_completed BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

ALTER TABLE IF EXISTS public.profiles
    ADD COLUMN IF NOT EXISTS phone_number TEXT;

CREATE TABLE IF NOT EXISTS public.notification_preferences (
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE PRIMARY KEY,
    phone_number TEXT,
    email TEXT,
    preferred_channel TEXT NOT NULL DEFAULT 'push' CHECK (preferred_channel IN ('sms', 'email', 'push')),
    alert_frequency TEXT NOT NULL DEFAULT 'realtime' CHECK (alert_frequency IN ('realtime', 'hourly', 'daily')),
    price_alerts BOOLEAN NOT NULL DEFAULT TRUE,
    volume_alerts BOOLEAN NOT NULL DEFAULT TRUE,
    simulation_summary BOOLEAN NOT NULL DEFAULT TRUE,
    quiet_start TIME NOT NULL DEFAULT TIME '22:00',
    quiet_end TIME NOT NULL DEFAULT TIME '07:00',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO public.profiles (id, email, display_name, avatar_url)
    VALUES (
        NEW.id,
        NEW.email,
        COALESCE(NEW.raw_user_meta_data->>'full_name', NEW.email),
        NEW.raw_user_meta_data->>'avatar_url'
    )
    ON CONFLICT (id) DO NOTHING;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;
CREATE TRIGGER on_auth_user_created
    AFTER INSERT ON auth.users
    FOR EACH ROW EXECUTE FUNCTION public.handle_new_user();

CREATE TABLE IF NOT EXISTS public.tracker_agents (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL,
    symbol TEXT NOT NULL,
    name TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'paused', 'deleted')),
    triggers JSONB NOT NULL DEFAULT '{}',
    auto_simulate BOOLEAN DEFAULT FALSE,
    last_alert_at TIMESTAMPTZ,
    total_alerts INTEGER DEFAULT 0,
    last_checked_at TIMESTAMPTZ,
    last_price NUMERIC,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_tracker_agents_user ON public.tracker_agents (user_id);
CREATE INDEX IF NOT EXISTS idx_tracker_agents_active ON public.tracker_agents (status) WHERE status = 'active';
CREATE INDEX IF NOT EXISTS idx_tracker_agents_symbol ON public.tracker_agents (symbol);

CREATE TABLE IF NOT EXISTS public.tracker_alerts (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    agent_id UUID REFERENCES public.tracker_agents(id) ON DELETE CASCADE,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    symbol TEXT NOT NULL,
    trigger_reason TEXT NOT NULL,
    narrative TEXT,
    market_snapshot JSONB,
    investigation_data JSONB,
    poke_sent BOOLEAN DEFAULT FALSE,
    simulation_id TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_tracker_alerts_agent ON public.tracker_alerts (agent_id);
CREATE INDEX IF NOT EXISTS idx_tracker_alerts_user ON public.tracker_alerts (user_id);
CREATE INDEX IF NOT EXISTS idx_tracker_alerts_symbol ON public.tracker_alerts (symbol);
CREATE INDEX IF NOT EXISTS idx_tracker_alerts_created ON public.tracker_alerts (created_at DESC);

CREATE TABLE IF NOT EXISTS public.tracker_alert_context (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    agent_id UUID REFERENCES public.tracker_agents(id) ON DELETE CASCADE NOT NULL,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL,
    alert_id UUID REFERENCES public.tracker_alerts(id) ON DELETE SET NULL,
    symbol TEXT NOT NULL,
    event_type TEXT NOT NULL CHECK (event_type IN ('alert', 'report', 'report_skipped', 'noop')),
    context_summary TEXT,
    context_payload JSONB NOT NULL DEFAULT '{}',
    simulation_requested BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_tracker_alert_context_agent ON public.tracker_alert_context (agent_id);
CREATE INDEX IF NOT EXISTS idx_tracker_alert_context_user ON public.tracker_alert_context (user_id);
CREATE INDEX IF NOT EXISTS idx_tracker_alert_context_alert ON public.tracker_alert_context (alert_id);
CREATE INDEX IF NOT EXISTS idx_tracker_alert_context_created ON public.tracker_alert_context (created_at DESC);

CREATE TABLE IF NOT EXISTS public.tracker_agent_history (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    agent_id UUID REFERENCES public.tracker_agents(id) ON DELETE CASCADE,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL,
    event_type TEXT NOT NULL CHECK (event_type IN ('create_prompt', 'manager_instruction', 'system_update', 'agent_response')),
    raw_prompt TEXT,
    parsed_intent JSONB,
    trigger_snapshot JSONB,
    tool_outputs JSONB,
    note TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_tracker_agent_history_agent ON public.tracker_agent_history (agent_id);
CREATE INDEX IF NOT EXISTS idx_tracker_agent_history_user ON public.tracker_agent_history (user_id);
CREATE INDEX IF NOT EXISTS idx_tracker_agent_history_created ON public.tracker_agent_history (created_at DESC);

CREATE TABLE IF NOT EXISTS public.tracker_agent_runs (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    agent_id UUID REFERENCES public.tracker_agents(id) ON DELETE CASCADE,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL,
    symbol TEXT NOT NULL,
    run_type TEXT NOT NULL CHECK (run_type IN ('noop', 'alert', 'report', 'report_skipped')),
    trigger_reasons JSONB NOT NULL DEFAULT '[]',
    tools_used JSONB NOT NULL DEFAULT '[]',
    research_sources JSONB NOT NULL DEFAULT '[]',
    market_snapshot JSONB NOT NULL DEFAULT '{}',
    research_snapshot JSONB NOT NULL DEFAULT '{}',
    simulation_snapshot JSONB NOT NULL DEFAULT '{}',
    decision JSONB NOT NULL DEFAULT '{}',
    note TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_tracker_agent_runs_agent ON public.tracker_agent_runs (agent_id);
CREATE INDEX IF NOT EXISTS idx_tracker_agent_runs_user ON public.tracker_agent_runs (user_id);
CREATE INDEX IF NOT EXISTS idx_tracker_agent_runs_created ON public.tracker_agent_runs (created_at DESC);

CREATE TABLE IF NOT EXISTS public.tracker_agent_thesis (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    agent_id UUID REFERENCES public.tracker_agents(id) ON DELETE CASCADE UNIQUE,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL,
    symbol TEXT NOT NULL,
    stance_score NUMERIC NOT NULL DEFAULT 0,
    confidence NUMERIC NOT NULL DEFAULT 0.5,
    thesis JSONB NOT NULL DEFAULT '{}',
    summary TEXT,
    last_event_type TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_tracker_agent_thesis_agent ON public.tracker_agent_thesis (agent_id);
CREATE INDEX IF NOT EXISTS idx_tracker_agent_thesis_user ON public.tracker_agent_thesis (user_id);

CREATE TABLE IF NOT EXISTS public.simulations (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    config JSONB NOT NULL,
    status TEXT NOT NULL DEFAULT 'running' CHECK (status IN ('running', 'completed', 'failed')),
    results JSONB,
    modal_sandbox_id TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_simulations_user ON public.simulations (user_id);
CREATE INDEX IF NOT EXISTS idx_simulations_status ON public.simulations (status);
CREATE INDEX IF NOT EXISTS idx_simulations_created ON public.simulations (created_at DESC);

CREATE TABLE IF NOT EXISTS public.agent_activity (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    module TEXT NOT NULL CHECK (module IN ('research', 'simulation', 'tracker')),
    agent_name TEXT NOT NULL,
    action TEXT NOT NULL,
    details JSONB,
    status TEXT DEFAULT 'success' CHECK (status IN ('success', 'error', 'pending', 'running')),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_agent_activity_user ON public.agent_activity (user_id);
CREATE INDEX IF NOT EXISTS idx_agent_activity_created ON public.agent_activity (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_agent_activity_module ON public.agent_activity (module);

CREATE TABLE IF NOT EXISTS public.research_cache (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    symbol TEXT NOT NULL,
    data_type TEXT NOT NULL,
    data JSONB NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_research_cache_lookup ON public.research_cache (symbol, data_type);

CREATE TABLE IF NOT EXISTS public.watchlist (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL,
    symbol TEXT NOT NULL,
    added_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(user_id, symbol)
);

CREATE TABLE IF NOT EXISTS public.favorite_stocks (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL,
    symbol TEXT NOT NULL,
    added_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(user_id, symbol)
);

CREATE TABLE IF NOT EXISTS public.simulation_agents (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL,
    agent_name TEXT NOT NULL,
    config JSONB NOT NULL,
    icon_emoji TEXT,
    editor JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(user_id, agent_name)
);

ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.tracker_agents ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.tracker_alerts ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.tracker_alert_context ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.tracker_agent_history ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.tracker_agent_runs ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.tracker_agent_thesis ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.simulations ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.agent_activity ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.research_cache ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.watchlist ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.favorite_stocks ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.notification_preferences ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.simulation_agents ENABLE ROW LEVEL SECURITY;

DO $$ BEGIN
    CREATE POLICY "Users can view own profile" ON public.profiles
        FOR SELECT USING (auth.uid() = id);
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE POLICY "Users can update own profile" ON public.profiles
        FOR UPDATE USING (auth.uid() = id);
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE POLICY "Users can CRUD own tracker agents" ON public.tracker_agents
        FOR ALL USING (auth.uid() = user_id);
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE POLICY "Users can view own alerts" ON public.tracker_alerts
        FOR SELECT USING (auth.uid() = user_id);
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE POLICY "Service can insert alerts" ON public.tracker_alerts
        FOR INSERT TO service_role WITH CHECK (true);
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE POLICY "Users can CRUD own tracker alert context" ON public.tracker_alert_context
        FOR ALL USING (auth.uid() = user_id) WITH CHECK (auth.uid() = user_id);
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE POLICY "Service can insert tracker alert context" ON public.tracker_alert_context
        FOR INSERT TO service_role WITH CHECK (true);
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE POLICY "Users can CRUD own tracker history" ON public.tracker_agent_history
        FOR ALL USING (auth.uid() = user_id) WITH CHECK (auth.uid() = user_id);
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE POLICY "Service can insert tracker history" ON public.tracker_agent_history
        FOR INSERT TO service_role WITH CHECK (true);
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE POLICY "Users can CRUD own tracker runs" ON public.tracker_agent_runs
        FOR ALL USING (auth.uid() = user_id) WITH CHECK (auth.uid() = user_id);
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE POLICY "Service can insert tracker runs" ON public.tracker_agent_runs
        FOR INSERT TO service_role WITH CHECK (true);
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE POLICY "Users can CRUD own tracker thesis" ON public.tracker_agent_thesis
        FOR ALL USING (auth.uid() = user_id) WITH CHECK (auth.uid() = user_id);
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE POLICY "Service can upsert tracker thesis" ON public.tracker_agent_thesis
        FOR INSERT TO service_role WITH CHECK (true);
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE POLICY "Service can update tracker thesis" ON public.tracker_agent_thesis
        FOR UPDATE TO service_role USING (true);
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE POLICY "Users can CRUD own simulations" ON public.simulations
        FOR ALL USING (auth.uid() = user_id);
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE POLICY "Users can view own activity" ON public.agent_activity
        FOR SELECT USING (auth.uid() = user_id);
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE POLICY "Service can insert activity" ON public.agent_activity
        FOR INSERT TO service_role WITH CHECK (true);
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE POLICY "Users can CRUD own watchlist" ON public.watchlist
        FOR ALL USING (auth.uid() = user_id);
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE POLICY "Users can CRUD own favorites" ON public.favorite_stocks
        FOR ALL USING (auth.uid() = user_id);
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE POLICY "Users can CRUD own notification preferences" ON public.notification_preferences
        FOR ALL USING (auth.uid() = user_id) WITH CHECK (auth.uid() = user_id);
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE POLICY "Users can CRUD own simulation agents" ON public.simulation_agents
        FOR ALL USING (auth.uid() = user_id);
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE POLICY "Anyone can read cache" ON public.research_cache
        FOR SELECT USING (true);
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE POLICY "Service can write cache" ON public.research_cache
        FOR INSERT TO service_role WITH CHECK (true);
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE POLICY "Service can update cache" ON public.research_cache
        FOR UPDATE TO service_role USING (true);
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

CREATE OR REPLACE FUNCTION public.touch_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS profiles_touch_updated_at ON public.profiles;
CREATE TRIGGER profiles_touch_updated_at
    BEFORE UPDATE ON public.profiles
    FOR EACH ROW EXECUTE FUNCTION public.touch_updated_at();

DROP TRIGGER IF EXISTS tracker_agents_touch_updated_at ON public.tracker_agents;
CREATE TRIGGER tracker_agents_touch_updated_at
    BEFORE UPDATE ON public.tracker_agents
    FOR EACH ROW EXECUTE FUNCTION public.touch_updated_at();

DROP TRIGGER IF EXISTS simulation_agents_touch_updated_at ON public.simulation_agents;
CREATE TRIGGER simulation_agents_touch_updated_at
    BEFORE UPDATE ON public.simulation_agents
    FOR EACH ROW EXECUTE FUNCTION public.touch_updated_at();

DO $$ BEGIN
    INSERT INTO storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
    VALUES ('tracker-exports', 'tracker-exports', false, 20000000, ARRAY['text/csv'])
    ON CONFLICT (id) DO NOTHING;
EXCEPTION WHEN undefined_table THEN NULL; END $$;

DO $$ BEGIN
    INSERT INTO storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
    VALUES ('tracker-memory', 'tracker-memory', false, 20000000, ARRAY['text/csv'])
    ON CONFLICT (id) DO NOTHING;
EXCEPTION WHEN undefined_table THEN NULL; END $$;
