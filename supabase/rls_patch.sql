-- Allow anon/authenticated API key inserts for backend-service style writes
-- when service-role key is not used in development.

DO $$
BEGIN
  IF to_regclass('public.agent_activity') IS NULL THEN
    RAISE NOTICE 'Skipping agent_activity policy patch because table does not exist yet.';
  ELSE
    DROP POLICY IF EXISTS "Service can insert activity" ON public.agent_activity;
    CREATE POLICY "Service can insert activity"
    ON public.agent_activity
    FOR INSERT
    TO anon, authenticated
    WITH CHECK (true);
  END IF;
END $$;

DO $$
BEGIN
  IF to_regclass('public.tracker_alerts') IS NULL THEN
    RAISE NOTICE 'Skipping tracker_alerts policy patch because table does not exist yet.';
  ELSE
    DROP POLICY IF EXISTS "Service can insert alerts" ON public.tracker_alerts;
    CREATE POLICY "Service can insert alerts"
    ON public.tracker_alerts
    FOR INSERT
    TO anon, authenticated
    WITH CHECK (true);
  END IF;
END $$;

DO $$
BEGIN
  IF to_regclass('public.tracker_alert_context') IS NULL THEN
    RAISE NOTICE 'Skipping tracker_alert_context policy patch because table does not exist yet.';
  ELSE
    DROP POLICY IF EXISTS "Service can insert tracker alert context" ON public.tracker_alert_context;
    CREATE POLICY "Service can insert tracker alert context"
    ON public.tracker_alert_context
    FOR INSERT
    TO anon, authenticated
    WITH CHECK (true);
  END IF;
END $$;

DO $$
BEGIN
  IF to_regclass('public.tracker_agent_runs') IS NULL THEN
    RAISE NOTICE 'Skipping tracker_agent_runs policy patch because table does not exist yet.';
  ELSE
    DROP POLICY IF EXISTS "Service can insert tracker runs" ON public.tracker_agent_runs;
    CREATE POLICY "Service can insert tracker runs"
    ON public.tracker_agent_runs
    FOR INSERT
    TO anon, authenticated
    WITH CHECK (true);
  END IF;
END $$;

DO $$
BEGIN
  IF to_regclass('public.tracker_agent_history') IS NULL THEN
    RAISE NOTICE 'Skipping tracker_agent_history policy patch because table does not exist yet.';
  ELSE
    DROP POLICY IF EXISTS "Service can insert tracker history" ON public.tracker_agent_history;
    CREATE POLICY "Service can insert tracker history"
    ON public.tracker_agent_history
    FOR INSERT
    TO anon, authenticated
    WITH CHECK (true);
  END IF;
END $$;

DO $$
BEGIN
  IF to_regclass('public.tracker_agent_thesis') IS NULL THEN
    RAISE NOTICE 'Skipping tracker_agent_thesis policy patch because table does not exist yet.';
  ELSE
    DROP POLICY IF EXISTS "Service can upsert tracker thesis" ON public.tracker_agent_thesis;
    CREATE POLICY "Service can upsert tracker thesis"
    ON public.tracker_agent_thesis
    FOR INSERT
    TO anon, authenticated
    WITH CHECK (true);

    DROP POLICY IF EXISTS "Service can update tracker thesis" ON public.tracker_agent_thesis;
    CREATE POLICY "Service can update tracker thesis"
    ON public.tracker_agent_thesis
    FOR UPDATE
    TO anon, authenticated
    USING (true);
  END IF;
END $$;

DO $$
BEGIN
  IF to_regclass('public.research_cache') IS NULL THEN
    RAISE NOTICE 'Skipping research_cache policy patch because table does not exist yet.';
  ELSE
    DROP POLICY IF EXISTS "Service can write cache" ON public.research_cache;
    CREATE POLICY "Service can write cache"
    ON public.research_cache
    FOR INSERT
    TO anon, authenticated
    WITH CHECK (true);

    DROP POLICY IF EXISTS "Service can update cache" ON public.research_cache;
    CREATE POLICY "Service can update cache"
    ON public.research_cache
    FOR UPDATE
    TO anon, authenticated
    USING (true);
  END IF;
END $$;
