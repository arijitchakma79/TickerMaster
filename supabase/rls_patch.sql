-- Allow anon/authenticated API key inserts for backend-service style writes
-- when service-role key is not used in development.

DO $$ BEGIN
  DROP POLICY IF EXISTS "Service can insert activity" ON public.agent_activity;
EXCEPTION WHEN undefined_object THEN NULL; END $$;

CREATE POLICY "Service can insert activity"
ON public.agent_activity
FOR INSERT
TO anon, authenticated
WITH CHECK (true);

DO $$ BEGIN
  DROP POLICY IF EXISTS "Service can insert alerts" ON public.tracker_alerts;
EXCEPTION WHEN undefined_object THEN NULL; END $$;

CREATE POLICY "Service can insert alerts"
ON public.tracker_alerts
FOR INSERT
TO anon, authenticated
WITH CHECK (true);
