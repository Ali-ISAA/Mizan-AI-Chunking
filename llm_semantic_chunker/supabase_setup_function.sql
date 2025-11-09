-- Run this SQL ONCE in Supabase SQL Editor to enable automatic table creation
-- This creates a function that allows the Python script to execute DDL commands

-- Drop the function if it already exists
DROP FUNCTION IF EXISTS create_document_table(text);

-- Create the function
CREATE OR REPLACE FUNCTION create_document_table(table_name text)
RETURNS void
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
BEGIN
  -- Create table with dynamic name
  EXECUTE format('
    CREATE TABLE IF NOT EXISTS %I (
      id bigserial primary key,
      content text,
      metadata jsonb,
      embedding vector(768)
    )', table_name);

  -- Create index for vector similarity search
  EXECUTE format('
    CREATE INDEX IF NOT EXISTS %I
    ON %I
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100)',
    table_name || '_embedding_idx',
    table_name
  );
END;
$$;

-- Grant execute permission to authenticated and service role
GRANT EXECUTE ON FUNCTION create_document_table(text) TO authenticated;
GRANT EXECUTE ON FUNCTION create_document_table(text) TO service_role;
GRANT EXECUTE ON FUNCTION create_document_table(text) TO anon;

-- Test the function (optional - creates a test table)
-- SELECT create_document_table('test_table');
-- DROP TABLE IF EXISTS test_table;
