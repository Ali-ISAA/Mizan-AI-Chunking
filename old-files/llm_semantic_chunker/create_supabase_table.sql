-- SQL Script to Create Document Table in Supabase
-- Run this in your Supabase SQL Editor before using the Python script

-- Replace 'doc_your_document_name' with your actual table name
-- The Python script will tell you the exact table name to use

-- Example: If processing "Digital Government Policies - V2.0.pdf_processed.md"
-- Table name will be: doc_digital_government_policies_v2_0_pdf_processed

CREATE TABLE IF NOT EXISTS doc_digital_government_policies_v2_0_pdf_processed (
  id bigserial primary key,
  content text,
  metadata jsonb,
  embedding vector(768)
);

-- Create index for vector similarity search
CREATE INDEX IF NOT EXISTS doc_digital_government_policies_v2_0_pdf_processed_embedding_idx
ON doc_digital_government_policies_v2_0_pdf_processed
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Test the table
SELECT COUNT(*) FROM doc_digital_government_policies_v2_0_pdf_processed;

-- View table structure
SELECT column_name, data_type
FROM information_schema.columns
WHERE table_name = 'doc_digital_government_policies_v2_0_pdf_processed';
