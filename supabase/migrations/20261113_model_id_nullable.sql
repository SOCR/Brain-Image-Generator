-- Run after initial_schema.sql.
-- The back end records every signed-in generation in generated_images with
-- model_id = '00000000-0000-0000-0000-000000000000', and ml_models has no rows, so the NOT NULL
-- foreign key created by initial_schema.sql rejects every such insert (the files upload, but no
-- row is written and the Library stays empty). The production database already has model_id
-- nullable without the foreign key; this migration makes a new project match it.
ALTER TABLE public.generated_images DROP CONSTRAINT IF EXISTS generated_images_model_id_fkey;
ALTER TABLE public.generated_images ALTER COLUMN model_id DROP NOT NULL;
