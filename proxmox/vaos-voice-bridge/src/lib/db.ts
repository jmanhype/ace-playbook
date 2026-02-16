import { createClient, type SupabaseClient } from '@supabase/supabase-js';
import { getEnv } from './config.js';

export class SupabaseError extends Error {
  constructor(message: string, public code?: string) {
    super(message);
    this.name = 'SupabaseError';
  }
}

let client: SupabaseClient | null = null;

export function getSupabase(): SupabaseClient {
  if (!client) {
    const env = getEnv();
    client = createClient(env.SUPABASE_URL, env.SUPABASE_SERVICE_ROLE_KEY);
  }
  return client;
}

/**
 * Safe Supabase query wrapper — returns real Promises, not thenables.
 * Prevents the v1 thenable crash bug.
 */
export async function sb<T = any>(query: PromiseLike<{ data: any; error: any }>): Promise<T> {
  let result: { data: any; error: any };
  try {
    result = await query;
  } catch (e) {
    throw new SupabaseError(`Query execution failed: ${e}`, 'QUERY_EXEC');
  }
  if (result.error) {
    throw new SupabaseError(result.error.message, result.error.code);
  }
  return result.data as T;
}

export async function sbVoid(query: PromiseLike<{ data: any; error: any }>): Promise<void> {
  let result: { data: any; error: any };
  try {
    result = await query;
  } catch (e) {
    throw new SupabaseError(`Query execution failed: ${e}`, 'QUERY_EXEC');
  }
  if (result.error) {
    throw new SupabaseError(result.error.message, result.error.code);
  }
}
