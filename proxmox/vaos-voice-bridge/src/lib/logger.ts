import pino from 'pino';
import { getEnv } from './config.js';

export function createLogger(component: string) {
  const env = getEnv();

  return pino({
    name: component,
    level: env.LOG_LEVEL,
    formatters: {
      level(label) {
        return { level: label };
      },
    },
    base: {
      component,
      version: '1.0.0',
    },
  });
}

export type Logger = ReturnType<typeof createLogger>;
