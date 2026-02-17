import pino from 'pino';
import { getConfig } from './config.js';

const VERSION = '1.0.0';

export function createLogger(component: string) {
  const config = getConfig();
  return pino({
    name: component,
    level: config.logLevel,
    formatters: {
      level(label) {
        return { level: label };
      },
    },
    base: { component, version: VERSION },
  });
}

export type Logger = ReturnType<typeof createLogger>;
