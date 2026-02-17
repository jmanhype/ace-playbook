export interface Config {
  port: number;
  personaplex: {
    host: string;
    port: number;
    wsUrl: string;
    sshUser: string;
    venvPath: string;
    defaultVoice: string;
  };
  letta: {
    baseUrl: string;
    token: string;
    agentName: string;
    agentId: string | null;
  };
  routing: {
    complexityThreshold: number;
    agentTimeoutMs: number;
  };
  logLevel: string;
}

export function getConfig(): Config {
  const pxHost = process.env.PERSONAPLEX_HOST || '192.168.1.143';
  const pxPort = parseInt(process.env.PERSONAPLEX_PORT || '8998', 10);

  return {
    port: parseInt(process.env.PORT || '9001', 10),
    personaplex: {
      host: pxHost,
      port: pxPort,
      wsUrl: `ws://${pxHost}:${pxPort}/api/chat`,
      sshUser: process.env.PERSONAPLEX_SSH_USER || 'straughter',
      venvPath: process.env.PERSONAPLEX_VENV || '/home/straughter/personaplex/venv',
      defaultVoice: process.env.DEFAULT_VOICE || 'NATF0.pt',
    },
    letta: {
      baseUrl: process.env.LETTA_BASE_URL || `http://${pxHost}:8283`,
      token: process.env.LETTA_TOKEN || 'e9acq0WgooNgt5ncWWSJEQ',
      agentName: process.env.LETTA_AGENT_NAME || 'voice-reasoner',
      agentId: process.env.LETTA_AGENT_ID || null,
    },
    routing: {
      complexityThreshold: parseFloat(process.env.COMPLEXITY_THRESHOLD || '0.6'),
      agentTimeoutMs: parseInt(process.env.AGENT_TIMEOUT_MS || '30000', 10),
    },
    logLevel: process.env.LOG_LEVEL || 'info',
  };
}
