/// <reference types="vite/client" />

interface ImportMetaEnv {
  /**
   * Origin of the gateway service (services/gateway, port 8080).
   * Leave blank to use the Vite dev proxy configured in vite.config.ts.
   */
  readonly VITE_GATEWAY_URL?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
