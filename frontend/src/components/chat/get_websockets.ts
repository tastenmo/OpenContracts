/**
 * Get WebSocket URL for document queries.
 * @param documentId - Document identifier.
 * @param token - Authentication token from the user session.
 * @param conversationId - (Optional) If provided, the conversation id to load from.
 * @returns WebSocket URL with necessary query parameters.
 */
export function getDocumentQueryWebSocket(
  documentId: string,
  token: string,
  conversationId?: string,
  corpusId?: string
): string {
  // Use environment variables or fallback to window.location for production
  const wsBaseUrl =
    process.env.REACT_APP_WS_URL ||
    process.env.REACT_APP_API_URL ||
    `${window.location.protocol === "https:" ? "wss" : "ws"}://${
      window.location.host
    }`;

  const normalizedBaseUrl = wsBaseUrl
    .replace(/\/+$/, "")
    .replace(/^http/, "ws")
    .replace(/^https/, "wss");

  let url = `${normalizedBaseUrl}/ws/document/${encodeURIComponent(
    documentId
  )}/query/`;

  if (corpusId) {
    url += `corpus/${encodeURIComponent(corpusId)}/`;
  }

  const params: string[] = [];

  if (conversationId) {
    params.push(
      `load_from_conversation_id=${encodeURIComponent(conversationId)}`
    );
  }

  if (token) {
    params.push(`token=${encodeURIComponent(token)}`);
  }

  if (params.length > 0) {
    url += `?${params.join("&")}`;
  }

  return url;
}

/**
 * Get WebSocket URL for corpus queries.
 * @param corpusId - Corpus identifier.
 * @param token - Authentication token from the user session.
 * @returns WebSocket URL with necessary query parameters.
 */
export function getCorpusQueryWebSocket(
  corpusId: string,
  token: string
): string {
  // Use environment variables or fallback to window.location for production
  const wsBaseUrl =
    process.env.REACT_APP_WS_URL ||
    process.env.REACT_APP_API_URL ||
    `${window.location.protocol === "https:" ? "wss" : "ws"}://${
      window.location.host
    }`;

  const normalizedBaseUrl = wsBaseUrl
    .replace(/\/+$/, "")
    .replace(/^http/, "ws")
    .replace(/^https/, "wss");

  let url = `${normalizedBaseUrl}/ws/corpus/${encodeURIComponent(
    corpusId
  )}/query/`;

  const params: string[] = [];

  if (token) {
    params.push(`token=${encodeURIComponent(token)}`);
  }

  if (params.length > 0) {
    url += `?${params.join("&")}`;
  }

  return url;
}
