// Chromium `BrowserDriver` implementation. Wraps `chrome.debugger.*`
// so each tool implementation can think in terms of typed `send<T>`
// calls and a single attach-once-per-tab cache.
//
// Lifecycle notes:
// * `chrome.debugger.attach()` will fail with "Another debugger is
//   already attached" if invoked twice for the same tab — we track
//   attachments in `attachedTabs` to coalesce. The CDP protocol
//   version pinned here ("1.3") matches what intern uses and what
//   Playwright targets for stable Chrome / Edge / Brave.
// * Closing a tab implicitly detaches; we do not race-clean here.
//   Higher-level code calls `detach()` explicitly when a session
//   stops to keep the "Agent controlling — DevTools" infobar visible
//   only while we actually need it.
// * MV3 service workers can be evicted mid-call. The wrapper
//   propagates `chrome.runtime.lastError` as a thrown Error so
//   callers can decide whether to retry vs. surface an `cdp_failed`.
// * Native JS dialogs (`alert` / `confirm` / `prompt` / `beforeunload`)
//   block CDP until dismissed. We listen for `Page.javascriptDialogOpening`,
//   record the payload for tool results, and auto-accept so automation
//   can continue.

import type { JavaScriptDialogInfo, JavaScriptDialogType } from "@/transport/types";

/**
 * Minimal slice of `chrome.debugger` the rest of the extension
 * depends on. Stays as an explicit interface so vitest can inject a
 * fake without monkey-patching the real `chrome` global.
 */
export interface CdpDebuggerApi {
  attach(target: chrome.debugger.Debuggee, requiredVersion: string): Promise<void>;
  detach(target: chrome.debugger.Debuggee): Promise<void>;
  sendCommand(
    target: chrome.debugger.Debuggee,
    method: string,
    commandParams?: object,
  ): Promise<unknown>;
  /**
   * Fires for every CDP event (`Page.lifecycleEvent`, `DOM.documentUpdated`,
   * …). The first callback argument is the source debuggee; the second
   * is the CDP method name; the third is the payload.
   */
  onEvent: chrome.events.Event<
    (source: chrome.debugger.Debuggee, method: string, params: unknown) => void
  >;
  /**
   * Fires when Chrome unilaterally detaches us — most commonly because
   * the tab navigated to a chrome:// URL or the user clicked
   * "Cancel debugging" on the system infobar.
   */
  onDetach: chrome.events.Event<(source: chrome.debugger.Debuggee, reason: string) => void>;
}

/**
 * Production-backed [`CdpDebuggerApi`]. `onEvent` / `onDetach` are
 * exposed as getters so the *module* loads in vitest (where `chrome`
 * is undefined) — the actual property access only fires when a real
 * caller wires the driver in `background.ts`.
 */
export const chromeDebuggerApi: CdpDebuggerApi = {
  attach: (target, version) => chrome.debugger.attach(target, version),
  detach: (target) => chrome.debugger.detach(target),
  sendCommand: (target, method, commandParams) =>
    chrome.debugger.sendCommand(target, method, commandParams),
  get onEvent() {
    return chrome.debugger.onEvent;
  },
  get onDetach() {
    return chrome.debugger.onDetach;
  },
};

export const CDP_PROTOCOL_VERSION = "1.3";

/** Per-tab monotonic sequence returned by [`ChromiumCdp.dialogCursor`]. */
export type DialogCursor = number;

const MAX_DIALOG_BUFFER = 32;
const MAX_DIALOG_FIELD_LENGTH = 4096;

// --- Console / network observability (bsk 二开: console & network capture) ---
const MAX_CONSOLE_BUFFER = 200;
const MAX_NETWORK_BUFFER = 200;
const MAX_OBS_FIELD_LENGTH = 4096;

/** A captured console / log / uncaught-exception line. */
export interface ConsoleEntry {
  kind: "console" | "exception" | "log";
  level: string;
  text: string;
  url?: string;
  line?: number;
  timestamp?: number;
}

/** A captured network response / failure. */
export interface NetworkEntry {
  method?: string;
  url: string;
  status?: number;
  statusText?: string;
  mimeType?: string;
  resourceType?: string;
  failed?: boolean;
  errorText?: string;
  timestamp?: number;
}

interface ParsedDialogOpening {
  type: JavaScriptDialogType;
  message: string;
  url?: string;
  defaultPrompt?: string;
  hasBrowserHandler?: boolean;
}

/**
 * Wrapper around `chrome.debugger` that owns the "attach once per
 * tabId" cache and exposes typed `send<T>()`.
 */
export class ChromiumCdp {
  private readonly api: CdpDebuggerApi;
  private readonly attachedTabs = new Set<number>();
  private readonly attachInFlight = new Map<number, Promise<void>>();
  private readonly tabOwners = new Map<number, Set<string>>();
  private readonly dialogBuffers = new Map<number, JavaScriptDialogInfo[]>();
  private readonly dialogSequences = new Map<number, number>();
  private readonly consoleBuffers = new Map<number, ConsoleEntry[]>();
  private readonly networkBuffers = new Map<number, NetworkEntry[]>();
  private readonly netRequestMeta = new Map<string, { url: string; method?: string }>();
  private detachSubscription: { dispose(): void } | null = null;
  private dialogSubscription: { dispose(): void } | null = null;
  private obsSubscription: { dispose(): void } | null = null;

  constructor(api: CdpDebuggerApi = chromeDebuggerApi) {
    this.api = api;
    this.bindAutoDetach();
    this.bindDialogHandler();
    this.bindObservabilityHandler();
  }

  /** Attach to `tabId` if we haven't already in this driver. */
  async ensureAttached(tabId: number): Promise<void> {
    if (this.attachedTabs.has(tabId)) return;
    const existing = this.attachInFlight.get(tabId);
    if (existing) {
      await existing;
      return;
    }
    const attach = (async () => {
      await this.api.attach({ tabId }, CDP_PROTOCOL_VERSION);
      await this.enablePageDomain(tabId);
      await this.enableObservabilityDomains(tabId);
      this.attachedTabs.add(tabId);
    })()
      .catch((err) => {
        // Chrome surfaces "Another debugger is already attached" when
        // (rare) the user opened DevTools on the same tab. Don't swallow
        // — let the caller decide how to surface it.
        throw normalizeError(err);
      })
      .finally(() => {
        this.attachInFlight.delete(tabId);
      });
    this.attachInFlight.set(tabId, attach);
    await attach;
  }

  /**
   * Send a CDP command and decode the result as `T`. Throws on any
   * `chrome.runtime.lastError`.
   */
  async send<T = unknown>(tabId: number, method: string, params?: object): Promise<T> {
    if (!this.attachedTabs.has(tabId)) {
      await this.ensureAttached(tabId);
    }
    try {
      const result = await this.api.sendCommand({ tabId }, method, params ?? {});
      return result as T;
    } catch (err) {
      throw normalizeError(err);
    }
  }

  /** Return a cursor marking the current dialog sequence for `tabId`. */
  dialogCursor(tabId: number): DialogCursor {
    return this.dialogSequences.get(tabId) ?? 0;
  }

  /** Dialogs observed on `tabId` with sequence strictly greater than `cursor`. */
  dialogsSince(tabId: number, cursor: DialogCursor): JavaScriptDialogInfo[] {
    const buf = this.dialogBuffers.get(tabId) ?? [];
    return buf.filter((entry) => entry.sequence > cursor);
  }

  /** Console / log / exception lines buffered for `tabId` since attach. */
  consoleEntries(tabId: number): ConsoleEntry[] {
    return [...(this.consoleBuffers.get(tabId) ?? [])];
  }

  /** Network responses / failures buffered for `tabId` since attach. */
  networkEntries(tabId: number): NetworkEntry[] {
    return [...(this.networkBuffers.get(tabId) ?? [])];
  }

  /** Detach if attached; never throws. */
  async detach(tabId: number): Promise<void> {
    this.attachInFlight.delete(tabId);
    if (!this.attachedTabs.has(tabId)) return;
    this.attachedTabs.delete(tabId);
    this.clearDialogState(tabId);
    this.clearObservability(tabId);
    try {
      await this.api.detach({ tabId });
    } catch (err) {
      // Tab may already be gone — Chrome auto-detaches on close. Log
      // at debug so production builds aren't noisy.
      console.debug("[bsk cdp] detach failed (likely tab already closed)", err);
    }
  }

  /** True iff `ensureAttached(tabId)` has succeeded since the last detach. */
  isAttached(tabId: number): boolean {
    return this.attachedTabs.has(tabId);
  }

  /** Remember that `sessionId` used CDP on `tabId` so stop can detach it. */
  trackSessionTab(sessionId: string, tabId: number): void {
    const owners = this.tabOwners.get(tabId) ?? new Set<string>();
    owners.add(sessionId);
    this.tabOwners.set(tabId, owners);
  }

  /** Subscribe to all CDP events. Returned disposable removes the listener. */
  onEvent(handler: (source: chrome.debugger.Debuggee, method: string, params: unknown) => void): {
    dispose(): void;
  } {
    this.api.onEvent.addListener(handler);
    return {
      dispose: () => this.api.onEvent.removeListener(handler),
    };
  }

  /** Best-effort detach of every cached tab. Used on session.stop. */
  async detachAll(): Promise<void> {
    const tabs = Array.from(this.attachedTabs);
    this.attachInFlight.clear();
    this.tabOwners.clear();
    this.attachedTabs.clear();
    this.dialogBuffers.clear();
    this.dialogSequences.clear();
    this.consoleBuffers.clear();
    this.networkBuffers.clear();
    this.netRequestMeta.clear();
    await Promise.all(
      tabs.map(async (tabId) => {
        try {
          await this.api.detach({ tabId });
        } catch (err) {
          console.debug("[bsk cdp] detachAll: tab already gone", { tabId, err });
        }
      }),
    );
  }

  private async enablePageDomain(tabId: number): Promise<void> {
    await this.api.sendCommand({ tabId }, "Page.enable", {});
  }

  /**
   * Enable the CDP domains needed for console / network capture. Each is
   * best-effort: a restricted page that rejects `Network.enable` must not
   * break attach or the existing Page/dialog flow.
   */
  private async enableObservabilityDomains(tabId: number): Promise<void> {
    for (const method of ["Runtime.enable", "Log.enable", "Network.enable"]) {
      try {
        await this.api.sendCommand({ tabId }, method, {});
      } catch (err) {
        console.debug("[bsk cdp] observability enable failed", { tabId, method, err });
      }
    }
  }

  private bindObservabilityHandler(): void {
    if (this.obsSubscription) return;
    const listener = (source: chrome.debugger.Debuggee, method: string, params: unknown) => {
      const tabId = source.tabId;
      if (typeof tabId !== "number") return;
      switch (method) {
        case "Runtime.consoleAPICalled":
          this.appendConsole(tabId, parseConsoleApi(params));
          break;
        case "Runtime.exceptionThrown":
          this.appendConsole(tabId, parseException(params));
          break;
        case "Log.entryAdded":
          this.appendConsole(tabId, parseLogEntry(params));
          break;
        case "Network.requestWillBeSent":
          rememberRequest(this.netRequestMeta, params);
          break;
        case "Network.responseReceived":
          this.appendNetwork(tabId, parseResponse(this.netRequestMeta, params));
          break;
        case "Network.loadingFailed":
          this.appendNetwork(tabId, parseLoadingFailed(this.netRequestMeta, params));
          break;
      }
    };
    this.api.onEvent.addListener(listener);
    this.obsSubscription = {
      dispose: () => this.api.onEvent.removeListener(listener),
    };
  }

  private appendConsole(tabId: number, entry: ConsoleEntry | null): void {
    if (!entry) return;
    const buf = this.consoleBuffers.get(tabId) ?? [];
    buf.push(entry);
    while (buf.length > MAX_CONSOLE_BUFFER) buf.shift();
    this.consoleBuffers.set(tabId, buf);
  }

  private appendNetwork(tabId: number, entry: NetworkEntry | null): void {
    if (!entry) return;
    const buf = this.networkBuffers.get(tabId) ?? [];
    buf.push(entry);
    while (buf.length > MAX_NETWORK_BUFFER) buf.shift();
    this.networkBuffers.set(tabId, buf);
  }

  private clearObservability(tabId: number): void {
    this.consoleBuffers.delete(tabId);
    this.networkBuffers.delete(tabId);
  }

  private bindDialogHandler(): void {
    if (this.dialogSubscription) return;
    const listener = (source: chrome.debugger.Debuggee, method: string, params: unknown) => {
      if (method !== "Page.javascriptDialogOpening") return;
      const tabId = source.tabId;
      if (typeof tabId !== "number") return;
      void this.onJavaScriptDialogOpening(tabId, params);
    };
    this.api.onEvent.addListener(listener);
    this.dialogSubscription = {
      dispose: () => this.api.onEvent.removeListener(listener),
    };
  }

  private async onJavaScriptDialogOpening(tabId: number, params: unknown): Promise<void> {
    const parsed = parseDialogOpeningParams(params);
    try {
      const handleParams: { accept: boolean; promptText?: string } = { accept: true };
      if (parsed.type === "prompt") {
        handleParams.promptText = parsed.defaultPrompt ?? "";
      }
      await this.api.sendCommand({ tabId }, "Page.handleJavaScriptDialog", handleParams);
      const sequence = (this.dialogSequences.get(tabId) ?? 0) + 1;
      this.dialogSequences.set(tabId, sequence);
      this.appendDialog(tabId, {
        tab_id: tabId,
        type: parsed.type,
        message: parsed.message,
        url: parsed.url,
        default_prompt: parsed.defaultPrompt,
        has_browser_handler: parsed.hasBrowserHandler,
        handled: "accepted",
        sequence,
      });
    } catch (err) {
      console.debug("[bsk cdp] Page.handleJavaScriptDialog failed", { tabId, err });
    }
  }

  private appendDialog(tabId: number, entry: JavaScriptDialogInfo): void {
    const buf = this.dialogBuffers.get(tabId) ?? [];
    buf.push(entry);
    while (buf.length > MAX_DIALOG_BUFFER) {
      buf.shift();
    }
    this.dialogBuffers.set(tabId, buf);
  }

  private clearDialogState(tabId: number): void {
    this.dialogBuffers.delete(tabId);
    this.dialogSequences.delete(tabId);
  }

  private bindAutoDetach(): void {
    if (this.detachSubscription) return;
    const listener = (source: chrome.debugger.Debuggee, _reason: string) => {
      if (typeof source.tabId === "number") {
        this.attachedTabs.delete(source.tabId);
        this.attachInFlight.delete(source.tabId);
        this.tabOwners.delete(source.tabId);
        this.clearDialogState(source.tabId);
        this.clearObservability(source.tabId);
      }
    };
    this.api.onDetach.addListener(listener);
    this.detachSubscription = {
      dispose: () => this.api.onDetach.removeListener(listener),
    };
  }

  /** Remove internal Chrome event listeners; tests and SW teardown call this. */
  dispose(): void {
    this.detachSubscription?.dispose();
    this.detachSubscription = null;
    this.dialogSubscription?.dispose();
    this.dialogSubscription = null;
    this.obsSubscription?.dispose();
    this.obsSubscription = null;
  }

  /** Detach tabs only when no other live session has claimed them. */
  async detachSession(sessionId: string): Promise<void> {
    const tabsToDetach: number[] = [];
    for (const [tabId, owners] of this.tabOwners) {
      owners.delete(sessionId);
      if (owners.size === 0) {
        this.tabOwners.delete(tabId);
        tabsToDetach.push(tabId);
      }
    }
    await Promise.all(tabsToDetach.map((tabId) => this.detach(tabId)));
  }
}

function parseDialogOpeningParams(params: unknown): ParsedDialogOpening {
  const raw = (params ?? {}) as Record<string, unknown>;
  const type = normalizeDialogType(raw.type);
  const message = truncateDialogField(typeof raw.message === "string" ? raw.message : "");
  const url = typeof raw.url === "string" ? truncateDialogField(raw.url) : undefined;
  const defaultPrompt =
    typeof raw.defaultPrompt === "string" ? truncateDialogField(raw.defaultPrompt) : undefined;
  const hasBrowserHandler =
    typeof raw.hasBrowserHandler === "boolean" ? raw.hasBrowserHandler : undefined;
  return { type, message, url, defaultPrompt, hasBrowserHandler };
}

function normalizeDialogType(value: unknown): JavaScriptDialogType {
  switch (value) {
    case "alert":
    case "confirm":
    case "prompt":
    case "beforeunload":
      return value;
    default:
      return "alert";
  }
}

function truncateDialogField(value: string): string {
  if (value.length <= MAX_DIALOG_FIELD_LENGTH) return value;
  return `${value.slice(0, MAX_DIALOG_FIELD_LENGTH)}... [truncated]`;
}

function normalizeError(err: unknown): Error {
  if (err instanceof Error) return err;
  if (typeof err === "string") return new Error(err);
  if (err && typeof err === "object" && "message" in err) {
    return new Error(String((err as { message: unknown }).message));
  }
  return new Error("unknown chrome.debugger error");
}

// --- Console / network event parsers (best-effort, defensive casts) ---------

function obsField(value: unknown): string {
  const s = typeof value === "string" ? value : String(value ?? "");
  return s.length <= MAX_OBS_FIELD_LENGTH ? s : `${s.slice(0, MAX_OBS_FIELD_LENGTH)}... [truncated]`;
}

function remoteArgToText(arg: unknown): string {
  const a = (arg ?? {}) as Record<string, unknown>;
  if ("value" in a && a.value !== undefined) return String(a.value);
  if (typeof a.description === "string") return a.description;
  if (typeof a.unserializableValue === "string") return a.unserializableValue;
  if (typeof a.type === "string") return `[${a.type}]`;
  return "";
}

/** `Runtime.consoleAPICalled` → ConsoleEntry. */
function parseConsoleApi(params: unknown): ConsoleEntry | null {
  const p = (params ?? {}) as Record<string, unknown>;
  const args = Array.isArray(p.args) ? p.args : [];
  const text = obsField(args.map(remoteArgToText).filter(Boolean).join(" "));
  const stack = (p.stackTrace ?? {}) as Record<string, unknown>;
  const frames = Array.isArray(stack.callFrames) ? (stack.callFrames as Record<string, unknown>[]) : [];
  const top = frames[0];
  return {
    kind: "console",
    level: typeof p.type === "string" ? p.type : "log",
    text,
    url: top && typeof top.url === "string" ? obsField(top.url) : undefined,
    line: top && typeof top.lineNumber === "number" ? top.lineNumber + 1 : undefined,
    timestamp: typeof p.timestamp === "number" ? p.timestamp : undefined,
  };
}

/** `Runtime.exceptionThrown` → ConsoleEntry (level=error). */
function parseException(params: unknown): ConsoleEntry | null {
  const p = (params ?? {}) as Record<string, unknown>;
  const details = (p.exceptionDetails ?? {}) as Record<string, unknown>;
  const exception = (details.exception ?? {}) as Record<string, unknown>;
  const text =
    (typeof exception.description === "string" && exception.description) ||
    (typeof details.text === "string" && details.text) ||
    "Uncaught (unknown error)";
  return {
    kind: "exception",
    level: "error",
    text: obsField(text),
    url: typeof details.url === "string" ? obsField(details.url) : undefined,
    line: typeof details.lineNumber === "number" ? details.lineNumber + 1 : undefined,
    timestamp: typeof p.timestamp === "number" ? p.timestamp : undefined,
  };
}

/** `Log.entryAdded` → ConsoleEntry (engine-level: network errors, CSP, …). */
function parseLogEntry(params: unknown): ConsoleEntry | null {
  const p = (params ?? {}) as Record<string, unknown>;
  const entry = (p.entry ?? {}) as Record<string, unknown>;
  return {
    kind: "log",
    level: typeof entry.level === "string" ? entry.level : "info",
    text: obsField(typeof entry.text === "string" ? entry.text : ""),
    url: typeof entry.url === "string" ? obsField(entry.url) : undefined,
    line: typeof entry.lineNumber === "number" ? entry.lineNumber : undefined,
    timestamp: typeof entry.timestamp === "number" ? entry.timestamp : undefined,
  };
}

function rememberRequest(
  meta: Map<string, { url: string; method?: string }>,
  params: unknown,
): void {
  const p = (params ?? {}) as Record<string, unknown>;
  const id = typeof p.requestId === "string" ? p.requestId : undefined;
  const req = (p.request ?? {}) as Record<string, unknown>;
  if (!id || typeof req.url !== "string") return;
  meta.set(id, { url: req.url, method: typeof req.method === "string" ? req.method : undefined });
  if (meta.size > 1024) {
    // bound the correlation map — drop oldest insertion
    const first = meta.keys().next().value;
    if (first !== undefined) meta.delete(first);
  }
}

/** `Network.responseReceived` → NetworkEntry. */
function parseResponse(
  meta: Map<string, { url: string; method?: string }>,
  params: unknown,
): NetworkEntry | null {
  const p = (params ?? {}) as Record<string, unknown>;
  const id = typeof p.requestId === "string" ? p.requestId : undefined;
  const resp = (p.response ?? {}) as Record<string, unknown>;
  const url = typeof resp.url === "string" ? resp.url : (id ? meta.get(id)?.url : undefined);
  if (!url) return null;
  return {
    method: id ? meta.get(id)?.method : undefined,
    url: obsField(url),
    status: typeof resp.status === "number" ? resp.status : undefined,
    statusText: typeof resp.statusText === "string" ? obsField(resp.statusText) : undefined,
    mimeType: typeof resp.mimeType === "string" ? resp.mimeType : undefined,
    resourceType: typeof p.type === "string" ? p.type : undefined,
    timestamp: typeof p.timestamp === "number" ? p.timestamp : undefined,
  };
}

/** `Network.loadingFailed` → NetworkEntry (failed=true). */
function parseLoadingFailed(
  meta: Map<string, { url: string; method?: string }>,
  params: unknown,
): NetworkEntry | null {
  const p = (params ?? {}) as Record<string, unknown>;
  const id = typeof p.requestId === "string" ? p.requestId : undefined;
  const info = id ? meta.get(id) : undefined;
  return {
    method: info?.method,
    url: obsField(info?.url ?? "(unknown)"),
    failed: true,
    errorText: typeof p.errorText === "string" ? obsField(p.errorText) : undefined,
    resourceType: typeof p.type === "string" ? p.type : undefined,
    timestamp: typeof p.timestamp === "number" ? p.timestamp : undefined,
  };
}
